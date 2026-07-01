"""
archive.py — mission rosbag + embedding archive for the Anveshan GCS.
================================================================================
Owns WHERE mission data lives and HOW it is named, so the orchestrator, the
indexing trigger, and the (future) replay dropdown all agree.

THE MODEL (see config.json → "rosbag" + "archive")
--------------------------------------------------------------------------------
  live_folder/                         one CURRENT mission bag lives here
      rosbag_<STAMP>/                  ← ros2 bag record -o rosbag_<STAMP>
  archive_folder/
      rosbag_archives/                 previous bags moved here on re-record
          rosbag_<STAMP>/
      embeddings_archives/             every embedding .pt lives here, forever
          embddg_<STAMP>.pt

  STAMP = "%H%M_%d%m%Y"  → e.g. 1430_01072026   (config: archive.stamp_format)

  A "mission" is the PAIR (rosbag_<STAMP>, embddg_<STAMP>.pt) — the embedding
  inherits the bag's stamp, so picking a date+time gives you both halves back.

AUTONOMOUS RUN (the default, no operator choice needed)
  1. record  → rosbag_<STAMP> in live_folder      (orchestrator "rosbag" step)
  2. process → read that live bag, write embddg_<STAMP>.pt to embeddings_archives
               and the retriever loads it                (indexing trigger + "semantic" step)
  Live *video* never comes from here — that is the HTTP MJPEG stream. Only the
  IMAGE-PROCESSING pipeline uses the rosbag.

REPLAY (add-on, manual) — revisit a past mission
  list_missions() feeds a latest-first dropdown; selecting one points the
  semantic pipeline at that archived (bag, .pt) pair instead of the live folder.

LOW-LATENCY / FAILSAFE ARCHIVING  (bags are 1–10 GB!)
  Keep live_folder and archive_folder ON THE SAME FILESYSTEM (the defaults share
  ~/IROC) so moving a bag is an atomic rename — instant, no GB copy. The move
  also runs in a BACKGROUND thread and never raises into the caller: a failed
  move logs and leaves the bag in place rather than blocking bring-up.

--------------------------------------------------------------------------------
FOR LATER EDITS
  * Change naming?           → archive.rosbag_prefix / embedding_prefix / stamp_format
  * Change locations?        → rosbag.live_folder / archive.folder (+ *_subdir)
  * Different pairing key?   → _stamp_of() + list_missions() (parsing the stamp back)
  No ROS / heavy imports here on purpose — safe to import anywhere.
"""

import os
import shutil
import threading
from datetime import datetime
from typing import Callable, Dict, List, Optional

import config_store


# ── path + naming helpers (all config-driven) ────────────────────────────────
def _expand(p) -> str:
    return os.path.expanduser(str(p)) if p else ""


def _arch(cfg: Dict) -> Dict:
    return cfg.get("archive", {}) or {}


def live_folder(cfg: Dict) -> str:
    return _expand(cfg.get("rosbag", {}).get("live_folder", "~/IROC/live_rosbags"))


def archive_root(cfg: Dict) -> str:
    return _expand(_arch(cfg).get("folder", "~/IROC/archive"))


def rosbag_archive_dir(cfg: Dict) -> str:
    return os.path.join(archive_root(cfg), _arch(cfg).get("rosbag_subdir", "rosbag_archives"))


def embeddings_archive_dir(cfg: Dict) -> str:
    return os.path.join(archive_root(cfg), _arch(cfg).get("embeddings_subdir", "embeddings_archives"))


def _rosbag_prefix(cfg: Dict) -> str:
    return _arch(cfg).get("rosbag_prefix", "rosbag_")


def _embedding_prefix(cfg: Dict) -> str:
    return _arch(cfg).get("embedding_prefix", "embddg_")


def _stamp_format(cfg: Dict) -> str:
    return _arch(cfg).get("stamp_format", "%H%M_%d%m%Y")


def new_stamp(cfg: Dict) -> str:
    """Fresh STAMP for 'now' (e.g. 1430_01072026)."""
    return datetime.now().strftime(_stamp_format(cfg))


def rosbag_name(cfg: Dict, stamp: str) -> str:
    return f"{_rosbag_prefix(cfg)}{stamp}"


def embedding_name(cfg: Dict, stamp: str) -> str:
    return f"{_embedding_prefix(cfg)}{stamp}.pt"


def _stamp_of(name: str, prefix: str) -> str:
    """'rosbag_1430_01072026' + 'rosbag_' → '1430_01072026' (strip a trailing .pt too)."""
    base = os.path.basename(name.rstrip("/\\"))
    if base.endswith(".pt"):
        base = base[:-3]
    return base[len(prefix):] if base.startswith(prefix) else base


def ensure_dirs(cfg: Dict) -> None:
    """Create live + archive folders if missing (safe to call every run)."""
    for d in (live_folder(cfg), rosbag_archive_dir(cfg), embeddings_archive_dir(cfg)):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass


# ── discovering what exists ──────────────────────────────────────────────────
def _bag_dirs(folder: str, prefix: str) -> List[str]:
    """Directories in `folder` that look like a ros2 bag we named (prefix match)."""
    out = []
    try:
        for n in os.listdir(folder):
            p = os.path.join(folder, n)
            if os.path.isdir(p) and n.startswith(prefix):
                out.append(p)
    except OSError:
        pass
    return out


def find_live_bag(cfg: Dict) -> Optional[str]:
    """The current mission bag in live_folder (newest, prefers a finalized bag)."""
    bags = _bag_dirs(live_folder(cfg), _rosbag_prefix(cfg))
    if not bags:
        return None
    # a finalized ros2 bag has metadata.yaml; prefer those, newest first
    finalized = [b for b in bags if os.path.exists(os.path.join(b, "metadata.yaml"))]
    pool = finalized or bags
    return max(pool, key=lambda b: os.path.getmtime(b))


# ── the failsafe, low-latency archival move ──────────────────────────────────
def _unique_dest(dest: str) -> str:
    """Avoid clobbering an existing archive entry (same-minute stamp collision)."""
    if not os.path.exists(dest):
        return dest
    i = 2
    while os.path.exists(f"{dest}_{i}"):
        i += 1
    return f"{dest}_{i}"


def archive_existing_bags(cfg: Dict,
                          logger: Optional[Callable[[str, str, str], None]] = None,
                          background: bool = True) -> None:
    """Move every bag currently in live_folder into rosbag_archives.

    Call this JUST BEFORE recording a new bag ("about to re-record"). Same-fs =
    instant rename; otherwise a slow copy runs off-thread so bring-up never waits.
    Never raises: a failed move is logged and the bag left in place.
    """
    ensure_dirs(cfg)
    bags = _bag_dirs(live_folder(cfg), _rosbag_prefix(cfg))
    if not bags:
        return
    dest_root = rosbag_archive_dir(cfg)

    def _emit(msg: str):
        if logger:
            try:
                logger("SYS", "archive", msg)
            except Exception:
                pass

    def _move_all():
        for src in bags:
            dest = _unique_dest(os.path.join(dest_root, os.path.basename(src)))
            try:
                shutil.move(src, dest)          # rename if same fs, else copy+del
                _emit(f"archived {os.path.basename(src)} → rosbag_archives")
            except Exception as e:
                _emit(f"archive FAILED for {os.path.basename(src)}: {e} (left in place)")

    if background:
        threading.Thread(target=_move_all, daemon=True).start()
    else:
        _move_all()


# ── autonomous path resolution ───────────────────────────────────────────────
def autonomous_paths(cfg: Dict) -> Dict[str, Optional[str]]:
    """Resolve the (rosbag, embedding) paths for an AUTONOMOUS run.

    rosbag  = the live bag if one exists (else None → nothing to process yet),
    embedding = embeddings_archives/embddg_<STAMP>.pt, STAMP inherited from the
                bag name so the pair matches; falls back to 'now' if unparseable.
    """
    ensure_dirs(cfg)
    bag = find_live_bag(cfg)
    if bag:
        stamp = _stamp_of(bag, _rosbag_prefix(cfg))
    else:
        stamp = new_stamp(cfg)
    db = os.path.join(embeddings_archive_dir(cfg), embedding_name(cfg, stamp))
    return {"rosbag_path": bag, "db_path": db, "stamp": stamp}


# ── mission listing (feeds the replay dropdown, latest-first) ────────────────
def _parse_stamp(cfg: Dict, stamp: str) -> datetime:
    try:
        return datetime.strptime(stamp, _stamp_format(cfg))
    except (ValueError, TypeError):
        return datetime.min


def list_missions(cfg: Dict) -> List[Dict]:
    """All known missions as (stamp, bag, embedding) rows, newest first.

    Scans live_folder + rosbag_archives for bags and embeddings_archives for .pt,
    then joins them by STAMP. A row may have only a bag or only an embedding.
    """
    ensure_dirs(cfg)
    rp, ep = _rosbag_prefix(cfg), _embedding_prefix(cfg)
    rows: Dict[str, Dict] = {}

    def _add(stamp, **kw):
        rows.setdefault(stamp, {"stamp": stamp, "rosbag_path": None,
                                "embedding_path": None, "location": None})
        rows[stamp].update({k: v for k, v in kw.items() if v is not None})

    for b in _bag_dirs(live_folder(cfg), rp):
        _add(_stamp_of(b, rp), rosbag_path=b, location="live")
    for b in _bag_dirs(rosbag_archive_dir(cfg), rp):
        st = _stamp_of(b, rp)
        _add(st, rosbag_path=b)
        if rows[st]["location"] is None:
            rows[st]["location"] = "archive"
    try:
        for n in os.listdir(embeddings_archive_dir(cfg)):
            if n.startswith(ep) and n.endswith(".pt"):
                _add(_stamp_of(n, ep),
                     embedding_path=os.path.join(embeddings_archive_dir(cfg), n))
    except OSError:
        pass

    out = list(rows.values())
    for r in out:
        dt = _parse_stamp(cfg, r["stamp"])
        r["when"] = dt.strftime("%H:%M · %d/%m/%Y") if dt != datetime.min else r["stamp"]
        r["_sort"] = dt
    out.sort(key=lambda r: r["_sort"], reverse=True)
    for r in out:
        r.pop("_sort", None)
    return out
