# Anveshan GCS — Plain-English Handbook

> The base-station console for **Anveshan / IRoC-U 2026 (ASCEND)** — the screen that
> gets recorded as scored evidence. This doc is for *anyone* on the team: what the
> GUI is, what each piece does, how to run it, and the gotchas we hit. No prior
> knowledge of the codebase assumed.

---

## 1. The 30-second picture

The GUI is a **dashboard you open in a browser**. It has two tabs:

| Tab | What it's for |
|---|---|
| **Controller** | Live flight view — nadir camera feed, attitude, battery, velocity, compass, the SLAM map, and telemetry. Pure observation. |
| **Analysis** | The scored part — proves the drone autonomously found the seed objects. Shows matched frames, similarity scores, and each feature's XYZ position **relative to base**. |

There's also a hidden **OVERRIDES** drawer (top-right) with emergency flight commands
(START / ABORT / HOLD / …), kept out of the way so the console reads as observation-first.

Everything on screen is **real or honestly empty** — no fake numbers. If a value
isn't available yet, you see `—` or "awaiting …", never a made-up reading.

---

## 2. Who runs what (the machines)

Three machines, three jobs:

```
┌──────────────────────┐     Wi-Fi / LAN      ┌──────────────────────────┐
│  GROUND LAPTOP       │  <---------------->  │  DRONE (Jetson Orin Nano)│
│  • GUI backend :5000 │     ROS 2 (same      │  • semantic nodes        │
│  • GUI frontend:8080 │      ROS_DOMAIN_ID)  │  • rosbag of the flight  │
│  (you watch here)    │                      │  • flight stack (team)   │
└──────────┬───────────┘                      └──────────────────────────┘
           │ TCP :55555 (docking — planned)
           ▼
┌──────────────────────┐
│  DOCKING STATION RPi │  → Arduino Mega (dock motors) + charging module
│  • gcs.py            │
└──────────────────────┘
```

- **Ground laptop** runs the *web* part (backend + frontend). This is the screen you record.
- **Jetson (drone)** runs the *ROS nodes* that do the heavy ML and hold the rosbag.
- **Docking-station RPi** runs `gcs.py` and physically docks + charges the drone. *(Integration into the GUI is the next milestone — see §8.)*

For bench testing, **all of it can run on the Jetson alone** — that's how we've been validating.

---

## 3. The semantic pipeline, in plain words

The rulebook task: the drone surveys an arena, and we must prove it **found specific
seed objects** and report **where** they are. Here's the flow:

1. **Seeds** — before launch, the operator uploads reference images of the objects to find ("seeds").
2. **Index** — after the drone lands, we replay the flight's rosbag and run every camera
   frame through **DINOv2** (an image-understanding model), turning each frame into a
   fingerprint. This builds an "embedding database" (`.pt` file).
3. **Retrieve** — each seed is fingerprinted the same way, then compared (cosine similarity)
   against every frame. Frames that score high enough are **matches**.
4. **Show** — matches appear in the Analysis tab: the HD frame, the score, the timestamp,
   and the XYZ position (from VSLAM odometry, relative to base).

Three small ROS nodes do this (package `semantic_retrieval`):

| Node | Job |
|---|---|
| `indexer_node` | Reads the rosbag → embeds frames → saves the `.pt` database. Triggered by a service call after docking. |
| `retriever_node` | Loads the DB + seeds → matches → publishes results. Accepts new seeds live. |
| `hd_frame_server` | Serves a single HD frame from the rosbag on demand (so the laptop GUI doesn't need the bag). |

> **Where do the XYZ coords come from?** VSLAM (`/visual_slam/tracking/odometry`) —
> the drone's local position in metres from base. GPS-denied, so it's camera+IMU, not lat/lon.
> If the bag has no odometry, cards honestly show `pose · unavailable`.

---

## 4. How to run it (one command)

Everything the GUI team owns comes up from a single launch file. On the Jetson, with
the Python venv active and the workspace sourced:

```bash
ros2 launch /path/to/iroc_gui/launch/gui_bringup.launch.py \
  rosbag_path:=/home/nidar/Desktop/test_bag31 \
  camera_topic:=/image_raw \
  db_path:=/home/nidar/semantic_db/test_bag31.pt \
  seeds_dir:=/home/nidar/anveshan_seeds \
  with_indexer:=false
```

This starts the retriever + HD-frame server + backend + frontend. Open
`http://localhost:8080`. One `Ctrl-C` stops everything.

### Launch knobs you'll actually use

| Argument | Meaning | Default |
|---|---|---|
| `rosbag_path` | The mission rosbag folder (has `metadata.yaml`). | — |
| `camera_topic` | Camera topic in the bag (e.g. `/image_raw` for test bags). | `/camera/camera/color/image_raw` |
| `db_path` | Where the embedding DB lives. | `~/semantic_db/mission.pt` |
| `seeds_dir` | Folder of seed images loaded at startup. | `~/anveshan_seeds` |
| `threshold` | **Retrieval floor** — `0.0` runs the whole search and lets the GUI threshold live. | `0.0` |
| `display_threshold` | **The GUI's starting threshold** (the on-screen box). Set this for autonomous runs. | `0.57` |
| `top_k` | Max candidate frames per seed in the pool. | `50` |
| `downsample_seed` | `true` if your seeds are full-res photos (downsamples them to match frames). | `false` |
| `with_indexer` | Run the indexer node (off if the DB already exists). | `true` |
| `with_backend` / `with_frontend` / `with_nodes` | Toggle pieces (e.g. Jetson-only vs laptop-only). | `true` |

---

## 5. Features you can use today

- **Live (dynamic) thresholding.** The retriever computes a *pool* of candidate matches;
  the **`match ≥` box** in the Analysis header filters them instantly — drag it down to see
  more, up to tighten. No re-run. Set the starting value from the terminal with
  `display_threshold:=…` (great for autonomous runs); change it live for manual demos.
- **Seed thumbnails.** The Seed References rail shows the actual uploaded image (or reads it
  from the seeds folder when on one box).
- **Remove a seed.** Each seed in the rail has a **×** — removes it everywhere (tells the
  retriever to drop it too).
- **Seed downsample toggle.** `downsample_seed:=true` brings full-res seed photos into the
  same 128×128 domain the frames live in.
- **Emergency overrides.** The OVERRIDES drawer publishes fixed commands on `/gcs/command`
  (START / ABORT / HOLD / RTL / ABORT_DOCK / RECALL). ABORT needs you to type "ABORT" to
  confirm. A `200` means *published*, not *obeyed* — the flight team's drone confirms via
  `/gcs/command_ack` (shown as "drone ack: …"). See `HANDOFF.md`.
- **Export report.** Analysis → Export Report → a CSV of exactly what's shown (respects the
  threshold box), for the rulebook evidence package.

---

## 6. Quick reference — the data contracts

| Direction | Topic / Service | Type | Notes |
|---|---|---|---|
| GUI ← drone | `/semantic_retrieval/results` | `SemanticMatch` | matches per seed (latched) |
| GUI → drone | `/semantic_retrieval/add_seed` | `sensor_msgs/Image` | `frame_id` = seed name |
| GUI → drone | `/semantic_retrieval/remove_seed` | `std_msgs/String` | seed name to drop |
| trigger | `/semantic_retrieval/trigger_indexing` | `TriggerIndexing` srv | start indexing post-land |
| on demand | `/semantic_retrieval/get_hd_frame` | `GetHdFrame` srv | one HD frame by index |
| GUI → drone | `/gcs/command` | `std_msgs/String` | emergency vocab, RELIABLE/VOLATILE/depth-10 |
| drone → GUI | `/gcs/command_ack` | `std_msgs/String` | optional ack |
| drone → GUI | `/mavros/*`, `/visual_slam/*`, `/drone/status` | various | telemetry, pose, dock status |

Backend REST highlights (FastAPI, port 5000, interactive docs at `/docs`):
`/api/telemetry`, `/api/semantic_results`, `/api/frame/{i}`, `/api/add_seed`,
`/api/seed_image/{name}`, `DELETE /api/seed/{name}`, `/api/command`, `/api/config`,
`/ws` (10 Hz push), `/video_feed` (MJPEG).

---

## 7. Gotchas we already hit (so you don't)

- **The ML nodes need the venv Python.** `transformers` lives only in the
  `img_p_new` venv. `ros2 run` uses system Python and fails with
  `ModuleNotFoundError: transformers`. The launch file runs them as
  `python3 -m semantic_retrieval.<node>` from the active venv — so **always activate
  the venv before launching**.
- **DINOv2 weights are cached on the Jetson** — the launch sets `HF_HUB_OFFLINE=1`, so no
  internet needed at run time.
- **`python-multipart`** is required by seed upload (it's in `requirements.txt`).
- **Editing a node?** You run the *installed* copy, so after changing
  `semantic_retrieval/...` you must `colcon build --packages-select semantic_retrieval`.
  Editing the backend/frontend just needs a re-pull (no build).
- **Config default paths** point at `/home/user/...` (doesn't exist) — always pass real
  paths via launch args.
- The `rcl_shutdown already called` traceback on Ctrl-C is harmless (a Humble quirk).

---

## 8. What's next — docking + charging (planned)

The docking station (RPi `gcs.py`) docks the drone and charges it with a PID loop. The
plan is to bring this into the GUI: a **green "hacker terminal" panel** streaming the TCP
comms and the live PID charging output, plus an auto state machine (settle → dock → charge,
and re-dock + charge after each landing). Logic first, launch-merge last. See the team
chat / planning notes for the current design and open questions.

---

*Validated end-to-end on the Jetson (build → index → retrieve → HD frames → full GUI) with
`test_bag31`. If something here drifts from the code, the code wins — ping whoever last
touched `iroc_gui/`.*
