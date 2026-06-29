"""
embedding_engine.py

Pure-Python embedding core — zero ROS2 dependency.
Both the indexer and retriever import from here.
This keeps all DINOv2 logic in one place.
"""

import time
import torch
import numpy as np
import cv2

from PIL import Image, ImageFilter, ImageEnhance
from transformers import AutoImageProcessor, AutoModel


# ============================================================
# DEVICE SETUP
# ============================================================

def get_device(use_fp16: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # On Jetson Orin Nano CUDA is available but memory is shared with CPU
    # fp16 cuts VRAM roughly in half — keep it on unless debugging
    return device


# ============================================================
# MODEL LOADER (singleton pattern — load once, reuse)
# ============================================================

_processor = None
_model = None
_device = None
_fp16 = None


def load_model(model_name: str, use_fp16: bool = True):
    global _processor, _model, _device, _fp16

    if _model is not None:
        return _processor, _model, _device

    _device = get_device()
    _fp16 = use_fp16 and (_device == "cuda")

    print(f"[EmbeddingEngine] Loading {model_name} on {_device} | fp16={_fp16}")

    _processor = AutoImageProcessor.from_pretrained(model_name)
    _model = AutoModel.from_pretrained(model_name).to(_device)
    _model.eval()

    if _fp16:
        _model = _model.half()

    print("[EmbeddingEngine] Model ready.")
    return _processor, _model, _device


# ============================================================
# PYRAMIDAL GAUSSIAN DOWNSAMPLING
# ============================================================

def pyr_gauss_downsample(frame_bgr: np.ndarray, target_size=(128, 128)) -> np.ndarray:
    """
    Iterative gaussian blur + 2x downsample until near target_size,
    then final resize. Preserves low-frequency structure better than
    direct resize — important for cross-resolution retrieval.
    """
    current = frame_bgr.copy()
    tw, th = target_size

    while True:
        h, w = current.shape[:2]
        if h <= th * 2 or w <= tw * 2:
            break
        current = cv2.GaussianBlur(current, (5, 5), sigmaX=1.0)
        current = cv2.resize(current, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    current = cv2.resize(current, target_size, interpolation=cv2.INTER_AREA)
    return current


# ============================================================
# 6-WAY SPATIAL DECOMPOSITION
# ============================================================

def split_6(frame: np.ndarray):
    """
    Returns [full, top-left, top-right, bottom-left, bottom-right, center]
    Each is a numpy array (RGB or BGR, caller decides before this call).
    """
    h, w = frame.shape[:2]
    return [
        frame,
        frame[:h // 2, :w // 2],
        frame[:h // 2, w // 2:],
        frame[h // 2:, :w // 2],
        frame[h // 2:, w // 2:],
        frame[h // 4:3 * h // 4, w // 4:3 * w // 4],
    ]


# ============================================================
# EMBEDDING
# ============================================================

@torch.no_grad()
def get_embeddings(pil_images: list) -> torch.Tensor:
    """
    Input  : list of PIL.Image
    Output : normalized float32 tensor  (N, D)  on CPU
    """
    processor, model, device = _processor, _model, _device

    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if _fp16:
        inputs = {k: v.half() for k, v in inputs.items()}

    outputs = model(**inputs)
    emb = outputs.last_hidden_state[:, 0, :]          # CLS token
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)

    return emb.float().cpu()


# ============================================================
# EMBED ONE FRAME  (downsampled BGR numpy → 6 embeddings)
# ============================================================

def embed_frame(frame_bgr: np.ndarray, target_size=(128, 128)) -> torch.Tensor:
    """
    Full pipeline for one video frame:
      BGR numpy  →  downsample  →  split_6  →  embed
    Returns tensor of shape (6, D).
    """
    small = pyr_gauss_downsample(frame_bgr, target_size)
    crops = split_6(small)

    pil_images = []
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb))

    emb = get_embeddings(pil_images)    # (6, D)
    return emb


# ============================================================
# EMBED ONE SEED  (PIL image → 1 embedding)
# ============================================================

def embed_seed(pil_image: Image.Image, pyr_downsample: bool = False,
               target_size=(128, 128)) -> torch.Tensor:
    """
    Returns tensor of shape (1, D).

    pyr_downsample:
      False (default) — embed the seed as given. Assumes the operator already
        provides a ~128x128 downsampled crop, matching the indexed-frame domain.
      True — run the seed through the SAME pyramidal Gaussian downsample the
        indexer applies to video frames, bringing a full-resolution seed photo
        into the same low-res/blurry domain before embedding. Use this when seeds
        are real high-res reference images rather than pre-downsampled crops.
    """
    if pyr_downsample:
        bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        small = pyr_gauss_downsample(bgr, target_size)
        pil_image = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    emb = get_embeddings([pil_image])
    return emb


# ============================================================
# COSINE SEARCH  (vectorised)
# ============================================================

def cosine_search(
    seed_emb: torch.Tensor,          # (1, D)
    video_embeddings: torch.Tensor,  # (N, 6, D)
    frame_indices: list,
    timestamps: list,
    top_k: int,
    threshold: float,
    poses: list = None,              # list of pose dicts or None entries
) -> dict:
    """
    Cosine similarity search.
    Since embeddings are L2-normalised, dot product == cosine similarity.

    Returns dict:
        has_match       : bool
        frame_indices   : list[int]   (empty if no match)
        timestamps_sec  : list[float] (empty if no match)
        scores          : list[float] (empty if no match)
        best_score      : float
        all_scores      : np.ndarray  (N,) full score array for all frames
    """
    N, _, D = video_embeddings.shape
    flat = video_embeddings.reshape(-1, D)             # (N*6, D)

    sims = torch.matmul(seed_emb, flat.T)              # (1, N*6)
    sims = sims.reshape(N, 6)
    frame_scores, _ = torch.max(sims, dim=1)           # (N,)  max across 6 crops
    all_scores_np = frame_scores.numpy()

    # --- Apply threshold before top-k ---
    # Mask frames that don't clear the similarity threshold
    valid_mask = frame_scores >= threshold             # (N,) bool

    if not valid_mask.any():
        # Nothing cleared the threshold → publish empty result
        return {
            "has_match":     False,
            "frame_indices": [],
            "timestamps_sec": [],
            "scores":        [],
            "poses":         [],
            "best_score":    float(frame_scores.max().item()),
            "all_scores":    all_scores_np,
        }

    # Zero out invalid frames before top-k so they can't be selected
    masked_scores = frame_scores.clone()
    masked_scores[~valid_mask] = -1.0

    k = min(top_k, int(valid_mask.sum().item()))
    top_scores, top_idx = torch.topk(masked_scores, k=k)

    matched_frames = [frame_indices[i] for i in top_idx.numpy()]
    matched_times  = [timestamps[i]    for i in top_idx.numpy()]
    matched_scores = top_scores.numpy().tolist()
    matched_poses  = (
        [poses[i] for i in top_idx.numpy()]
        if poses is not None
        else [None] * len(matched_frames)
    )

    return {
        "has_match":     True,
        "frame_indices": matched_frames,
        "timestamps_sec": matched_times,
        "scores":        matched_scores,
        "poses":         matched_poses,
        "best_score":    matched_scores[0],
        "all_scores":    all_scores_np,
    }


# ============================================================
# SAVE / LOAD EMBEDDING DATABASE
# ============================================================

def save_db(
    path: str,
    embeddings: torch.Tensor,
    frame_indices: list,
    timestamps: list,
    poses: list = None,
):
    """
    Save embedding database to disk.
    embeddings shape: (N, 6, D)

    poses: optional list of dicts per frame:
        {"x": float, "y": float, "z": float,
         "qx": float, "qy": float, "qz": float, "qw": float}
        Populated from /visual_slam/tracking/odometry when available.
        None entries allowed (frames where odometry had no match).
    """
    torch.save({
        "embeddings":    embeddings,
        "frame_indices": frame_indices,
        "timestamps":    timestamps,
        "poses":         poses or [None] * len(frame_indices),
        "model":         _model.__class__.__name__ if _model else "unknown",
    }, path)
    print(f"[EmbeddingEngine] DB saved → {path}  ({len(frame_indices)} frames)")


def load_db(path: str) -> tuple:
    """
    Returns (embeddings, frame_indices, timestamps, poses)
    embeddings shape: (N, 6, D)
    poses: list of dicts or None entries
    """
    # Security: weights_only=True blocks pickle-RCE from a tampered .pt. The DB
    # holds only tensors + plain containers (lists/dicts/str/None), so it loads fine.
    data = torch.load(path, map_location="cpu", weights_only=True)
    poses = data.get("poses", [None] * len(data["frame_indices"]))
    print(f"[EmbeddingEngine] DB loaded ← {path}  ({len(data['frame_indices'])} frames)")
    return data["embeddings"], data["frame_indices"], data["timestamps"], poses
