# semantic_retrieval — Anveshan seed-matching pipeline

DINOv2 semantic video-frame retrieval for the ASCEND mission. After the drone
lands and docks, this indexes the mission rosbag, matches the operator's seed
images against the survey footage, and reports — per seed — the matching frame
indices, their VSLAM XYZ coordinates (relative to base), a similarity score, and
HD verification imagery on demand.

## Layout (two packages — required by ROS2)

```
semantic_retrieval/
├── semantic_retrieval_interfaces/   # ament_cmake — msg + srv ONLY
│   ├── msg/SemanticMatch.msg        # per-seed result metadata
│   ├── srv/TriggerIndexing.srv      # mission manager → indexer (post-dock)
│   └── srv/GetHdFrame.srv           # GUI → HD frame by index (on demand)
└── semantic_retrieval/              # ament_python — nodes
    ├── semantic_retrieval/
    │   ├── embedding_engine.py      # DINOv2 core (the tested algorithm)
    │   ├── indexer_node.py          # builds embedding DB from the rosbag
    │   ├── retriever_node.py        # matches seeds → publishes SemanticMatch
    │   └── hd_frame_server.py       # serves HD frames from the bag (GetHdFrame)
    ├── config/config.yaml
    └── launch/semantic_retrieval.launch.py
```

Interfaces and nodes are split because a single ament package cannot both
generate `rosidl` messages and ship Python nodes under the same module name.

## Architecture (GUI backend runs on the ground laptop)

The pipeline runs on the **Jetson** (GPU + rosbag). The GUI backend runs on the
**ground laptop**, so it has no rosbag access — HD imagery therefore travels
over ROS **on demand** via the `GetHdFrame` service (compressed JPEG), keeping
the `/semantic_retrieval/results` topic light over WiFi.

```
/semantic_retrieval/trigger_indexing  (srv)   mission/GUI → indexer, after dock
/semantic_retrieval/add_seed          (Image) GUI → retriever, runtime seeds
/semantic_retrieval/results           (SemanticMatch, latched)  retriever → GUI
/semantic_retrieval/status            (String) status log → GUI
/semantic_retrieval/get_hd_frame      (srv)   GUI → HD JPEG by frame_index
```

Indexing runs **in parallel with charging** (fired when the drone docks) with
`sample_rate: 1` (every frame; slow but maximal coverage). A similarity
threshold (0.75) rejects false positives — a seed with no frame above threshold
reports `has_match: false`.

## Build & run on the Jetson (ROS2 Humble)

```bash
# put both packages under a colcon workspace src/
colcon build --packages-select semantic_retrieval_interfaces semantic_retrieval
source install/setup.bash
ros2 launch semantic_retrieval semantic_retrieval.launch.py

# after docking (mission manager or GUI calls this):
ros2 service call /semantic_retrieval/trigger_indexing \
  semantic_retrieval_interfaces/srv/TriggerIndexing \
  "{rosbag_path: '/data/mission_xxx', output_db_path: '/home/user/semantic_db/m.pt', force_reindex: false}"
```

The algorithm core (`embedding_engine.py`) is the team's tested pipeline,
preserved faithfully; only the ROS2 plumbing around it was corrected.
