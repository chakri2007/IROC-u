# Published per seed after retrieval
# Topic: /semantic_retrieval/results

std_msgs/Header header

# Seed identifier
string seed_name

# Whether any match cleared the similarity threshold
bool has_match

# Number of valid matches found (0 if has_match is false)
int32 match_count

# Per-match arrays — all same length (match_count)
int32[]   frame_indices          # frame index in original rosbag
float64[] timestamps_sec         # ROS timestamp (seconds) of that frame

# VSLAM pose at time of matched frame (from /visual_slam/tracking/odometry)
# Flattened: [x0,y0,z0, x1,y1,z1, ...] — stride 3 per match
# Empty if odometry was not in the rosbag or pose gap exceeded 200ms
float64[] positions_xyz          # length = match_count * 3

# Orientation quaternions [qx,qy,qz,qw] per match — stride 4
float64[] orientations_xyzw     # length = match_count * 4

# Pose valid flag per match (true = pose was within 200ms of frame timestamp)
bool[]    pose_valid

float32[] similarity_scores      # cosine similarity [0.0 - 1.0]

# HD frames — one sensor_msgs/Image per match
sensor_msgs/Image[] hd_frames

# Threshold used (so GUI can display confidence bar)
float32 similarity_threshold
