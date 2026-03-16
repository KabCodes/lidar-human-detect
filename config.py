# ─── Network ─────────────────────────────────────────────────────────────────
LIDAR_IP   = "192.168.1.62"
LIDAR_PORT = 6101
HOST_IP    = "192.168.1.2"
HOST_PORT  = 6201

# ─── Visualiser ──────────────────────────────────────────────────────────────
WINDOW_WIDTH   = 1280
WINDOW_HEIGHT  = 720
MAX_PACKETS    = 600

# ─── Background map ──────────────────────────────────────────────────────────
BG_MAP_PATH      = "maps/background.npy"
BG_SCAN_DURATION = 30
BG_TOLERANCE     = 0.2    # metres

# ─── Detection region (metres) ───────────────────────────────────────────────
REGION_X_MIN = -5.0
REGION_X_MAX =  5.0
REGION_Y_MIN = -5.0
REGION_Y_MAX =  5.0
REGION_Z_MIN =  0.05
REGION_Z_MAX =  2.3

# ─── RANSAC plane removal ────────────────────────────────────────────────────
PLANE_DISTANCE_THRESHOLD = 0.01
PLANE_RANSAC_N           = 7
PLANE_NUM_ITERS          = 500
PLANE_MIN_POINTS         = 200
PLANE_MAX_PLANES         = 6

# ─── Voxel clustering ────────────────────────────────────────────────────────
VOXEL_SIZE            = 0.10   # increased from 0.08 — coarser grid, fewer body-part splits
VOXEL_NEIGHBOR_FACTOR = 2.5    # increased from 1.8 — connects torso/legs/head into one cluster
MIN_CLUSTER_POINTS    = 50     # discard tiny clusters (reflections tend to be sparse)

# ─── Human size limits (metres) ──────────────────────────────────────────────
HUMAN_HEIGHT_MIN = 0.8
HUMAN_HEIGHT_MAX = 2.2
HUMAN_WIDTH_MIN  = 0.15
HUMAN_WIDTH_MAX  = 0.9

# ─── Classifier thresholds ───────────────────────────────────────────────────
SHADOW_RADIUS  = 0.5    # slightly larger radius
SHADOW_MIN_PTS = 50     # increased from 20 — reflections won't have this many real points nearby
MIN_VOTES      = 2

# ─── Reflection filter ───────────────────────────────────────────────────────
REFLECTION_FILTER_ENABLED = True
REFLECTION_RADIUS         = 0.1   # metres — points with no neighbour within this are noise
REFLECTION_MIN_NEIGHBOURS = 3     # minimum neighbours to keep a point

# ─── Mounting mode ───────────────────────────────────────────────────────────
MOUNT_MODE = "floor"

# ─── Human merge radius ───────────────────────────────────────────────────────
# Detections closer than this are merged into one — same person, split cluster
HUMAN_MERGE_RADIUS = 0.8   # metres

# ─── Scan radius cap ──────────────────────────────────────────────────────────
MAX_SCAN_RADIUS = 3.0   # metres — points beyond this are not displayed or detected