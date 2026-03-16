import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import open3d as o3d
import config


def filter_region(pts: np.ndarray) -> np.ndarray:
    # Cap scan radius first
    dists = np.linalg.norm(pts, axis=1)
    pts   = pts[dists <= config.MAX_SCAN_RADIUS]
    if len(pts) == 0:
        return pts
    mask = (
        (pts[:, 0] >= config.REGION_X_MIN) & (pts[:, 0] <= config.REGION_X_MAX) &
        (pts[:, 1] >= config.REGION_Y_MIN) & (pts[:, 1] <= config.REGION_Y_MAX) &
        (pts[:, 2] >= config.REGION_Z_MIN) & (pts[:, 2] <= config.REGION_Z_MAX)
    )
    return pts[mask]


def filter_reflections(pts: np.ndarray) -> np.ndarray:
    if not config.REFLECTION_FILTER_ENABLED or len(pts) < 10:
        return pts
    tree   = cKDTree(pts)
    counts = tree.query_ball_point(pts, config.REFLECTION_RADIUS, return_length=True)
    return pts[counts >= config.REFLECTION_MIN_NEIGHBOURS]


def remove_planes(pts: np.ndarray) -> np.ndarray:
    if len(pts) < config.PLANE_MIN_POINTS:
        return pts
    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    for _ in range(config.PLANE_MAX_PLANES):
        if len(pcd.points) < config.PLANE_MIN_POINTS:
            break
        plane_model, inliers = pcd.segment_plane(
            distance_threshold = config.PLANE_DISTANCE_THRESHOLD,
            ransac_n           = config.PLANE_RANSAC_N,
            num_iterations     = config.PLANE_NUM_ITERS
        )
        if len(inliers) < config.PLANE_MIN_POINTS:
            break
        outliers = np.setdiff1d(np.arange(len(pcd.points)), inliers)
        pcd      = pcd.select_by_index(outliers)
    if len(pcd.points) == 0:
        return np.array([]).reshape(0, 3)
    return np.asarray(pcd.points, dtype=np.float32)


def voxelize(pts: np.ndarray) -> dict:
    voxels  = {}
    indices = np.floor(pts / config.VOXEL_SIZE).astype(int)
    for idx, pt in zip(indices, pts):
        voxels.setdefault(tuple(idx), []).append(pt)
    return voxels


def group_voxels(voxels: dict) -> list:
    if not voxels:
        return []
    centers    = np.array([np.mean(v, axis=0) for v in voxels.values()])
    voxel_list = list(voxels.values())
    tree       = cKDTree(centers)
    labels     = [-1] * len(centers)
    current    = 0
    for i in range(len(centers)):
        if labels[i] != -1:
            continue
        q         = [i]
        labels[i] = current
        while q:
            idx = q.pop()
            for n in tree.query_ball_point(centers[idx], config.VOXEL_SIZE * config.VOXEL_NEIGHBOR_FACTOR):
                if labels[n] == -1:
                    labels[n] = current
                    q.append(n)
        current += 1
    objects = {}
    for label, voxel in zip(labels, voxel_list):
        objects.setdefault(label, []).extend(voxel)
    return [np.array(v) for v in objects.values()]


def shape_classifier(pts: np.ndarray) -> int:
    if len(pts) < config.MIN_CLUSTER_POINTS:
        return 0
    dims   = pts.max(axis=0) - pts.min(axis=0)
    height = dims[2]
    width  = dims[0]
    depth  = dims[1]
    return int(
        config.HUMAN_HEIGHT_MIN <= height <= config.HUMAN_HEIGHT_MAX and
        config.HUMAN_WIDTH_MIN  <= width  <= config.HUMAN_WIDTH_MAX  and
        config.HUMAN_WIDTH_MIN  <= depth  <= config.HUMAN_WIDTH_MAX
    )


def normal_classifier(pts: np.ndarray) -> int:
    if len(pts) < config.MIN_CLUSTER_POINTS:
        return 0
    try:
        pca   = PCA(n_components=3)
        pca.fit(pts)
        cos_z = abs(np.dot(pca.components_[0], [0, 0, 1]))
        return int(cos_z > 0.7)
    except Exception:
        return 0


def shadow_classifier(pts: np.ndarray, all_pts: np.ndarray) -> int:
    if len(pts) < 5 or len(all_pts) == 0:
        return 0
    centroid = pts.mean(axis=0)
    tree     = cKDTree(all_pts)
    count    = len(tree.query_ball_point(centroid, config.SHADOW_RADIUS))
    return int(count >= config.SHADOW_MIN_PTS)


def merge_nearby_humans(humans: list) -> list:
    """
    Merge detections closer than HUMAN_MERGE_RADIUS — same person split into fragments.
    Uses a greedy approach so fragments chain-merge correctly.
    """
    if len(humans) <= 1:
        return humans

    used   = [False] * len(humans)
    merged = []

    for i in range(len(humans)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        # Keep expanding group until no more close neighbours found
        changed = True
        while changed:
            changed = False
            for j in range(len(humans)):
                if used[j]:
                    continue
                # Check against all current group members
                for k in group:
                    dist = float(np.linalg.norm(
                        humans[k]['centroid'] - humans[j]['centroid']
                    ))
                    if dist < config.HUMAN_MERGE_RADIUS:
                        group.append(j)
                        used[j] = True
                        changed = True
                        break

        all_pts  = np.vstack([humans[k]['points'] for k in group])
        centroid = all_pts.mean(axis=0)
        merged.append({
            'centroid':   centroid,
            'bbox_min':   all_pts.min(axis=0),
            'bbox_max':   all_pts.max(axis=0),
            'points':     all_pts,
            'distance_m': float(np.linalg.norm(centroid))
        })

    return merged


def detect(frame_pts: np.ndarray, debug: bool = False) -> tuple:
    """
    Pipeline:
      1. Region crop + radius cap
      2. Reflection filter
      3. RANSAC plane removal
      4. Voxel clustering
      5. Three-classifier voting
      6. Merge nearby fragments

    Returns: (list of human dicts, non-human points array)
    """
    region = filter_region(frame_pts)
    if len(region) < 10:
        return [], np.array([]).reshape(0, 3)

    clean = filter_reflections(region)
    if len(clean) < 10:
        return [], np.array([]).reshape(0, 3)

    filtered = remove_planes(clean)
    if len(filtered) < 10:
        return [], filtered

    objects = group_voxels(voxelize(filtered))

    humans     = []
    non_humans = []

    for obj in objects:
        if len(obj) < config.MIN_CLUSTER_POINTS:
            non_humans.extend(obj.tolist())
            continue

        s = shape_classifier(obj)
        n = normal_classifier(obj)
        sh = shadow_classifier(obj, filtered)
        votes = s + n + sh

        if debug:
            dims = obj.max(axis=0) - obj.min(axis=0)
            print(f"  cluster pts={len(obj):4d} | h={dims[2]:.2f} w={dims[0]:.2f} d={dims[1]:.2f} "
                  f"| shape={s} normal={n} shadow={sh} → {'HUMAN' if votes >= config.MIN_VOTES else 'skip'}")

        if votes >= config.MIN_VOTES:
            centroid = obj.mean(axis=0)
            humans.append({
                'centroid':   centroid,
                'bbox_min':   obj.min(axis=0),
                'bbox_max':   obj.max(axis=0),
                'points':     obj,
                'distance_m': float(np.linalg.norm(centroid))
            })
        else:
            non_humans.extend(obj.tolist())

    humans = merge_nearby_humans(humans)

    return humans, np.array(non_humans).reshape(-1, 3) if non_humans else np.array([]).reshape(0, 3)