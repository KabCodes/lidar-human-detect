import os
import time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import config
import udp_receiver


def scan_and_save(duration: float = None, path: str = None):
    """Scan empty room and save background map in metres. Run once with no people."""
    duration = duration or config.BG_SCAN_DURATION
    path     = path     or config.BG_MAP_PATH

    udp_receiver.start_receiver()

    print(f"\n{'='*50}")
    print(f"BACKGROUND SCAN — {duration}s")
    print("Make sure NO people are in the room!")
    print(f"{'='*50}\n")

    all_points = []
    start = time.time()

    while time.time() - start < duration:
        elapsed = time.time() - start
        print(f"\r  {elapsed:.1f}s / {duration}s  |  packets: {len(all_points)}", end="", flush=True)
        frames = udp_receiver.get_frames(n=10, timeout=2.0)
        all_points.extend(frames)

    print(f"\n\nScan complete — {len(all_points)} packets")

    if not all_points:
        print("ERROR: No packets received.")
        return None

    pts = np.vstack(all_points).astype(np.float32)
    print(f"Points: {len(pts):,}")
    print(f"Range check — X: {pts[:,0].min():.2f} → {pts[:,0].max():.2f}  (should be metres, e.g. -5 to 5)")

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    np.save(path, pts)
    print(f"Saved → {path}")
    return pts


def load(path: str = None) -> np.ndarray:
    path = path or config.BG_MAP_PATH
    pts  = np.load(path).astype(np.float32)

    # Auto-fix: if coordinates look like mm (range > 100) convert to metres
    coord_range = float(np.abs(pts).max())
    if coord_range > 100.0:
        print(f"WARNING: map looks like mm (max={coord_range:.0f}), converting to metres...")
        pts = pts / 1000.0
        np.save(path, pts)
        print("Converted and resaved.")

    print(f"Loaded background: {len(pts):,} points")
    print(f"  X: {pts[:,0].min():.2f} → {pts[:,0].max():.2f} m")
    print(f"  Y: {pts[:,1].min():.2f} → {pts[:,1].max():.2f} m")
    print(f"  Z: {pts[:,2].min():.2f} → {pts[:,2].max():.2f} m")
    return pts


def build_tree(bg_pts: np.ndarray) -> cKDTree:
    """Downsample to 0.15m voxels and build KD-tree."""
    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bg_pts)
    pcd_down   = pcd.voxel_down_sample(voxel_size=0.15)
    pts_down   = np.asarray(pcd_down.points, dtype=np.float32)
    print(f"Background KD-tree: {len(bg_pts):,} → {len(pts_down):,} points (voxel=0.15m)")
    return cKDTree(pts_down)


def subtract(frame_pts: np.ndarray, tree: cKDTree) -> np.ndarray:
    """Return only points NOT in the background (foreground objects)."""
    if len(frame_pts) == 0:
        return frame_pts
    dists, _ = tree.query(frame_pts, k=1)
    return frame_pts[dists > config.BG_TOLERANCE]


if __name__ == "__main__":
    scan_and_save()