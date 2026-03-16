"""
Unitree L2 LiDAR — Real-Time Human Detection
=============================================
Usage:
  python main.py            — live detection
  python main.py --scan-bg  — scan empty room and save background map
  python main.py --view-bg  — view saved background map
  python main.py --debug    — print per-cluster classifier votes each frame
"""

import sys
import os
import threading
import time
import numpy as np

import config
import udp_receiver
import background_map
import detector
import visualiser

_recent = []
_lock   = threading.Lock()

DEBUG = "--debug" in sys.argv

_tracks  = {}
_next_id = [0]
MATCH_RADIUS = 1.0
VANISH_SEC   = 2.0


def _clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def _get_frame():
    with _lock:
        if not _recent:
            return None
        return np.vstack(_recent)


def _match_and_update(humans, now):
    results            = []
    matched_track_ids  = set()
    matched_human_idxs = set()

    if _tracks and humans:
        from scipy.spatial import cKDTree
        track_ids   = list(_tracks.keys())
        track_cents = np.array([_tracks[tid]['centroid'] for tid in track_ids])
        human_cents = np.array([h['centroid'] for h in humans])
        tree        = cKDTree(track_cents)
        dists, idxs = tree.query(human_cents, k=1)

        for hi, (dist, ti) in enumerate(zip(dists, idxs)):
            if dist < MATCH_RADIUS:
                tid = track_ids[ti]
                if tid not in matched_track_ids:
                    matched_track_ids.add(tid)
                    matched_human_idxs.add(hi)
                    track    = _tracks[tid]
                    centroid = humans[hi]['centroid']
                    dt       = now - track['last_time']
                    disp     = float(np.linalg.norm(centroid - track['centroid']))
                    vel      = disp / dt if dt > 0 else 0.0
                    track['total_displacement'] += disp
                    track['centroid']  = centroid
                    track['last_time'] = now
                    results.append((tid, humans[hi], disp, vel, track['total_displacement']))

    for hi, human in enumerate(humans):
        if hi not in matched_human_idxs:
            tid = _next_id[0]
            _next_id[0] += 1
            _tracks[tid] = {
                'centroid':           human['centroid'],
                'last_time':          now,
                'total_displacement': 0.0,
            }
            results.append((tid, human, 0.0, 0.0, 0.0))

    stale = [tid for tid, t in _tracks.items()
             if now - t['last_time'] > VANISH_SEC and tid not in matched_track_ids]
    for tid in stale:
        del _tracks[tid]

    return results


def _print_dashboard(frame_n, tracked, fg_count):
    _clear()
    W = 55
    print("=" * W)
    print("   UNITREE L2 — HUMAN DETECTION DASHBOARD")
    print("=" * W)
    print(f"  Frame       : {frame_n}")
    print(f"  Foreground  : {fg_count} points")
    print(f"  Detected    : {len(tracked)} human(s)")
    print("-" * W)

    if not tracked:
        print("\n  [ No humans detected ]\n")
    else:
        for tid, human, disp, vel, total in tracked:
            c = human['centroid']
            # Motion bar — scale velocity 0-2 m/s to 20 chars
            bar_len  = min(int(vel / 1.0 * 20), 20)
            bar      = "█" * bar_len + "░" * (20 - bar_len)

            print(f"\n  ┌─ Human {tid+1}  (ID {tid}) {'─'*(W-18)}")
            print(f"  │  Position   : X={c[0]:+.2f}  Y={c[1]:+.2f}  Z={c[2]:+.2f}  m")
            print(f"  │  Distance   : {human['distance_m']:.2f} m from LiDAR")
            print(f"  │  Displacement: {disp*100:.1f} cm  (last 0.5s)")
            print(f"  │  Velocity   : {vel:.3f} m/s  ({vel*3.6:.2f} km/h)")
            print(f"  │  Speedometer  : [{bar}]")
            print(f"  └  Total moved: {total:.2f} m this session")

    print("\n" + "=" * W)
    print("  Close the Open3D window to exit")
    print("=" * W)


def _ingestion_loop():
    from packet_parser import parse_packet
    while True:
        try:
            data   = udp_receiver.packet_queue.get(timeout=1.0)
            result = parse_packet(data)
            if result and result['type'] == 'pointcloud':
                with _lock:
                    _recent.append(result['points'][:, :3])
                    if len(_recent) > config.MAX_PACKETS:
                        _recent[:] = _recent[-config.MAX_PACKETS:]
        except Exception:
            pass


def _detection_loop(bg_tree):
    frame_n = 0
    while True:
        time.sleep(0.5)
        frame_n += 1

        with _lock:
            if not _recent:
                continue
            frame_pts = np.vstack(_recent)

        fg = background_map.subtract(frame_pts, bg_tree)
        visualiser.push_result([], frame_pts)

        if len(fg) < 10:
            _print_dashboard(frame_n, [], len(fg))
            continue

        if DEBUG:
            print(f"\n[{frame_n}] fg={len(fg)} pts")

        humans, _ = detector.detect(fg, debug=DEBUG)
        visualiser.push_result(humans, frame_pts)

        now     = time.time()
        tracked = _match_and_update(humans, now) if humans else []
        _print_dashboard(frame_n, tracked, len(fg))


def run_live():
    print("\n" + "="*55)
    print("  Unitree L2 LiDAR — Live Human Detection")
    if DEBUG:
        print("  DEBUG MODE ON")
    print("="*55 + "\n")

    try:
        bg_pts = background_map.load()
    except FileNotFoundError:
        print(f"ERROR: No background map at '{config.BG_MAP_PATH}'")
        print("Run:  python main.py --scan-bg")
        sys.exit(1)

    bg_tree = background_map.build_tree(bg_pts)
    udp_receiver.start_receiver()

    print("\nAccumulating initial scan...")
    frames = udp_receiver.get_frames(n=200, timeout=15.0)
    if not frames:
        print("ERROR: No packets received. Check LiDAR connection.")
        sys.exit(1)

    with _lock:
        _recent.extend(frames)
    print(f"Ready ({len(frames)} packets).\n")

    threading.Thread(target=_ingestion_loop, daemon=True).start()
    threading.Thread(target=_detection_loop, args=(bg_tree,), daemon=True).start()

    visualiser.run(_get_frame)
    print("\nExiting.")


def run_scan_bg():
    udp_receiver.start_receiver()
    background_map.scan_and_save()


def run_view_bg():
    import open3d as o3d
    bg_pts     = background_map.load()
    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(bg_pts)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile([0.2, 0.4, 1.0], (len(bg_pts), 1))
    )
    o3d.visualization.draw_geometries(
        [pcd], window_name="Background Map",
        width=config.WINDOW_WIDTH, height=config.WINDOW_HEIGHT
    )


if __name__ == "__main__":
    if "--scan-bg" in sys.argv:
        run_scan_bg()
    elif "--view-bg" in sys.argv:
        run_view_bg()
    else:
        run_live()