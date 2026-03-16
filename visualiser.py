import queue
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import config

_result_queue = queue.Queue(maxsize=1)


def push_result(humans, env_points):
    if not _result_queue.full():
        _result_queue.put((humans, env_points))


def run(get_frame_fn):
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Unitree L2 — Human Detection",
        width=config.WINDOW_WIDTH,
        height=config.WINDOW_HEIGHT
    )

    pcd_env   = o3d.geometry.PointCloud()
    pcd_human = o3d.geometry.PointCloud()
    pcd_env.points   = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd_human.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))

    vis.add_geometry(pcd_env)
    vis.add_geometry(pcd_human)

    opt = vis.get_render_option()
    opt.point_size       = 2.5
    opt.background_color = np.array([1.0, 1.0, 1.0])

    camera_set  = [False]
    last_humans = [None]

    print("Visualiser open — rotate: left-drag | zoom: scroll | pan: right-drag")

    while True:
        frame = get_frame_fn()
        if frame is not None and len(frame) > 0:
            # Cap scan radius
            dists = np.linalg.norm(frame, axis=1)
            frame = frame[dists <= config.MAX_SCAN_RADIUS]

            if len(frame) > 0:
                humans = last_humans[0]
                colors = np.tile([0.2, 0.4, 1.0], (len(frame), 1))

                if humans:
                    # Build a KD-tree from all detected human points combined
                    all_human_pts = np.vstack([h['points'] for h in humans])
                    tree          = cKDTree(all_human_pts)
                    dists_to_human, _ = tree.query(frame, k=1)

                    # Colour frame points that are within 8cm of any human point
                    # This stays tight to the body and won't bleed into furniture
                    mask           = dists_to_human < 0.08
                    colors[mask]   = [1.0, 0.0, 0.0]

                pcd_env.points = o3d.utility.Vector3dVector(frame)
                pcd_env.colors = o3d.utility.Vector3dVector(colors)
                pcd_human.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
                pcd_human.colors = o3d.utility.Vector3dVector(np.zeros((1, 3)))

                vis.update_geometry(pcd_env)
                vis.update_geometry(pcd_human)

            if not camera_set[0]:
                vis.reset_view_point(True)
                camera_set[0] = True

        try:
            humans, _ = _result_queue.get_nowait()
            last_humans[0] = humans if humans else None
        except queue.Empty:
            pass

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()