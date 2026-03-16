import socket
import threading
import queue
import time
import numpy as np
import open3d as o3d
from packet_parser import parse_packet

packet_queue = queue.Queue()

def receive_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 6201))
    while True:
        try:
            data, addr = sock.recvfrom(65535)
            packet_queue.put(data)
        except:
            pass

# Start receiver thread
t = threading.Thread(target=receive_loop, daemon=True)
t.start()
print("Receiver started, accumulating initial scan...")

# Accumulate initial points before opening window
recent_points = []
while len(recent_points) < 200:
    try:
        data = packet_queue.get(timeout=5)
        result = parse_packet(data)
        if result and result['type'] == 'pointcloud':
            recent_points.append(result['points'][:, :3])
    except queue.Empty:
        break

print(f"Initial scan ready ({len(recent_points)} packets). Opening window...")

# Build initial point cloud
pts = np.vstack(recent_points)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

MAX_PACKETS = 300

def update_callback(vis):
    global recent_points
    new_batch = []
    try:
        while True:
            data = packet_queue.get_nowait()
            result = parse_packet(data)
            if result and result['type'] == 'pointcloud':
                new_batch.append(result['points'][:, :3])
    except queue.Empty:
        pass

    if new_batch:
        recent_points.extend(new_batch)
        if len(recent_points) > MAX_PACKETS:
            recent_points = recent_points[-MAX_PACKETS:]
        pts = np.vstack(recent_points)
        pcd.points = o3d.utility.Vector3dVector(pts)
        vis.update_geometry(pcd)
    return False

# Run visualiser on main thread with callback
o3d.visualization.draw_geometries_with_animation_callback(
    [pcd],
    update_callback,
    window_name="Unitree L2 LiDAR - Live",
    width=1280,
    height=720
)