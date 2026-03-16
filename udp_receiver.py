import socket
import threading
import queue
from packet_parser import parse_packet
import config

packet_queue = queue.Queue()

_started = False
_lock    = threading.Lock()


def start_receiver():
    global _started
    with _lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_loop, daemon=True).start()
    print(f"UDP receiver listening on port {config.HOST_PORT}")


def _loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", config.HOST_PORT))
    while True:
        try:
            data, addr = sock.recvfrom(65535)
            packet_queue.put(data)
        except Exception:
            pass


def get_frames(n: int, timeout: float = 15.0) -> list:
    """Block until n point cloud frames received. Returns list of (N,3) arrays in metres."""
    import time
    frames   = []
    deadline = time.time() + timeout
    while len(frames) < n:
        rem = deadline - time.time()
        if rem <= 0:
            break
        try:
            data   = packet_queue.get(timeout=min(rem, 1.0))
            result = parse_packet(data)
            if result and result['type'] == 'pointcloud':
                frames.append(result['points'][:, :3])
        except queue.Empty:
            pass
    return frames