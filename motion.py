import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

# Tunable parameters (all in mm)
BG_TOLERANCE     = 500   # how close a point must be to background to be ignored
DBSCAN_EPS       = 200   # cluster radius
DBSCAN_MIN_PTS   = 10    # minimum points to form a cluster
MIN_CLUSTER_PTS  = 15    # discard tiny clusters (noise)
MOVE_THRESHOLD   = 150   # mm a cluster must move to count as motion
PERSIST_FRAMES   = 3     # frames a cluster must exist before we trust it
VANISH_FRAMES    = 5     # frames before we drop a lost track

class MovingCluster:
    def __init__(self, cluster_id, centroid, points):
        self.id         = cluster_id
        self.centroid   = centroid
        self.points     = points
        self.history    = [centroid]   # position history
        self.age        = 1            # frames seen
        self.lost       = 0            # frames since last match
        self.is_human   = False        # confirmed moving?

    def update(self, centroid, points):
        self.centroid = centroid
        self.points   = points
        self.history.append(centroid)
        if len(self.history) > 30:
            self.history = self.history[-30:]
        self.age  += 1
        self.lost  = 0

        # Check if total displacement across history exceeds threshold
        if len(self.history) >= 2:
            displacement = np.linalg.norm(
                np.array(self.history[-1]) - np.array(self.history[0])
            )
            if displacement > MOVE_THRESHOLD:
                self.is_human = True


class MotionDetector:
    def __init__(self, background_points):
        print(f"Building background KD-tree from {len(background_points)} points...")
        self.bg_tree   = cKDTree(background_points)
        self.tracks    = {}
        self.next_id   = 0

    def subtract_background(self, frame_points):
        distances, _ = self.bg_tree.query(frame_points, k=1)
        mask = distances > BG_TOLERANCE
        return frame_points[mask]

    def cluster(self, foreground_points):
        if len(foreground_points) < DBSCAN_MIN_PTS:
            return []

        db     = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_PTS).fit(foreground_points)
        labels = db.labels_

        clusters = []
        for label in set(labels):
            if label == -1:
                continue
            pts = foreground_points[labels == label]
            if len(pts) < MIN_CLUSTER_PTS:
                continue
            clusters.append(pts)
        return clusters

    def match_clusters(self, clusters):
        # Match new clusters to existing tracks by nearest centroid
        new_centroids = [c.mean(axis=0) for c in clusters]

        matched_track_ids = set()
        matched_cluster_ids = set()

        if self.tracks and new_centroids:
            track_ids      = list(self.tracks.keys())
            track_cents    = np.array([self.tracks[tid].centroid for tid in track_ids])
            cluster_cents  = np.array(new_centroids)

            tree = cKDTree(track_cents)
            dists, indices = tree.query(cluster_cents, k=1)

            for ci, (dist, ti) in enumerate(zip(dists, indices)):
                if dist < 1000:  # max match distance in mm
                    tid = track_ids[ti]
                    if tid not in matched_track_ids:
                        self.tracks[tid].update(new_centroids[ci], clusters[ci])
                        matched_track_ids.add(tid)
                        matched_cluster_ids.add(ci)

        # Spawn new tracks for unmatched clusters
        for ci, pts in enumerate(clusters):
            if ci not in matched_cluster_ids:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = MovingCluster(new_id, new_centroids[ci], pts)

        # Age out lost tracks
        lost_ids = []
        for tid in self.tracks:
            if tid not in matched_track_ids:
                self.tracks[tid].lost += 1
                if self.tracks[tid].lost > VANISH_FRAMES:
                    lost_ids.append(tid)
        for tid in lost_ids:
            del self.tracks[tid]

    def process_frame(self, frame_points):
        # Step 1: background subtraction
        foreground = self.subtract_background(frame_points)

        # TEMP: print foreground count every time
        print(f"Frame pts: {len(frame_points)} | Foreground pts: {len(foreground)} | Tracks: {len(self.tracks)}")
    
        # Step 2: cluster foreground
        clusters = self.cluster(foreground)
        print(f"Clusters found: {len(clusters)}")

        # Step 3: match to existing tracks
        self.match_clusters(clusters)

        # Step 4: return confirmed humans (moving tracks)
        humans = [
            t for t in self.tracks.values()
            if t.is_human and t.age >= PERSIST_FRAMES
        ]

        return humans, foreground