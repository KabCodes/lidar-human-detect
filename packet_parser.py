import struct
import math
import numpy as np

MAGIC           = b"\x55\xAA\x05\x0A"
TYPE_POINTCLOUD = 102
TYPE_IMU        = 104


def parse_packet(data: bytes) -> dict:
    if len(data) < 16:
        return None
    if data[0:4] != MAGIC:
        return None
    packet_type = struct.unpack_from("<I", data, 4)[0]
    if packet_type == TYPE_POINTCLOUD:
        return _parse_pointcloud(data)
    elif packet_type == TYPE_IMU:
        return _parse_imu(data)
    return None


def _parse_pointcloud(data: bytes) -> dict:
    try:
        (a_axis_dist, b_axis_dist,
         theta_angle_bias, alpha_angle_bias,
         beta_angle, xi_angle,
         range_bias, range_scale
         ) = struct.unpack_from("<ffffffff", data, 64)

        (h_angle_start, h_angle_step, scan_period,
         range_min, range_max,
         angle_min, angle_increment, time_increment
         ) = struct.unpack_from("<ffffffff", data, 96)

        point_num = struct.unpack_from("<I", data, 128)[0]
        if point_num == 0 or point_num > 300:
            return None

        ranges      = struct.unpack_from(f"<{point_num}H", data, 132)
        intensities = struct.unpack_from(f"<{point_num}B", data, 732)

        sin_beta        = math.sin(beta_angle)
        cos_beta        = math.cos(beta_angle)
        sin_xi          = math.sin(xi_angle)
        cos_xi          = math.cos(xi_angle)
        cos_beta_sin_xi = cos_beta * sin_xi
        sin_beta_cos_xi = sin_beta * cos_xi
        sin_beta_sin_xi = sin_beta * sin_xi
        cos_beta_cos_xi = cos_beta * cos_xi

        alpha_cur = angle_min     + alpha_angle_bias
        theta_cur = h_angle_start + theta_angle_bias

        points = []
        for j in range(point_num):
            raw_r = ranges[j]
            if raw_r < 1:
                alpha_cur += angle_increment
                theta_cur += h_angle_step
                continue

            r = range_scale * (raw_r + range_bias)

            sin_a = math.sin(alpha_cur)
            cos_a = math.cos(alpha_cur)
            sin_t = math.sin(theta_cur)
            cos_t = math.cos(theta_cur)

            A = (-cos_beta_sin_xi + sin_beta_cos_xi * sin_a) * r + b_axis_dist
            B = cos_a * cos_xi * r
            C = (sin_beta_sin_xi + cos_beta_cos_xi * sin_a) * r

            x = cos_t * A - sin_t * B
            y = sin_t * A + cos_t * B
            z = C + a_axis_dist

            points.append([x, y, z, intensities[j]])

            alpha_cur += angle_increment
            theta_cur += h_angle_step

        if not points:
            return None

        pts = np.array(points, dtype=np.float32)
        return {'type': 'pointcloud', 'points': pts, 'count': len(pts)}

    except Exception:
        return None


def _parse_imu(data: bytes) -> dict:
    try:
        q = struct.unpack_from('<4f', data, 8)
        g = struct.unpack_from('<3f', data, 24)
        a = struct.unpack_from('<3f', data, 36)
        return {'type': 'imu', 'quaternion': q, 'gyro': g, 'accel': a}
    except Exception:
        return None