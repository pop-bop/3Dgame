import json
import math
import os
import numpy as np


GRAVITY      = -9.81
TERMINAL_V   = -30.0
PLAYER_H     =  1.8
PLAYER_R     =  0.3
ACCEL        =  60.0
FRICTION_GND =  3.0
FRICTION_AIR =  2.0
JUMP_IMPULSE =  6.0
MAX_SPEED    =  8.0
FLY_SPEED    =  MAX_SPEED * 1.5


_HALF = np.array([PLAYER_R, PLAYER_H / 2.0, PLAYER_R], dtype=np.float64)


_FLOOR_THRESH = 0.7

_SEAM_SLOP = -0.04


statics      = {}
mesh_statics = {}

player = {
    "pos":       np.array([0.0, 2.0, 5.0], dtype=np.float64),
    "vel":       np.zeros(3, dtype=np.float64),
    "on_ground": False,
    "fly_mode":  False,
    "no_clip":   False,
}


def player_aabb():

    return player["pos"] - _HALF, player["pos"] + _HALF


def player_camera_pos():

    p = player["pos"].copy()
    p[1] += PLAYER_H * 0.5 - 0.1
    return p.tolist()


def load_physics_props(name, root="."):
    path     = os.path.join(root, f"{name}.physics.json")
    defaults = {"mass": 1.0, "restitution": 0.0, "friction": 0.5, "solid": True}
    if os.path.exists(path):
        with open(path) as f:
            defaults.update(json.load(f))
    return defaults



def add_static(name, mn, mx):

    statics[name] = {
        "mn": np.array(mn, dtype=np.float64),
        "mx": np.array(mx, dtype=np.float64),
    }


def add_mesh_static(name, mesh, pos, rot_deg=None):


    from renderer import rotation_matrix

    verts = mesh.get("verts")
    if verts is None:
        verts = np.array(mesh["tris"], dtype=np.float32)
    triangles = verts.reshape(-1, 3, 3).astype(np.float64).copy()

    if rot_deg is not None:
        R         = rotation_matrix(rot_deg).astype(np.float64)
        triangles = triangles @ R.T

    triangles += np.array(pos, dtype=np.float64)

    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]

    edge_a  = v1 - v0
    edge_b  = v2 - v0
    normals = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)

    valid   = (lengths[:, 0] > 1e-10)
    normals = normals[valid] / lengths[valid]
    v0      = v0[valid]
    v1      = v1[valid]
    v2      = v2[valid]


    plane_d = np.einsum("ij,ij->i", normals, v0)

    mesh_statics[name] = {
        "normals": normals,
        "plane_d": plane_d,
        "v0":      v0,
        "v1":      v1,
        "v2":      v2,
    }


def _aabb_penetration(mn_a, mx_a, mn_b, mx_b):

    if not ((mn_a <= mx_b).all() and (mx_a >= mn_b).all()):
        return None
    overlap_depths = np.minimum(mx_a, mx_b) - np.maximum(mn_a, mn_b)
    axis           = int(np.argmin(overlap_depths))
    center_a       = (mn_a[axis] + mx_a[axis]) * 0.5
    center_b       = (mn_b[axis] + mx_b[axis]) * 0.5
    sign           = 1.0 if center_a > center_b else -1.0
    return axis, sign * overlap_depths[axis]


def _resolve_aabb_ground():

    mn, mx = player_aabb()
    for slab in statics.values():
        result = _aabb_penetration(mn, mx,
                                   slab["mn"].astype(np.float64),
                                   slab["mx"].astype(np.float64))
        if result is None:
            continue
        axis, depth = result
        correction        = np.zeros(3, dtype=np.float64)
        correction[axis]  = depth
        player["pos"]    += correction
        mn, mx            = player_aabb()
        if axis == 1 and depth > 0:
            player["on_ground"] = True
            if player["vel"][1] < 0:
                player["vel"][1] = 0.0


def _point_near_triangle(plane_projection, triangle_normal, v0, v1, v2,
                         generous=False):

    edge_a = v1 - v0
    edge_b = v2 - v0
    to_pt  = plane_projection - v0

    d00 = np.dot(edge_a, edge_a)
    d01 = np.dot(edge_a, edge_b)
    d11 = np.dot(edge_b, edge_b)
    d20 = np.dot(to_pt,  edge_a)
    d21 = np.dot(to_pt,  edge_b)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return False

    inv    = 1.0 / denom
    bary_v = (d11 * d20 - d01 * d21) * inv
    bary_w = (d00 * d21 - d01 * d20) * inv
    bary_u = 1.0 - bary_v - bary_w

    if not generous:

        return bary_u >= _SEAM_SLOP and bary_v >= _SEAM_SLOP and bary_w >= _SEAM_SLOP

    if bary_u >= 0 and bary_v >= 0 and bary_w >= 0:
        return True

    bv_c = max(0.0, min(1.0, bary_v))
    bw_c = max(0.0, min(1.0, bary_w))
    if bv_c + bw_c > 1.0:
        total = bv_c + bw_c
        bv_c /= total
        bw_c /= total
    closest = v0 + bv_c * edge_a + bw_c * edge_b
    dist_sq = np.dot(plane_projection - closest, plane_projection - closest)
    margin  = math.sqrt(float(np.dot(_HALF ** 2, 1.0 - triangle_normal ** 2)))
    return dist_sq <= margin * margin


def _cancel_velocity_into_surface(vel, surface_normal):

    vel_along_normal = np.dot(vel, surface_normal)   # scalar projection onto normal
    if vel_along_normal < 0.0:
        # vel_along_normal is negative here, so we subtract a negative value,
        # effectively adding back the inward component to zero it out.
        vel -= vel_along_normal * surface_normal


def _resolve_triangles():

    if not mesh_statics:
        return

    center = player["pos"]
    vel    = player["vel"]

    for _ in range(5):
        any_push = False

        for ms in mesh_statics.values():
            normals = ms["normals"]
            plane_d = ms["plane_d"]
            v0      = ms["v0"]
            v1      = ms["v1"]
            v2      = ms["v2"]


            signed_dist = normals @ center - plane_d

            support_radius = np.abs(normals) @ _HALF


            penetration = support_radius - signed_dist

            candidates = np.where(
                (penetration > 1e-4) & (signed_dist > -support_radius)
            )[0]
            if candidates.size == 0:
                continue

            order = candidates[np.argsort(-penetration[candidates])]

            for idx in order:
                triangle_normal = normals[idx]

                sd_now  = np.dot(triangle_normal, center) - plane_d[idx]
                pen_now = (np.abs(triangle_normal) @ _HALF) - sd_now
                if pen_now <= 1e-4:
                    continue

                plane_proj = center - sd_now * triangle_normal
                inside_mesh = sd_now < 0.0
                if not _point_near_triangle(plane_proj, triangle_normal,
                                            v0[idx], v1[idx], v2[idx],
                                            generous=inside_mesh):
                    continue

                center += triangle_normal * pen_now

                _cancel_velocity_into_surface(vel, triangle_normal)


                if triangle_normal[1] > _FLOOR_THRESH:
                    player["on_ground"] = True

                any_push = True

        if not any_push:
            break


    player["pos"] = center
    player["vel"] = vel


def move_player(wish_x, wish_z, jump, dt):

    dt = min(dt, 0.05)

    if player["fly_mode"]:
        player["vel"][0] = wish_x * FLY_SPEED
        player["vel"][2] = wish_z * FLY_SPEED
        player["pos"]   += player["vel"] * dt
        retur

    if not player["on_ground"]:
        player["vel"][1] = max(player["vel"][1] + GRAVITY * dt, TERMINAL_V)
    else:
        player["vel"][1] = max(player["vel"][1], -0.1)


    if jump and player["on_ground"]:
        player["vel"][1]    = JUMP_IMPULSE
        player["on_ground"] = False

    friction       = FRICTION_GND if player["on_ground"] else FRICTION_AIR
    horizontal_spd = math.hypot(player["vel"][0], player["vel"][2])
    if horizontal_spd > 1e-6:
        new_spd              = max(horizontal_spd - horizontal_spd * friction * dt, 0.0)
        scale                = new_spd / horizontal_spd
        player["vel"][0]    *= scale
        player["vel"][2]    *= scale

    wish_len = math.hypot(wish_x, wish_z)
    if wish_len > 1e-6:
        wish_dir_x         = wish_x / wish_len
        wish_dir_z         = wish_z / wish_len
        player["vel"][0]  += wish_dir_x * ACCEL * dt
        player["vel"][2]  += wish_dir_z * ACCEL * dt

    horizontal_spd = math.hypot(player["vel"][0], player["vel"][2])
    if horizontal_spd > MAX_SPEED:
        scale             = MAX_SPEED / horizontal_spd
        player["vel"][0] *= scale
        player["vel"][2] *= scale

    player["on_ground"]  = False
    player["pos"]       += player["vel"] * dt

    if not player["no_clip"]:
        _resolve_aabb_ground()
        _resolve_triangles()
