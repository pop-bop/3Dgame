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

CAPSULE_TOP    = np.array([0.0,  PLAYER_H / 2.0 - PLAYER_R, 0.0], dtype=np.float64)
CAPSULE_BOTTOM = np.array([0.0, -PLAYER_H / 2.0 + PLAYER_R, 0.0], dtype=np.float64)

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
    half = np.array([PLAYER_R, PLAYER_H / 2.0, PLAYER_R], dtype=np.float64)
    return player["pos"] - half, player["pos"] + half


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


def _closest_point_on_segment(p, a, b):
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-10)
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab


def _closest_point_on_triangle(p, v0, v1, v2):
    edge_a = v1 - v0
    edge_b = v2 - v0
    to_pt  = p - v0

    d00 = np.dot(edge_a, edge_a)
    d01 = np.dot(edge_a, edge_b)
    d11 = np.dot(edge_b, edge_b)
    d20 = np.dot(to_pt,  edge_a)
    d21 = np.dot(to_pt,  edge_b)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return v0

    inv    = 1.0 / denom
    v = (d11 * d20 - d01 * d21) * inv
    w = (d00 * d21 - d01 * d20) * inv
    u = 1.0 - v - w

    if u >= 0 and v >= 0 and w >= 0:
        return v0 + v * edge_a + w * edge_b

    def clamp_edge(s, edge, start):
        t = np.clip(s, 0.0, 1.0)
        return start + t * edge

    t_01 = np.dot(to_pt, edge_a) / (d00 + 1e-10)
    p_01 = clamp_edge(t_01, edge_a, v0)

    t_02 = np.dot(to_pt, edge_b) / (d11 + 1e-10)
    p_02 = clamp_edge(t_02, edge_b, v0)

    edge_c = v2 - v1
    to_v1 = p - v1
    t_12 = np.dot(to_v1, edge_c) / (np.dot(edge_c, edge_c) + 1e-10)
    p_12 = clamp_edge(t_12, edge_c, v1)

    dist_01 = np.linalg.norm(p - p_01)
    dist_02 = np.linalg.norm(p - p_02)
    dist_12 = np.linalg.norm(p - p_12)

    if dist_01 <= dist_02 and dist_01 <= dist_12:
        return p_01
    elif dist_02 <= dist_12:
        return p_02
    else:
        return p_12


def _capsule_triangle_collision(center, v0, v1, v2, normal):
    cap_bottom = center + CAPSULE_BOTTOM
    cap_top = center + CAPSULE_TOP

    min_dist = float('inf')
    best_capsule_pt = None
    best_tri_pt = None

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        capsule_pt = cap_bottom + alpha * (cap_top - cap_bottom)
        tri_pt = _closest_point_on_triangle(capsule_pt, v0, v1, v2)
        dist = np.linalg.norm(capsule_pt - tri_pt)

        if dist < min_dist:
            min_dist = dist
            best_capsule_pt = capsule_pt
            best_tri_pt = tri_pt

    penetration = PLAYER_R - min_dist

    if penetration > 1e-4:
        if min_dist < 1e-6:
            collision_normal = normal.copy()
            to_player = center - best_tri_pt
            if np.dot(collision_normal, to_player) < 0:
                collision_normal = -collision_normal
        else:
            collision_normal = (best_capsule_pt - best_tri_pt) / min_dist

        return penetration, collision_normal

    return None


def _resolve_aabb_ground():
    half = np.array([PLAYER_R, PLAYER_H / 2.0, PLAYER_R], dtype=np.float64)
    mn = player["pos"] - half
    mx = player["pos"] + half

    for slab in statics.values():
        if not ((mn <= slab["mx"]).all() and (mx >= slab["mn"]).all()):
            continue

        overlap_depths = np.minimum(mx, slab["mx"]) - np.maximum(mn, slab["mn"])
        axis = int(np.argmin(overlap_depths))

        center_a = (mn[axis] + mx[axis]) * 0.5
        center_b = (slab["mn"][axis] + slab["mx"][axis]) * 0.5
        sign = 1.0 if center_a > center_b else -1.0

        correction = np.zeros(3, dtype=np.float64)
        correction[axis] = sign * overlap_depths[axis]
        player["pos"] += correction

        if axis == 1 and sign > 0:
            player["on_ground"] = True
            if player["vel"][1] < 0:
                player["vel"][1] = 0.0


def _resolve_capsule_mesh():
    if not mesh_statics:
        return

    center = player["pos"]
    vel = player["vel"]

    MAX_ITERATIONS = 3

    for iteration in range(MAX_ITERATIONS):
        any_collision = False
        collision_count = 0

        for ms in mesh_statics.values():
            normals = ms["normals"]
            plane_d = ms["plane_d"]
            v0 = ms["v0"]
            v1 = ms["v1"]
            v2 = ms["v2"]

            signed_dist = normals @ center - plane_d

            candidates = np.where(np.abs(signed_dist) < (PLAYER_H / 2.0 + PLAYER_R + 0.1))[0]

            if len(candidates) > 0:
                penetrations = []
                for idx in candidates:
                    result = _capsule_triangle_collision(
                        center, v0[idx], v1[idx], v2[idx], normals[idx]
                    )
                    if result is not None:
                        penetrations.append((result[0], idx, result[1]))

                penetrations.sort(reverse=True, key=lambda x: x[0])

                for penetration, idx, collision_normal in penetrations[:3]:
                    center += collision_normal * (penetration + 0.001)

                    vel_into_surface = np.dot(vel, collision_normal)
                    if vel_into_surface < 0:
                        vel -= vel_into_surface * collision_normal

                    if collision_normal[1] > _FLOOR_THRESH:
                        player["on_ground"] = True

                    any_collision = True
                    collision_count += 1

                    break

        if not any_collision or collision_count > 5:
            break

    player["pos"] = center
    player["vel"] = vel


def move_player(wish_x, wish_z, jump, dt):
    dt = min(dt, 0.05)

    if player["fly_mode"]:
        player["vel"][0] = wish_x * FLY_SPEED
        player["vel"][2] = wish_z * FLY_SPEED
        player["pos"] += player["vel"] * dt
        return

    if not player["on_ground"]:
        player["vel"][1] = max(player["vel"][1] + GRAVITY * dt, TERMINAL_V)
    else:
        player["vel"][1] = max(player["vel"][1], -0.1)

    if jump and player["on_ground"]:
        player["vel"][1] = JUMP_IMPULSE
        player["on_ground"] = False

    friction = FRICTION_GND if player["on_ground"] else FRICTION_AIR
    horizontal_spd = math.hypot(player["vel"][0], player["vel"][2])
    if horizontal_spd > 1e-6:
        new_spd = max(horizontal_spd - horizontal_spd * friction * dt, 0.0)
        scale = new_spd / horizontal_spd
        player["vel"][0] *= scale
        player["vel"][2] *= scale

    wish_len = math.hypot(wish_x, wish_z)
    if wish_len > 1e-6:
        wish_dir_x = wish_x / wish_len
        wish_dir_z = wish_z / wish_len
        player["vel"][0] += wish_dir_x * ACCEL * dt
        player["vel"][2] += wish_dir_z * ACCEL * dt

    horizontal_spd = math.hypot(player["vel"][0], player["vel"][2])
    if horizontal_spd > MAX_SPEED:
        scale = MAX_SPEED / horizontal_spd
        player["vel"][0] *= scale
        player["vel"][2] *= scale

    player["on_ground"] = False
    player["pos"] += player["vel"] * dt

    if not player["no_clip"]:
        _resolve_aabb_ground()
        _resolve_capsule_mesh()
