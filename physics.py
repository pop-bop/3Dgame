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

AIR_DRAG_COEFF = 0.1
ANGULAR_DRAG_COEFF = 0.1

CAPSULE_TOP    = np.array([0.0,  PLAYER_H / 2.0 - PLAYER_R, 0.0], dtype=np.float64)
CAPSULE_BOTTOM = np.array([0.0, -PLAYER_H / 2.0 + PLAYER_R, 0.0], dtype=np.float64)

_FLOOR_THRESH = 0.7
_SEAM_SLOP = -0.04

statics       = {}
mesh_statics  = {}
rigid_bodies  = {}
static_bodies = {}

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


def rotation_matrix(angles):
    yaw, pitch, roll = (np.radians(a % 360.0) for a in angles)

    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)

    R_yaw   = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    R_pitch = np.array([[1, 0, 0],   [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    R_roll  = np.array([[cr, -sr, 0],[sr, cr,  0], [0,  0,  1]], dtype=np.float64)

    return R_yaw @ R_pitch @ R_roll


def rot_to_angles(R):
    sp = float(np.clip(-R[1, 2], -1.0, 1.0))
    pitch = float(np.arcsin(sp))
    cp = np.cos(pitch)
    if abs(cp) > 1e-6:
        roll = float(np.arctan2(R[1, 0], R[1, 1]))
        yaw = float(np.arctan2(R[0, 2], R[2, 2]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-R[2, 0], R[0, 0]))
    return [float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))]


def load_rigid_body_props(file_path_or_dict, root="."):
    raw = {}
    if isinstance(file_path_or_dict, dict):
        raw = file_path_or_dict
    elif isinstance(file_path_or_dict, str):
        candidates = [
            file_path_or_dict,
            os.path.join(root, file_path_or_dict),
            os.path.join(root, f"{file_path_or_dict}.json"),
            os.path.join(root, f"{file_path_or_dict}.physics.json"),
        ]
        for c in candidates:
            if os.path.exists(c) and os.path.isfile(c):
                try:
                    with open(c, "r") as f:
                        raw = json.load(f)
                    break
                except Exception as e:
                    print(f"Error loading physics file {c}: {e}")

    clean = {}
    for k, v in raw.items():
        clean[k.lower().strip().replace(" ", "_").replace("-", "_")] = v

    props = {}
    props["mass"] = float(clean.get("mass", clean.get("m", 1.0)))

    # Initial velocity
    vel_raw = clean.get("init_vel", clean.get("velocity", clean.get("vel", clean.get("initial_velocity", [0.0, 0.0, 0.0]))))
    if isinstance(vel_raw, (int, float)):
        props["init_vel"] = [0.0, float(vel_raw), 0.0]
    elif isinstance(vel_raw, (list, tuple)):
        if len(vel_raw) == 2:
            props["init_vel"] = [float(vel_raw[0]), float(vel_raw[1]), 0.0]
        elif len(vel_raw) >= 3:
            props["init_vel"] = [float(vel_raw[0]), float(vel_raw[1]), float(vel_raw[2])]
        else:
            props["init_vel"] = [0.0, 0.0, 0.0]
    else:
        props["init_vel"] = [0.0, 0.0, 0.0]

    # Angular velocity
    ang_raw = clean.get("angular_velocity", clean.get("init_angular_vel", clean.get("ang_vel", clean.get("init_angular_velocity", [0.0, 0.0, 0.0]))))
    if isinstance(ang_raw, (int, float)):
        props["angular_velocity"] = [0.0, float(ang_raw), 0.0]
    elif isinstance(ang_raw, (list, tuple)):
        if len(ang_raw) == 2:
            props["angular_velocity"] = [float(ang_raw[0]), float(ang_raw[1]), 0.0]
        elif len(ang_raw) >= 3:
            props["angular_velocity"] = [float(ang_raw[0]), float(ang_raw[1]), float(ang_raw[2])]
        else:
            props["angular_velocity"] = [0.0, 0.0, 0.0]
    else:
        props["angular_velocity"] = [0.0, 0.0, 0.0]

    # Friction
    friction_val = clean.get("frictional_coefficient", clean.get("friction_coefficient", clean.get("friction", clean.get("frictional_coeff", None))))
    if friction_val is not None:
        f = float(friction_val)
        props["static_friction"] = float(clean.get("static_friction", f))
        props["dynamic_friction"] = float(clean.get("dynamic_friction", f * 0.7 if "static_friction" not in clean else f))
        props["frictional_coefficient"] = f
    else:
        props["static_friction"] = float(clean.get("static_friction", 0.3))
        props["dynamic_friction"] = float(clean.get("dynamic_friction", 0.2))
        props["frictional_coefficient"] = (props["static_friction"] + props["dynamic_friction"]) * 0.5

    # Drag / Air friction
    props["air_friction_coefficient"] = float(clean.get("air_friction_coefficient", clean.get("air_friction", clean.get("air_drag_coefficient", clean.get("air_drag_coeff", clean.get("air_drag", AIR_DRAG_COEFF))))))
    props["angular_drag_coefficient"] = float(clean.get("angular_drag_coefficient", clean.get("angular_friction_coefficient", clean.get("angular_drag_coeff", clean.get("angular_drag", ANGULAR_DRAG_COEFF)))))

    # Restitution
    props["restitution"] = float(clean.get("restitution", clean.get("elasticity", clean.get("bounciness", 0.7))))

    # Gravity
    props["use_gravity"] = bool(clean.get("use_gravity", clean.get("gravity", True)))

    if "size" in clean:
        props["size"] = clean["size"]
    if "half_extents" in clean:
        props["half_extents"] = clean["half_extents"]
    if "extents" in clean:
        props["extents"] = clean["extents"]
    if "center_offset" in clean:
        props["center_offset"] = clean["center_offset"]
    if "inertia" in clean:
        props["inertia"] = clean["inertia"]

    return props


def load_physics_props(name, root="."):
    return load_rigid_body_props(name, root=root)


class RigidBody:
    def __init__(self, name, mesh=None, pos=(0, 0, 0), rot=(0, 0, 0), props=None):
        self.name = name
        self.mesh = mesh
        props = props or {}

        self.mass = float(props.get("mass", 1.0))
        self.inv_mass = 1.0 / self.mass if self.mass > 0 else 0.0

        if "half_extents" in props:
            self.half_extents = np.array(props["half_extents"], dtype=np.float64)
            self.center_offset = np.array(props.get("center_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
        elif "extents" in props:
            self.half_extents = np.array(props["extents"], dtype=np.float64) * 0.5
            self.center_offset = np.array(props.get("center_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
        elif "size" in props:
            self.half_extents = np.array(props["size"], dtype=np.float64) * 0.5
            self.center_offset = np.array(props.get("center_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
        elif mesh is not None:
            verts = mesh.get("verts")
            if verts is None:
                verts = np.array(mesh["tris"], dtype=np.float32)
            flat = verts.reshape(-1, 3)
            mn, mx = flat.min(axis=0), flat.max(axis=0)
            self.half_extents = np.maximum((mx - mn) * 0.5, 0.05).astype(np.float64)
            self.center_offset = ((mx + mn) * 0.5).astype(np.float64)
        else:
            self.half_extents = np.array([0.5, 0.5, 0.5], dtype=np.float64)
            self.center_offset = np.zeros(3, dtype=np.float64)

        if self.inv_mass == 0:
            self.inertia_body = np.zeros((3, 3), dtype=np.float64)
            self.inv_inertia_body = np.zeros((3, 3), dtype=np.float64)
        elif "inertia" in props:
            I = props["inertia"]
            if isinstance(I, (int, float)):
                self.inertia_body = np.eye(3, dtype=np.float64) * float(I)
                self.inv_inertia_body = np.eye(3, dtype=np.float64) / float(I)
            else:
                self.inertia_body = np.diag(I).astype(np.float64)
                self.inv_inertia_body = np.diag([1.0 / x if x > 0 else 0.0 for x in I]).astype(np.float64)
        else:
            ex, ey, ez = self.half_extents
            I_xx = (1.0 / 3.0) * self.mass * (ey**2 + ez**2)
            I_yy = (1.0 / 3.0) * self.mass * (ex**2 + ez**2)
            I_zz = (1.0 / 3.0) * self.mass * (ex**2 + ey**2)
            self.inertia_body = np.diag([I_xx, I_yy, I_zz])
            self.inv_inertia_body = np.diag([1.0 / I_xx, 1.0 / I_yy, 1.0 / I_zz])

        self.position = np.array(pos, dtype=np.float64)
        self.velocity = np.array(props.get("init_vel", [0.0, 0.0, 0.0]), dtype=np.float64)
        self.acceleration = np.zeros(3, dtype=np.float64)
        self.force_accumulator = np.zeros(3, dtype=np.float64)

        if isinstance(rot, np.ndarray) and rot.shape == (3, 3):
            self.rotation_matrix = rot.astype(np.float64).copy()
        else:
            self.rotation_matrix = rotation_matrix(rot).astype(np.float64)

        self.angular_velocity = np.array(props.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=np.float64)
        self.angular_acceleration = np.zeros(3, dtype=np.float64)
        self.torque_accumulator = np.zeros(3, dtype=np.float64)

        self.restitution = float(props.get("restitution", 0.7))
        self.static_friction = float(props.get("static_friction", 0.3))
        self.dynamic_friction = float(props.get("dynamic_friction", 0.2))
        self.air_drag_coeff = float(props.get("air_friction_coefficient", AIR_DRAG_COEFF))
        self.angular_drag_coeff = float(props.get("angular_drag_coefficient", ANGULAR_DRAG_COEFF))
        self.use_gravity = bool(props.get("use_gravity", True))

    def get_center(self):
        return self.position + self.rotation_matrix @ self.center_offset

    def get_inv_inertia_world(self):
        if self.inv_mass == 0:
            return np.zeros((3, 3), dtype=np.float64)
        return self.rotation_matrix @ self.inv_inertia_body @ self.rotation_matrix.T

    def apply_force(self, force, application_point=None):
        f = np.asarray(force, dtype=np.float64)
        self.force_accumulator += f
        if application_point is not None:
            p = np.asarray(application_point, dtype=np.float64)
            r = p - self.get_center()
            torque = np.cross(r, f)
            self.torque_accumulator += torque

    def apply_torque(self, torque):
        self.torque_accumulator += np.asarray(torque, dtype=np.float64)

    def get_rotation_euler(self):
        return rot_to_angles(self.rotation_matrix)

    def set_rotation_euler(self, angles):
        self.rotation_matrix = rotation_matrix(angles).astype(np.float64)

    def _integrate_orientation(self, dt):
        omega = self.angular_velocity
        speed = np.linalg.norm(omega)
        if speed > 1e-10:
            angle = speed * dt
            axis = omega / speed
            K = np.array([
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0]
            ], dtype=np.float64)
            R_delta = np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
            self.rotation_matrix = R_delta @ self.rotation_matrix

            u, _, vt = np.linalg.svd(self.rotation_matrix)
            self.rotation_matrix = u @ vt
            if np.linalg.det(self.rotation_matrix) < 0:
                u[:, -1] *= -1
                self.rotation_matrix = u @ vt

    def get_vertices(self):
        ex, ey, ez = self.half_extents
        local_corners = [
            np.array([-ex, -ey, -ez]),
            np.array([ ex, -ey, -ez]),
            np.array([ ex,  ey, -ez]),
            np.array([-ex,  ey, -ez]),
            np.array([-ex, -ey,  ez]),
            np.array([ ex, -ey,  ez]),
            np.array([ ex,  ey,  ez]),
            np.array([-ex,  ey,  ez]),
        ]
        center = self.get_center()
        return [center + self.rotation_matrix @ c for c in local_corners]

    def update(self, dt):
        if dt <= 0 or self.inv_mass == 0:
            return

        drag_force = -self.velocity * self.air_drag_coeff
        gravity_force = np.array([0.0, GRAVITY * self.mass, 0.0], dtype=np.float64) if self.use_gravity else np.zeros(3, dtype=np.float64)
        self.acceleration = (self.force_accumulator + drag_force + gravity_force) * self.inv_mass

        drag_torque = -self.angular_velocity * self.angular_drag_coeff
        inv_I = self.get_inv_inertia_world()
        self.angular_acceleration = inv_I @ (self.torque_accumulator + drag_torque)

        # Semi-Implicit Euler
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

        self.angular_velocity += self.angular_acceleration * dt
        self._integrate_orientation(dt)

        self.force_accumulator[:] = 0.0
        self.torque_accumulator[:] = 0.0

        self.handle_ground_collisions()
        self.handle_static_box_collisions()

    def handle_ground_collisions(self):
        corners = self.get_vertices()
        deepest_penetration = 0.0
        collision_points = []
        ground_y = 0.0

        for corner in corners:
            val = corner[1]
            if val < ground_y:
                penetration = ground_y - val
                if penetration > deepest_penetration:
                    deepest_penetration = penetration
                collision_points.append(corner)

        if collision_points:
            self.position[1] += deepest_penetration
            avg_pt = np.mean(collision_points, axis=0)
            r = avg_pt - self.get_center()
            cv = self.velocity + np.cross(self.angular_velocity, r)
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            vn = np.dot(cv, normal)

            if vn < 0:
                r_cross_n = np.cross(r, normal)
                inv_I = self.get_inv_inertia_world()
                rot_term = np.dot(r_cross_n, inv_I @ r_cross_n)
                inv_mass_sum = self.inv_mass + rot_term
                if inv_mass_sum > 1e-10:
                    e = self.restitution if abs(vn) > 0.2 else 0.0
                    j = -(1.0 + e) * vn / inv_mass_sum
                    impulse = normal * j
                    self.velocity += impulse * self.inv_mass
                    self.angular_velocity += inv_I @ np.cross(r, impulse)

                    # Friction impulse
                    cv_new = self.velocity + np.cross(self.angular_velocity, r)
                    tangent = cv_new - normal * np.dot(cv_new, normal)
                    t_sq = np.dot(tangent, tangent)
                    if t_sq > 1e-6:
                        tangent = tangent / np.sqrt(t_sq)
                        vt = np.dot(cv_new, tangent)
                        r_cross_t = np.cross(r, tangent)
                        inv_mass_tangent = self.inv_mass + np.dot(r_cross_t, inv_I @ r_cross_t)
                        if inv_mass_tangent > 1e-10:
                            jt = -vt / inv_mass_tangent
                            if abs(jt) > abs(j) * self.static_friction:
                                jt = -np.sign(vt) * abs(j) * self.dynamic_friction
                            f_impulse = tangent * jt
                            self.velocity += f_impulse * self.inv_mass
                            self.angular_velocity += inv_I @ np.cross(r, f_impulse)

            if abs(self.velocity[1]) < 0.08 and deepest_penetration < 0.05:
                self.velocity[1] = 0.0
                self.angular_velocity *= 0.95

    def handle_static_box_collisions(self):
        if not statics:
            return
        corners = self.get_vertices()
        c_arr = np.array(corners)
        box_mn = c_arr.min(axis=0)
        box_mx = c_arr.max(axis=0)

        for name, slab in statics.items():
            if name == "ground":
                continue
            mn, mx = slab["mn"], slab["mx"]
            if (box_mx < mn).any() or (box_mn > mx).any():
                continue

            for corner in corners:
                if (corner >= mn).all() and (corner <= mx).all():
                    d_mn = corner - mn
                    d_mx = mx - corner
                    d = np.minimum(d_mn, d_mx)
                    axis = int(np.argmin(d))
                    normal = np.zeros(3, dtype=np.float64)
                    if d_mn[axis] < d_mx[axis]:
                        normal[axis] = -1.0
                        pen = d_mn[axis]
                    else:
                        normal[axis] = 1.0
                        pen = d_mx[axis]

                    self.position += normal * pen
                    r = corner - self.get_center()
                    cv = self.velocity + np.cross(self.angular_velocity, r)
                    vn = np.dot(cv, normal)
                    if vn < 0:
                        r_cross_n = np.cross(r, normal)
                        inv_I = self.get_inv_inertia_world()
                        inv_mass_sum = self.inv_mass + np.dot(r_cross_n, inv_I @ r_cross_n)
                        if inv_mass_sum > 1e-10:
                            j = -(1.0 + self.restitution) * vn / inv_mass_sum
                            imp = normal * j
                            self.velocity += imp * self.inv_mass
                            self.angular_velocity += inv_I @ np.cross(r, imp)

    def handle_mesh_static_collisions(self):
        if not mesh_statics or self.inv_mass == 0:
            return

        corners = self.get_vertices()
        c_arr = np.array(corners)
        box_mn = c_arr.min(axis=0) - 0.05
        box_mx = c_arr.max(axis=0) + 0.05

        for name, ms in mesh_statics.items():
            if name in rigid_bodies or name in static_bodies:
                continue

            v0 = ms["v0"]
            v1 = ms["v1"]
            v2 = ms["v2"]
            normals = ms["normals"]
            plane_d = ms["plane_d"]

            tri_mn = np.minimum(np.minimum(v0, v1), v2)
            tri_mx = np.maximum(np.maximum(v0, v1), v2)
            overlap_mask = (tri_mx >= box_mn) & (tri_mn <= box_mx)
            cand_indices = np.where(overlap_mask.all(axis=1))[0]

            if len(cand_indices) == 0:
                continue

            for idx in cand_indices:
                norm = normals[idx]
                p_d = plane_d[idx]
                tri_v0 = v0[idx]
                tri_v1 = v1[idx]
                tri_v2 = v2[idx]

                for corner in corners:
                    signed_dist = np.dot(norm, corner) - p_d
                    if -0.35 < signed_dist < 0.0:
                        tri_pt = _closest_point_on_triangle(corner, tri_v0, tri_v1, tri_v2)
                        dist = np.linalg.norm(corner - tri_pt)
                        if dist < 0.35:
                            pen = abs(signed_dist) + 0.002
                            self.position += norm * pen
                            r = corner - self.get_center()
                            cv = self.velocity + np.cross(self.angular_velocity, r)
                            vn = np.dot(cv, norm)
                            if vn < 0:
                                r_cross_n = np.cross(r, norm)
                                inv_I = self.get_inv_inertia_world()
                                denom = self.inv_mass + np.dot(r_cross_n, inv_I @ r_cross_n)
                                if denom > 1e-10:
                                    e = self.restitution if abs(vn) > 0.2 else 0.0
                                    j = -(1.0 + e) * vn / denom
                                    imp = norm * j
                                    self.velocity += imp * self.inv_mass
                                    self.angular_velocity += inv_I @ np.cross(r, imp)


def check_box_collision(b1, b2):
    """Separating Axis Theorem (SAT) detection between two oriented rigid bodies."""
    v1 = b1.get_vertices()
    v2 = b2.get_vertices()

    c1_mat = np.array(v1)
    c2_mat = np.array(v2)

    min1_aabb = c1_mat.min(axis=0)
    max1_aabb = c1_mat.max(axis=0)
    min2_aabb = c2_mat.min(axis=0)
    max2_aabb = c2_mat.max(axis=0)

    if (max1_aabb < min2_aabb).any() or (max2_aabb < min1_aabb).any():
        return False

    R1 = b1.rotation_matrix
    R2 = b2.rotation_matrix

    axes = []
    for i in range(3):
        axes.append(R1[:, i])
    for i in range(3):
        axes.append(R2[:, i])
    for i in range(3):
        for j in range(3):
            c = np.cross(R1[:, i], R2[:, j])
            l = np.linalg.norm(c)
            if l > 1e-5:
                axes.append(c / l)

    min_overlap = float('inf')
    collision_normal = None
    normal_from_b1 = True

    for idx, axis in enumerate(axes):
        p1 = c1_mat @ axis
        p2 = c2_mat @ axis
        min1, max1 = p1.min(), p1.max()
        min2, max2 = p2.min(), p2.max()

        if max1 < min2 or max2 < min1:
            return False

        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            collision_normal = axis
            normal_from_b1 = (idx < 3)

    if collision_normal is None:
        return False

    if np.dot(b2.get_center() - b1.get_center(), collision_normal) < 0:
        collision_normal = -collision_normal

    if normal_from_b1:
        projs = [np.dot(v, collision_normal) for v in v2]
        min_p = min(projs)
        contact_pts = [v for v, p in zip(v2, projs) if abs(p - min_p) < 1e-3]
        contact_point = np.mean(contact_pts, axis=0)
    else:
        projs = [np.dot(v, collision_normal) for v in v1]
        max_p = max(projs)
        contact_pts = [v for v, p in zip(v1, projs) if abs(p - max_p) < 1e-3]
        contact_point = np.mean(contact_pts, axis=0)

    resolve_rigid_collision(b1, b2, collision_normal, min_overlap, contact_point)
    return True


def resolve_rigid_collision(b1, b2, normal, overlap, contact_point):
    total_inv_mass = b1.inv_mass + b2.inv_mass
    if total_inv_mass <= 0:
        return

    # Positional Correction
    correction = (normal * overlap) / total_inv_mass
    b1.position -= correction * b1.inv_mass
    b2.position += correction * b2.inv_mass

    r1 = contact_point - b1.get_center()
    r2 = contact_point - b2.get_center()

    v1_contact = b1.velocity + np.cross(b1.angular_velocity, r1)
    v2_contact = b2.velocity + np.cross(b2.angular_velocity, r2)
    relative_velocity = v2_contact - v1_contact

    vn = np.dot(relative_velocity, normal)
    if vn > 0:
        return

    inv_I1 = b1.get_inv_inertia_world()
    inv_I2 = b2.get_inv_inertia_world()

    r1_cross_n = np.cross(r1, normal)
    r2_cross_n = np.cross(r2, normal)

    rot_term1 = np.dot(r1_cross_n, inv_I1 @ r1_cross_n)
    rot_term2 = np.dot(r2_cross_n, inv_I2 @ r2_cross_n)

    e = min(b1.restitution, b2.restitution)
    if abs(vn) < 0.2:
        e = 0.0

    inv_mass_sum = total_inv_mass + rot_term1 + rot_term2
    if inv_mass_sum < 1e-10:
        return

    j = -(1.0 + e) * vn / inv_mass_sum
    impulse = normal * j

    b1.velocity -= impulse * b1.inv_mass
    b1.angular_velocity -= inv_I1 @ np.cross(r1, impulse)
    b2.velocity += impulse * b2.inv_mass
    b2.angular_velocity += inv_I2 @ np.cross(r2, impulse)

    # Friction Impulse
    v1_contact = b1.velocity + np.cross(b1.angular_velocity, r1)
    v2_contact = b2.velocity + np.cross(b2.angular_velocity, r2)
    relative_velocity = v2_contact - v1_contact

    tangent = relative_velocity - (normal * np.dot(relative_velocity, normal))
    tangent_sq = np.dot(tangent, tangent)
    if tangent_sq > 1e-6:
        tangent = tangent / np.sqrt(tangent_sq)
        vt = np.dot(relative_velocity, tangent)

        r1_cross_t = np.cross(r1, tangent)
        r2_cross_t = np.cross(r2, tangent)
        rot_term1_t = np.dot(r1_cross_t, inv_I1 @ r1_cross_t)
        rot_term2_t = np.dot(r2_cross_t, inv_I2 @ r2_cross_t)

        inv_mass_tangent = total_inv_mass + rot_term1_t + rot_term2_t
        if inv_mass_tangent > 1e-10:
            jt = -vt / inv_mass_tangent
            mu = (b1.static_friction + b2.static_friction) / 2.0

            if abs(jt) > abs(j) * mu:
                dynamic_mu = (b1.dynamic_friction + b2.dynamic_friction) / 2.0
                jt = -abs(j) * dynamic_mu if vt > 0 else abs(j) * dynamic_mu

            f_impulse = tangent * jt
            b1.velocity -= f_impulse * b1.inv_mass
            b1.angular_velocity -= inv_I1 @ np.cross(r1, f_impulse)
            b2.velocity += f_impulse * b2.inv_mass
            b2.angular_velocity += inv_I2 @ np.cross(r2, f_impulse)


def add_rigid_body(name, mesh, pos, rot_deg=None, physics_file=None, props=None, root="."):
    combined_props = {}
    if physics_file:
        loaded = load_rigid_body_props(physics_file, root=root)
        combined_props.update(loaded)
    if props:
        combined_props.update(props)

    body = RigidBody(
        name,
        mesh=mesh,
        pos=pos,
        rot=rot_deg if rot_deg is not None else [0, 0, 0],
        props=combined_props
    )
    rigid_bodies[name] = body
    if mesh is not None:
        add_mesh_static(name, mesh, pos, rot_deg)
    return body


def update_rigid_bodies(dt):
    if not rigid_bodies:
        return

    substeps = 4 if dt > 0.005 else 1
    sub_dt = dt / substeps

    dyn_keys = list(rigid_bodies.keys())
    stat_keys = list(static_bodies.keys())

    for _ in range(substeps):
        # 1. Update dynamic bodies' kinematics & ground collisions
        for body in rigid_bodies.values():
            body.update(sub_dt)

        # 2. Dynamic vs Dynamic SAT collisions
        n = len(dyn_keys)
        for i in range(n):
            for j in range(i + 1, n):
                b1 = rigid_bodies[dyn_keys[i]]
                b2 = rigid_bodies[dyn_keys[j]]
                check_box_collision(b1, b2)

        # 3. Dynamic vs Static SAT collisions (walls and static objects)
        for b_dyn in rigid_bodies.values():
            for s_key in stat_keys:
                b_stat = static_bodies[s_key]
                check_box_collision(b_dyn, b_stat)

            # 4. Dynamic vs Mesh-Static triangle collisions (for any non-box triangles)
            b_dyn.handle_mesh_static_collisions()

    # Sync collision mesh for player interaction
    for key, body in rigid_bodies.items():
        if body.mesh is not None:
            add_mesh_static(key, body.mesh, body.position, body.get_rotation_euler())


def find_targeted_body(ray_origin, ray_dir, max_dist=4.5):
    best_body = None
    best_dist = float('inf')

    r_orig = np.asarray(ray_origin, dtype=np.float64)
    r_dir = np.asarray(ray_dir, dtype=np.float64)
    r_dir_len = np.linalg.norm(r_dir)
    if r_dir_len < 1e-6:
        return None
    r_dir = r_dir / r_dir_len

    for body in rigid_bodies.values():
        center = body.get_center()
        to_center = center - r_orig
        dist_along_ray = np.dot(to_center, r_dir)
        if dist_along_ray < 0.2 or dist_along_ray > max_dist:
            continue

        perp = to_center - dist_along_ray * r_dir
        perp_dist = np.linalg.norm(perp)

        radius = float(np.max(body.half_extents) + 0.35)
        if perp_dist <= radius and dist_along_ray < best_dist:
            best_dist = dist_along_ray
            best_body = body

    return best_body


def update_held_object(body, target_pos, dt):
    """
    Half-Life style motion controller (spring-damper) for held physics objects.
    """
    diff = np.asarray(target_pos, dtype=np.float64) - body.get_center()
    dist = np.linalg.norm(diff)
    if dist > 4.5:
        return False  # Lost grip (snagged or too far away)

    # Smooth tracking velocity
    body.velocity = diff * 12.0
    body.angular_velocity *= 0.88
    # Counteract gravity while held so it floats steadily
    if body.use_gravity:
        body.velocity[1] -= GRAVITY * dt

    return True


def throw_held_object(body, throw_dir, player_vel=None, throw_speed=18.0):
    t_dir = np.asarray(throw_dir, dtype=np.float64)
    norm = np.linalg.norm(t_dir)
    if norm > 1e-6:
        t_dir = t_dir / norm
    body.velocity = t_dir * throw_speed
    if player_vel is not None:
        body.velocity += np.asarray(player_vel, dtype=np.float64) * 0.5
    body.angular_velocity = np.random.uniform(-3.0, 3.0, 3)


def drop_held_object(body, player_vel=None):
    if player_vel is not None:
        body.velocity = np.asarray(player_vel, dtype=np.float64) * 0.5
    else:
        body.velocity *= 0.2


def add_static(name, mn, mx):
    statics[name] = {
        "mn": np.array(mn, dtype=np.float64),
        "mx": np.array(mx, dtype=np.float64),
    }


def add_mesh_static(name, mesh, pos, rot_deg=None):
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

    # Automatically register static collision body for OBB SAT
    if name not in rigid_bodies and name not in static_bodies:
        static_bodies[name] = RigidBody(
            name,
            mesh=mesh,
            pos=pos,
            rot=rot_deg if rot_deg is not None else [0, 0, 0],
            props={"mass": 0.0, "use_gravity": False, "restitution": 0.7, "static_friction": 0.3, "dynamic_friction": 0.2}
        )


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

        for name, ms in mesh_statics.items():
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

                    # Transfer push impulse if dynamic rigid body
                    if name in rigid_bodies and collision_normal[1] <= _FLOOR_THRESH:
                        b = rigid_bodies[name]
                        push_dir = -collision_normal
                        push_dir[1] = 0.0
                        norm_p = np.linalg.norm(push_dir)
                        if norm_p > 1e-4:
                            push_dir /= norm_p
                            push_speed = max(0.5, -vel_into_surface)
                            b.velocity += push_dir * (push_speed * (0.8 / max(b.mass, 0.5)))

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
