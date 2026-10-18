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
        physics_dir = os.path.join(root, "physics")
        basename = os.path.basename(file_path_or_dict)
        name_no_ext = os.path.splitext(basename)[0]

        candidates = [
            os.path.join(physics_dir, file_path_or_dict),
            os.path.join(physics_dir, basename),
            os.path.join(physics_dir, f"{basename}.json"),
            os.path.join(physics_dir, f"{name_no_ext}.json"),
            os.path.join(physics_dir, f"{name_no_ext}.physics.json"),
            os.path.join(physics_dir, f"{name_no_ext}_physics.json"),
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
        self.local_vertices = None
        self.is_cylinder = False

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
            flat = verts.reshape(-1, 3).astype(np.float64)
            mn, mx = flat.min(axis=0), flat.max(axis=0)
            raw_half = (mx - mn) * 0.5
            self.center_offset = ((mx + mn) * 0.5).astype(np.float64)
            if raw_half[1] < 1e-3:
                # Flat horizontal surface (e.g. floor): keep top surface at mx[1]
                self.center_offset[1] = mx[1] - 0.05
            self.half_extents = np.maximum(raw_half, 0.05).astype(np.float64)

            if self.inv_mass > 0 and np.all(raw_half > 1e-3):
                rel = flat - self.center_offset
                uniq = np.unique(np.round(rel, 5), axis=0)
                if 8 < len(uniq) <= 256:
                    # Exclude interior cap-center points on cylinders so only hull perimeter vertices remain
                    radial_xz = np.hypot(uniq[:, 0], uniq[:, 2])
                    max_r = radial_xz.max()
                    if max_r > 1e-3 and np.any(radial_xz < max_r * 0.1):
                        uniq = uniq[radial_xz >= max_r * 0.1]
                        self.is_cylinder = True
                    self.local_vertices = uniq
        else:
            self.half_extents = np.array([0.5, 0.5, 0.5], dtype=np.float64)
            self.center_offset = np.zeros(3, dtype=np.float64)

        if self.local_vertices is None:
            ex, ey, ez = self.half_extents
            self.local_vertices = np.array([
                [-ex, -ey, -ez],
                [ ex, -ey, -ez],
                [ ex,  ey, -ez],
                [-ex,  ey, -ez],
                [-ex, -ey,  ez],
                [ ex, -ey,  ez],
                [ ex,  ey,  ez],
                [-ex,  ey,  ez],
            ], dtype=np.float64)

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
        elif self.is_cylinder:
            ex, ey, ez = self.half_extents
            r = 0.5 * (ex + ez)
            h = 2.0 * ey
            I_yy = 0.5 * self.mass * (r ** 2)
            I_xx = (1.0 / 12.0) * self.mass * (3.0 * (r ** 2) + (h ** 2))
            I_zz = I_xx
            self.inertia_body = np.diag([I_xx, I_yy, I_zz])
            self.inv_inertia_body = np.diag([1.0 / I_xx, 1.0 / I_yy, 1.0 / I_zz])
        else:
            ex, ey, ez = self.half_extents
            I_xx = (1.0 / 3.0) * self.mass * (ey**2 + ez**2)
            I_yy = (1.0 / 3.0) * self.mass * (ex**2 + ez**2)
            I_zz = (1.0 / 3.0) * self.mass * (ex**2 + ey**2)
            self.inertia_body = np.diag([I_xx, I_yy, I_zz])
            self.inv_inertia_body = np.diag([1.0 / I_xx, 1.0 / I_yy, 1.0 / I_zz])

        self.local_face_normals = np.eye(3, dtype=np.float64)
        self.local_edge_dirs = np.eye(3, dtype=np.float64)
        if mesh is not None:
            verts_f = mesh.get("verts")
            if verts_f is None:
                verts_f = np.array(mesh["tris"], dtype=np.float32)
            tris_f = verts_f.reshape(-1, 3, 3).astype(np.float64)
            if len(tris_f) > 0:
                fn = np.cross(tris_f[:, 1] - tris_f[:, 0], tris_f[:, 2] - tris_f[:, 0])
                fl = np.linalg.norm(fn, axis=1, keepdims=True)
                valid_fn = fl[:, 0] > 1e-10
                if np.any(valid_fn):
                    fn = fn[valid_fn] / fl[valid_fn]
                    # Keep unique plane normal directions (up to sign)
                    uniq_fn = []
                    for n_vec in fn:
                        if not any(abs(np.dot(n_vec, u_vec)) > 0.999 for u_vec in uniq_fn):
                            uniq_fn.append(n_vec)
                    if len(uniq_fn) <= 32:
                        self.local_face_normals = np.array(uniq_fn, dtype=np.float64)

        self.local_face_dists = np.max(np.abs(self.local_vertices @ self.local_face_normals.T), axis=0)

        if self.is_cylinder:
            edge_dirs = [[0.0, 1.0, 0.0]]
            for n_vec in self.local_face_normals:
                if abs(n_vec[1]) < 0.5:
                    t_vec = np.array([-n_vec[2], 0.0, n_vec[0]], dtype=np.float64)
                    tl = np.linalg.norm(t_vec)
                    if tl > 1e-6:
                        edge_dirs.append((t_vec / tl).tolist())
            self.local_edge_dirs = np.array(edge_dirs, dtype=np.float64)

        n_v = len(self.local_vertices)
        hull_edges = []
        if self.is_cylinder and n_v >= 16:
            y_vals = self.local_vertices[:, 1]
            bot_idx = np.where(y_vals < 0.0)[0]
            top_idx = np.where(y_vals >= 0.0)[0]
            if len(bot_idx) >= 3:
                ang_b = np.arctan2(self.local_vertices[bot_idx, 2], self.local_vertices[bot_idx, 0])
                bot_sorted = bot_idx[np.argsort(ang_b)]
                for k in range(len(bot_sorted)):
                    hull_edges.append([int(bot_sorted[k]), int(bot_sorted[(k + 1) % len(bot_sorted)])])
            if len(top_idx) >= 3:
                ang_t = np.arctan2(self.local_vertices[top_idx, 2], self.local_vertices[top_idx, 0])
                top_sorted = top_idx[np.argsort(ang_t)]
                for k in range(len(top_sorted)):
                    hull_edges.append([int(top_sorted[k]), int(top_sorted[(k + 1) % len(top_sorted)])])
                if len(bot_idx) == len(top_idx):
                    for k in range(len(bot_sorted)):
                        hull_edges.append([int(bot_sorted[k]), int(top_sorted[k])])
        else:
            for i_v in range(n_v):
                for j_v in range(i_v + 1, n_v):
                    diff = np.abs(self.local_vertices[i_v] - self.local_vertices[j_v])
                    if np.sum(diff > 1e-4) == 1:
                        hull_edges.append([i_v, j_v])
        self.local_hull_edges = np.array(hull_edges, dtype=np.int32) if hull_edges else np.zeros((0, 2), dtype=np.int32)

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
        old_center = self.get_center()
        self.rotation_matrix = rotation_matrix(angles).astype(np.float64)
        self.position = old_center - self.rotation_matrix @ self.center_offset

    def _integrate_orientation(self, dt):
        omega = self.angular_velocity
        speed = np.linalg.norm(omega)
        if speed > 1e-10:
            old_center = self.get_center()
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
            self.position = old_center - self.rotation_matrix @ self.center_offset

    def get_vertices(self):
        center = self.get_center()
        world_verts = (self.local_vertices @ self.rotation_matrix.T) + center
        return list(world_verts)

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

        self.handle_mesh_static_collisions()
        if not mesh_statics:
            self.handle_ground_collisions()
        self.handle_static_box_collisions()

    def _resolve_plane_contact(self, normal, penetration, manifold_pts):
        self.position += normal * penetration
        shifted_pts = manifold_pts + normal * penetration
        cop_pt = _compute_center_of_pressure(self.get_center(), shifted_pts, normal)
        r = cop_pt - self.get_center()
        cv = self.velocity + np.cross(self.angular_velocity, r)
        vn = float(np.dot(cv, normal))

        if vn < 0.0:
            r_cross_n = np.cross(r, normal)
            inv_I = self.get_inv_inertia_world()
            rot_term = float(np.dot(r_cross_n, inv_I @ r_cross_n))
            inv_mass_sum = self.inv_mass + rot_term
            if inv_mass_sum > 1e-10:
                e = self.restitution if abs(vn) > 0.25 else 0.0
                j = -(1.0 + e) * vn / inv_mass_sum
                impulse = normal * j
                self.velocity += impulse * self.inv_mass
                self.angular_velocity += inv_I @ np.cross(r, impulse)

                # Coulomb friction impulse along the plane
                cv_new = self.velocity + np.cross(self.angular_velocity, r)
                tangent = cv_new - normal * np.dot(cv_new, normal)
                t_sq = float(np.dot(tangent, tangent))
                if t_sq > 1e-8:
                    tangent = tangent / np.sqrt(t_sq)
                    vt = float(np.dot(cv_new, tangent))
                    r_cross_t = np.cross(r, tangent)
                    inv_mass_tangent = self.inv_mass + float(np.dot(r_cross_t, inv_I @ r_cross_t))
                    if inv_mass_tangent > 1e-10:
                        jt = -vt / inv_mass_tangent
                        if abs(jt) > abs(j) * self.static_friction:
                            jt = -np.sign(vt) * abs(j) * self.dynamic_friction
                        f_impulse = tangent * jt
                        self.velocity += f_impulse * self.inv_mass
                        self.angular_velocity += inv_I @ np.cross(r, f_impulse)

        # Settle micro-vibrations only when supported underneath CoM on >= 2 contact points in a flat resting pose
        r_tangent = r - normal * np.dot(r, normal)
        flat_pose = (
            not self.is_cylinder or
            abs(np.dot(self.rotation_matrix[:, 1], normal)) > 0.98 or
            abs(np.dot(self.rotation_matrix[:, 1], normal)) < 0.05
        )
        if flat_pose and len(manifold_pts) >= 2 and np.linalg.norm(r_tangent) < 0.12 and np.linalg.norm(self.angular_velocity) < 0.25:
            vn_after = float(np.dot(self.velocity, normal))
            if abs(vn_after) < 0.08 and penetration < 0.02:
                self.velocity -= normal * vn_after
                self.angular_velocity *= 0.85

    def handle_ground_collisions(self):
        c_arr = (self.local_vertices @ self.rotation_matrix.T) + self.get_center()
        ground_y = 0.0
        min_y = float(c_arr[:, 1].min())

        if min_y < ground_y:
            deepest_penetration = ground_y - min_y
            manifold_mask = c_arr[:, 1] <= (min_y + 0.002)
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            self._resolve_plane_contact(normal, deepest_penetration, c_arr[manifold_mask])

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
                    self._resolve_plane_contact(normal, float(pen), np.array([corner]))

    def handle_mesh_static_collisions(self):
        if not mesh_statics or self.inv_mass == 0:
            return

        c_arr = (self.local_vertices @ self.rotation_matrix.T) + self.get_center()
        box_mn = c_arr.min(axis=0) - 0.05
        box_mx = c_arr.max(axis=0) + 0.05

        # For each penetrating vertex, keep only its shallowest triangle plane contact
        vert_best_plane = {}

        for name, ms in mesh_statics.items():
            if name in rigid_bodies or ms.get("is_convex_solid", False):
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
                p_d = float(plane_d[idx])
                signed_dists = c_arr @ norm - p_d

                pen_indices = np.where((signed_dists < 0.0) & (signed_dists > -0.12))[0]
                if len(pen_indices) == 0:
                    continue

                tri_v0 = v0[idx]
                tri_v1 = v1[idx]
                tri_v2 = v2[idx]

                for v_idx in pen_indices:
                    corner = c_arr[v_idx]
                    s_dist = float(signed_dists[v_idx])
                    proj_pt = corner - norm * s_dist
                    tri_pt = _closest_point_on_triangle(proj_pt, tri_v0, tri_v1, tri_v2)
                    if np.linalg.norm(proj_pt - tri_pt) <= 0.02:
                        pen = -s_dist
                        key = (round(float(norm[0]), 3), round(float(norm[1]), 3), round(float(norm[2]), 3), round(p_d, 2))
                        prev = vert_best_plane.get(v_idx)
                        if prev is None or pen < prev[0]:
                            vert_best_plane[v_idx] = (pen, key, norm, corner)

        plane_contacts = {}
        for v_idx, (pen, key, norm, corner) in vert_best_plane.items():
            if key not in plane_contacts:
                plane_contacts[key] = {"normal": norm, "verts": {}}
            plane_contacts[key]["verts"][v_idx] = (pen, corner)

        for info in plane_contacts.values():
            vert_map = info["verts"]
            if not vert_map:
                continue
            normal = info["normal"]
            max_pen = max(v[0] for v in vert_map.values())
            manifold_pts = np.array([v[1] for v in vert_map.values() if v[0] >= max_pen - 0.003], dtype=np.float64)
            self._resolve_plane_contact(normal, max_pen, manifold_pts)


def _clip_edges_to_body(p0_world, p1_world, body, tol=0.004):
    """Vectorized Liang-Barsky clipping of 3D hull edge segments [p0, p1] against body's local convex planes."""
    if len(p0_world) == 0:
        return np.empty((0, 3), dtype=np.float64)

    center = body.get_center()
    R = body.rotation_matrix
    p0 = (p0_world - center) @ R
    p1 = (p1_world - center) @ R
    dp = p1 - p0

    N = body.local_face_normals          # (K, 3)
    limits = body.local_face_dists + tol # (K,)

    x0 = p0 @ N.T                        # (E, K)
    dx = dp @ N.T                        # (E, K)

    parallel = np.abs(dx) < 1e-10
    outside_parallel = parallel & (np.abs(x0) > limits[None, :])

    safe_dx = np.where(parallel, 1.0, dx)
    t1 = (-limits[None, :] - x0) / safe_dx
    t2 = ( limits[None, :] - x0) / safe_dx

    t_enter = np.where(parallel, 0.0, np.minimum(t1, t2))
    t_exit  = np.where(parallel, 1.0, np.maximum(t1, t2))
    t_enter = np.where(outside_parallel, 2.0, t_enter)
    t_exit  = np.where(outside_parallel, -1.0, t_exit)

    t_min = np.maximum(0.0, np.max(t_enter, axis=1))
    t_max = np.minimum(1.0, np.min(t_exit, axis=1))

    valid = t_min <= t_max + 1e-7
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)

    v_p0 = p0_world[valid]
    v_dp = (p1_world - p0_world)[valid]
    v_tmin = np.clip(t_min[valid], 0.0, 1.0)[:, None]
    v_tmax = np.clip(t_max[valid], 0.0, 1.0)[:, None]

    pts_start = v_p0 + v_tmin * v_dp
    pts_end   = v_p0 + v_tmax * v_dp
    return np.vstack((pts_start, pts_end))


def _closest_point_between_features(pts1, pts2):
    """Finds the closest midpoint between two vertex/segment features in 3D."""
    if len(pts1) == 1 and len(pts2) == 1:
        return 0.5 * (pts1[0] + pts2[0])
    if len(pts1) == 1 and len(pts2) >= 2:
        p = pts1[0]
        d = pts2 - pts2[0]
        end_idx = int(np.argmax(np.einsum("ij,ij->i", d, d)))
        q = _closest_point_on_segment(p, pts2[0], pts2[end_idx])
        return 0.5 * (p + q)
    if len(pts2) == 1 and len(pts1) >= 2:
        q = pts2[0]
        d = pts1 - pts1[0]
        end_idx = int(np.argmax(np.einsum("ij,ij->i", d, d)))
        p = _closest_point_on_segment(q, pts1[0], pts1[end_idx])
        return 0.5 * (p + q)

    # Segment vs segment
    d1_all = pts1 - pts1[0]
    a0, a1 = pts1[0], pts1[int(np.argmax(np.einsum("ij,ij->i", d1_all, d1_all)))]
    d2_all = pts2 - pts2[0]
    b0, b1 = pts2[0], pts2[int(np.argmax(np.einsum("ij,ij->i", d2_all, d2_all)))]

    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denom = a * c - b * b

    if denom > 1e-8:
        s = np.clip((b * e - c * d) / denom, 0.0, 1.0)
    else:
        s = 0.5
    p_seg1 = a0 + s * u
    p_seg2 = _closest_point_on_segment(p_seg1, b0, b1)
    p_seg1 = _closest_point_on_segment(p_seg2, a0, a1)
    return 0.5 * (p_seg1 + p_seg2)


def _compute_center_of_pressure(body_center, manifold_pts, normal):
    """Finds the physical Center of Pressure: closest point in the convex hull of manifold_pts to the projected CoM."""
    if len(manifold_pts) == 0:
        return body_center.copy()
    if len(manifold_pts) == 1:
        return manifold_pts[0].copy()

    avg_pt = manifold_pts.mean(axis=0)
    p_com = body_center + normal * np.dot(avg_pt - body_center, normal)

    offsets = (manifold_pts - avg_pt) @ normal
    plane_pts = manifold_pts - np.outer(offsets, normal)
    uniq = np.unique(np.round(plane_pts, 4), axis=0)

    if len(uniq) == 1:
        return uniq[0]
    if len(uniq) == 2:
        return _closest_point_on_segment(p_com, uniq[0], uniq[1])

    centroid = uniq.mean(axis=0)
    rel = uniq - centroid
    dists_sq = np.einsum("ij,ij->i", rel, rel)
    max_idx = int(np.argmax(dists_sq))
    if dists_sq[max_idx] < 1e-8:
        return centroid

    u = rel[max_idx] / np.sqrt(dists_sq[max_idx])
    v = np.cross(normal, u)
    v_len = np.linalg.norm(v)
    if v_len < 1e-8:
        return _closest_point_on_segment(p_com, uniq[0], uniq[max_idx])
    v = v / v_len

    angles = np.arctan2(rel @ v, rel @ u)
    ordered = uniq[np.argsort(angles)]
    m = len(ordered)

    best_pt = centroid
    best_dist_sq = float('inf')
    for k in range(m):
        cand = _closest_point_on_triangle(p_com, centroid, ordered[k], ordered[(k + 1) % m])
        diff = p_com - cand
        d_sq = float(np.dot(diff, diff))
        if d_sq < best_dist_sq:
            best_dist_sq = d_sq
            best_pt = cand
            if d_sq < 1e-10:
                break
    return best_pt


def check_box_collision(b1, b2):
    """3D Convex SAT with plane normals, rim/box edge cross products, and Liang-Barsky 3D intersection manifold."""
    c1_mat = (b1.local_vertices @ b1.rotation_matrix.T) + b1.get_center()
    c2_mat = (b2.local_vertices @ b2.rotation_matrix.T) + b2.get_center()

    min1_aabb = c1_mat.min(axis=0)
    max1_aabb = c1_mat.max(axis=0)
    min2_aabb = c2_mat.min(axis=0)
    max2_aabb = c2_mat.max(axis=0)

    if (max1_aabb < min2_aabb).any() or (max2_aabb < min1_aabb).any():
        return False

    R1 = b1.rotation_matrix
    R2 = b2.rotation_matrix

    b1_normals = b1.local_face_normals @ R1.T
    b2_normals = b2.local_face_normals @ R2.T
    b1_edges = b1.local_edge_dirs @ R1.T
    b2_edges = b2.local_edge_dirs @ R2.T

    face_axes = []
    if b2.inv_mass == 0 and b1.inv_mass > 0:
        for n_vec in b2_normals:
            face_axes.append((n_vec, False))
        for n_vec in b1_normals:
            face_axes.append((n_vec, True))
    else:
        for n_vec in b1_normals:
            face_axes.append((n_vec, True))
        for n_vec in b2_normals:
            face_axes.append((n_vec, False))

    min_face_overlap = float('inf')
    face_normal = None
    face_from_b1 = True

    for axis, from_b1 in face_axes:
        p1 = c1_mat @ axis
        p2 = c2_mat @ axis
        min1, max1 = p1.min(), p1.max()
        min2, max2 = p2.min(), p2.max()
        if max1 < min2 or max2 < min1:
            return False
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_face_overlap - 1e-5:
            min_face_overlap = overlap
            face_normal = axis
            face_from_b1 = from_b1

    min_edge_overlap = float('inf')
    edge_normal = None

    for e1 in b1_edges:
        for e2 in b2_edges:
            c = np.cross(e1, e2)
            l = np.linalg.norm(c)
            if l > 1e-3:
                axis = c / l
                p1 = c1_mat @ axis
                p2 = c2_mat @ axis
                min1, max1 = p1.min(), p1.max()
                min2, max2 = p2.min(), p2.max()
                if max1 < min2 or max2 < min1:
                    return False
                overlap = min(max1, max2) - max(min1, min2)
                if overlap < min_edge_overlap - 1e-5:
                    min_edge_overlap = overlap
                    edge_normal = axis

    if edge_normal is not None and min_edge_overlap < min_face_overlap * 0.98 - 5e-5:
        min_overlap = min_edge_overlap
        collision_normal = edge_normal
        normal_from_b1 = False
        from_face = False
    else:
        min_overlap = min_face_overlap
        collision_normal = face_normal
        normal_from_b1 = face_from_b1
        from_face = True

    if collision_normal is None:
        return False

    p1 = c1_mat @ collision_normal
    p2 = c2_mat @ collision_normal
    mid1 = 0.5 * (p1.min() + p1.max())
    mid2 = 0.5 * (p2.min() + p2.max())
    if mid2 < mid1:
        collision_normal = -collision_normal
        p1 = -p1
        p2 = -p2

    max_p1 = float(p1.max())
    min_p2 = float(p2.min())

    e1_idx = b1.local_hull_edges
    e2_idx = b2.local_hull_edges
    clipped1 = _clip_edges_to_body(c1_mat[e1_idx[:, 0]], c1_mat[e1_idx[:, 1]], b2, tol=0.006) if len(e1_idx) > 0 else np.empty((0, 3), dtype=np.float64)
    clipped2 = _clip_edges_to_body(c2_mat[e2_idx[:, 0]], c2_mat[e2_idx[:, 1]], b1, tol=0.006) if len(e2_idx) > 0 else np.empty((0, 3), dtype=np.float64)

    if len(clipped1) == 0 and len(clipped2) == 0:
        return False

    deep1 = np.empty((0, 3), dtype=np.float64)
    if len(clipped1) > 0:
        cp1 = clipped1 @ collision_normal
        c1_max = float(cp1.max())
        if c1_max >= min_p2 - 0.008:
            deep1 = clipped1[cp1 >= c1_max - 0.005]
            if from_face and not normal_from_b1:
                clipped_pen = c1_max - min_p2
                if 0.0 < clipped_pen < min_overlap:
                    min_overlap = clipped_pen

    deep2 = np.empty((0, 3), dtype=np.float64)
    if len(clipped2) > 0:
        cp2 = clipped2 @ collision_normal
        c2_min = float(cp2.min())
        if c2_min <= max_p1 + 0.008:
            deep2 = clipped2[cp2 <= c2_min + 0.005]
            if from_face and normal_from_b1:
                clipped_pen = max_p1 - c2_min
                if 0.0 < clipped_pen < min_overlap:
                    min_overlap = clipped_pen

    if len(deep1) > 0 and len(deep2) > 0:
        manifold_pts = np.vstack((deep1, deep2))
    elif len(deep1) > 0:
        manifold_pts = deep1
    elif len(deep2) > 0:
        manifold_pts = deep2
    else:
        all_clipped = np.vstack([arr for arr in (clipped1, clipped2) if len(arr) > 0])
        manifold_pts = all_clipped

    ref_center = b1.get_center() if b1.inv_mass > 0 else b2.get_center()
    contact_point = _compute_center_of_pressure(ref_center, manifold_pts, collision_normal)

    resolve_rigid_collision(b1, b2, collision_normal, min_overlap, contact_point, manifold_pts)
    return True


def resolve_rigid_collision(b1, b2, normal, overlap, contact_point, manifold_pts=None):
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
    if vn < 0:
        inv_I1 = b1.get_inv_inertia_world()
        inv_I2 = b2.get_inv_inertia_world()

        r1_cross_n = np.cross(r1, normal)
        r2_cross_n = np.cross(r2, normal)

        rot_term1 = np.dot(r1_cross_n, inv_I1 @ r1_cross_n)
        rot_term2 = np.dot(r2_cross_n, inv_I2 @ r2_cross_n)

        e = min(b1.restitution, b2.restitution)
        if abs(vn) < 0.25:
            e = 0.0

        inv_mass_sum = total_inv_mass + rot_term1 + rot_term2
        if inv_mass_sum > 1e-10:
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
            if tangent_sq > 1e-8:
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

    # Settle micro-vibrations when b1 rests on static b2 in a flat equilibrium pose
    if b2.inv_mass == 0 and manifold_pts is not None and len(manifold_pts) >= 2:
        outward_n = -normal
        r1_tangent = r1 - outward_n * np.dot(r1, outward_n)
        flat_pose = (
            not b1.is_cylinder or
            abs(np.dot(b1.rotation_matrix[:, 1], outward_n)) > 0.98 or
            abs(np.dot(b1.rotation_matrix[:, 1], outward_n)) < 0.05
        )
        if flat_pose and np.linalg.norm(r1_tangent) < 0.12 and np.linalg.norm(b1.angular_velocity) < 0.25:
            vn_after = float(np.dot(b1.velocity, outward_n))
            if abs(vn_after) < 0.08 and overlap < 0.02:
                b1.velocity -= outward_n * vn_after
                b1.angular_velocity *= 0.85


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

        # 3. Dynamic vs Static convex SAT & plane-normal collisions
        for b_dyn in rigid_bodies.values():
            for s_key in stat_keys:
                if s_key not in mesh_statics or mesh_statics[s_key].get("is_convex_solid", False):
                    check_box_collision(b_dyn, static_bodies[s_key])
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
    all_pts = triangles.reshape(-1, 3)
    is_convex_solid = bool(
        len(normals) >= 4 and
        np.all((all_pts @ normals.T - plane_d[None, :]) <= 1e-3)
    )

    mesh_statics[name] = {
        "normals":         normals,
        "plane_d":         plane_d,
        "v0":              v0,
        "v1":              v1,
        "v2":              v2,
        "is_convex_solid": is_convex_solid,
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

            box_mn = center - np.array([PLAYER_R + 0.1, PLAYER_H / 2.0 + PLAYER_R + 0.1, PLAYER_R + 0.1])
            box_mx = center + np.array([PLAYER_R + 0.1, PLAYER_H / 2.0 + PLAYER_R + 0.1, PLAYER_R + 0.1])
            tri_mn = np.minimum(np.minimum(v0, v1), v2)
            tri_mx = np.maximum(np.maximum(v0, v1), v2)
            overlap_mask = (tri_mx >= box_mn) & (tri_mn <= box_mx)

            signed_dist = normals @ center - plane_d

            candidates = np.where(overlap_mask.all(axis=1) & (np.abs(signed_dist) < (PLAYER_H / 2.0 + PLAYER_R + 0.1)))[0]

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
