import struct
import numpy as np
import cv2
import pygame
import moderngl

_MAGIC   = b"QWZX"
_VERSION = 1

_USE_GPU_COMPUTE = True
_gpu_ctx = None
_gpu_geom_shader = None
_gpu_compute_shader = None
_gpu_color_tex = None
_gpu_readback_buf = None
_gpu_ssbo = None
_gpu_ssbo_cap = 0
_gpu_raw_tri_ssbo = None
_gpu_raw_tri_cap = 0
_gpu_instance_ssbo = None
_gpu_instance_cap = 0
_gpu_counter_ssbo = None
_gpu_tex_data_ssbo = None
_gpu_tex_meta_ssbo = None
_gpu_raw_tri_chunks = []
_gpu_total_cached_tris = 0
_gpu_textures = {}
_gpu_tex_slot_cache = {}
_gpu_tex_refs = []
_gpu_tex_meta_list = []
_gpu_tex_data_chunks = []
_gpu_total_texels = 0
_gpu_w = 0
_gpu_h = 0

SHADER_MODES = ("lit", "toon", "normals", "unlit")
_SHADER_MODE = "lit"

_LIGHT_DIR = np.array([0.55, 0.85, 0.45], dtype=np.float32)
_LIGHT_DIR /= np.linalg.norm(_LIGHT_DIR)
_LIGHT_COLOR = np.array([1.0, 0.98, 0.92], dtype=np.float32)
_AMBIENT_SKY = np.array([0.38, 0.42, 0.50], dtype=np.float32)
_AMBIENT_GROUND = np.array([0.18, 0.16, 0.14], dtype=np.float32)


def set_gpu_acceleration(enabled: bool):
    global _USE_GPU_COMPUTE
    _USE_GPU_COMPUTE = enabled


def is_gpu_acceleration_enabled() -> bool:
    return _USE_GPU_COMPUTE and (_gpu_ctx is not None or _init_gpu_compute(640, 640))


def set_shader_mode(mode: str):
    global _SHADER_MODE
    mode_clean = mode.strip().lower()
    if mode_clean not in SHADER_MODES:
        raise ValueError(f"Unknown shader mode '{mode}'. Choose from {SHADER_MODES}")
    _SHADER_MODE = mode_clean


def get_shader_mode() -> str:
    return _SHADER_MODE


def cycle_shader_mode() -> str:
    global _SHADER_MODE
    idx = (SHADER_MODES.index(_SHADER_MODE) + 1) % len(SHADER_MODES)
    _SHADER_MODE = SHADER_MODES[idx]
    return _SHADER_MODE


def set_light_params(light_dir=None, light_color=None, ambient_sky=None, ambient_ground=None):
    global _LIGHT_DIR, _LIGHT_COLOR, _AMBIENT_SKY, _AMBIENT_GROUND
    if light_dir is not None:
        d = np.asarray(light_dir, dtype=np.float32)
        n = np.linalg.norm(d)
        if n > 1e-8:
            _LIGHT_DIR = d / n
    if light_color is not None:
        _LIGHT_COLOR = np.asarray(light_color, dtype=np.float32)
    if ambient_sky is not None:
        _AMBIENT_SKY = np.asarray(ambient_sky, dtype=np.float32)
    if ambient_ground is not None:
        _AMBIENT_GROUND = np.asarray(ambient_ground, dtype=np.float32)


def get_submeshes(mesh_or_list):
    if isinstance(mesh_or_list, (list, tuple)):
        out = []
        for item in mesh_or_list:
            out.extend(get_submeshes(item))
        return out
    if isinstance(mesh_or_list, dict):
        subs = mesh_or_list.get("submeshes")
        if subs and not (len(subs) == 1 and subs[0] is mesh_or_list):
            out = []
            for sm in subs:
                if sm is mesh_or_list:
                    out.append(sm)
                else:
                    out.extend(get_submeshes(sm))
            return out
        return [mesh_or_list]
    return []


def combine_submeshes(meshes_list, name=None):
    subs = get_submeshes(meshes_list)
    if not subs:
        return {"name": name or "empty", "tris": [], "verts": np.zeros((0, 3, 3), dtype=np.float32),
                "colors": None, "texture": None, "uvs": None, "uv_arr": None, "submeshes": []}
    for sm in subs:
        _prepare_single_mesh(sm)
    if len(subs) == 1:
        single = subs[0]
        if name:
            single["name"] = name
        single["submeshes"] = [single]
        return single

    all_tris = []
    vert_blocks = []
    for sm in subs:
        all_tris.extend(sm["tris"])
        if len(sm["verts"]) > 0:
            vert_blocks.append(sm["verts"])
    all_verts = np.concatenate(vert_blocks, axis=0) if vert_blocks else np.zeros((0, 3, 3), dtype=np.float32)
    return {
        "name":      name or subs[0].get("name", "multi_mesh"),
        "tris":      all_tris,
        "verts":     all_verts,
        "colors":    None,
        "texture":   None,
        "uvs":       None,
        "uv_arr":    None,
        "submeshes": subs,
    }


def save_qwzx(path, meshes):
    mesh_list = get_submeshes(meshes)
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("B", _VERSION))
        f.write(struct.pack("<I", len(mesh_list)))
        for mesh in mesh_list:
            name_bytes = mesh.get("name", "mesh").encode("utf-8")
            f.write(struct.pack("B", len(name_bytes)))
            f.write(name_bytes)

            texture = mesh.get("texture")
            tris = mesh["tris"]
            colors = mesh.get("colors")
            mesh_alpha = float(mesh.get("alpha", 1.0))

            if texture is not None:
                channels = texture.shape[2] if texture.ndim == 3 else 1
                has_alpha_tint = (colors is not None and len(colors) == len(tris)) or (mesh_alpha < 0.999)
                if has_alpha_tint:
                    # Convert texture to 4-channel BGRA if needed and store per-tri RGBA tint (flags = 3)
                    if channels == 3:
                        h, w = texture.shape[:2]
                        bgra = np.empty((h, w, 4), dtype=np.uint8)
                        bgra[:, :, :3] = texture
                        bgra[:, :, 3] = 255
                        tex_to_write = bgra
                    else:
                        tex_to_write = texture[:, :, :4]
                    flags = 3
                elif channels == 4:
                    tex_to_write = texture[:, :, :4]
                    flags = 2
                else:
                    tex_to_write = texture[:, :, :3]
                    flags = 1

                f.write(struct.pack("B", flags))
                f.write(struct.pack("<I", len(tris)))
                h, w = tex_to_write.shape[:2]
                f.write(struct.pack("<II", w, h))
                f.write(np.ascontiguousarray(tex_to_write, dtype=np.uint8).tobytes())
                uvs = mesh["uvs"]
                for idx, (tri, uv) in enumerate(zip(tris, uvs)):
                    for vx, vy, vz in tri:
                        f.write(struct.pack("<fff", vx, vy, vz))
                    for u, v in uv:
                        f.write(struct.pack("<ff", u, v))
                    if flags == 3:
                        c = colors[idx] if (colors is not None and idx < len(colors)) else (255, 255, 255, 255)
                        ca = int(round((c[3] if len(c) >= 4 else 255) * mesh_alpha))
                        f.write(struct.pack("BBBB",
                                            int(np.clip(c[0], 0, 255)),
                                            int(np.clip(c[1], 0, 255)),
                                            int(np.clip(c[2], 0, 255)),
                                            int(np.clip(ca, 0, 255))))
            else:
                has_rgba = (mesh_alpha < 0.999) or any(len(c) >= 4 and int(c[3]) < 255 for c in (colors or []))
                flags = 4 if has_rgba else 0
                f.write(struct.pack("B", flags))
                f.write(struct.pack("<I", len(tris)))
                for idx, tri in enumerate(tris):
                    color = colors[idx] if (colors is not None and idx < len(colors)) else (200, 200, 200)
                    for vx, vy, vz in tri:
                        f.write(struct.pack("<fff", vx, vy, vz))
                    if flags == 4:
                        ca = int(round((color[3] if len(color) >= 4 else 255) * mesh_alpha))
                        f.write(struct.pack("BBBB",
                                            int(np.clip(color[0], 0, 255)),
                                            int(np.clip(color[1], 0, 255)),
                                            int(np.clip(color[2], 0, 255)),
                                            int(np.clip(ca, 0, 255))))
                    else:
                        f.write(struct.pack("BBB",
                                            int(np.clip(color[0], 0, 255)),
                                            int(np.clip(color[1], 0, 255)),
                                            int(np.clip(color[2], 0, 255))))


def load_qwzx(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != _MAGIC:
            raise ValueError(f"{path} is not a valid .qwzx file")
        version = struct.unpack("B", f.read(1))[0]
        if version != _VERSION:
            raise ValueError(f"Unsupported .qwzx version {version}")

        num_meshes = struct.unpack("<I", f.read(4))[0]
        meshes = []
        for _ in range(num_meshes):
            name_len = struct.unpack("B", f.read(1))[0]
            name     = f.read(name_len).decode("utf-8")
            flags    = struct.unpack("B", f.read(1))[0]
            num_tris = struct.unpack("<I", f.read(4))[0]

            tris    = []
            colors  = []
            uvs     = []
            texture = None

            if flags in (1, 2, 3):
                w, h = struct.unpack("<II", f.read(8))
                channels = 4 if flags in (2, 3) else 3
                tex_raw = f.read(w * h * channels)
                texture = np.frombuffer(tex_raw, dtype=np.uint8).reshape(h, w, channels).copy()
                for _ in range(num_tris):
                    raw = struct.unpack("<fffffffff", f.read(36))
                    tri = [[raw[0], raw[1], raw[2]],
                           [raw[3], raw[4], raw[5]],
                           [raw[6], raw[7], raw[8]]]
                    uv_raw = struct.unpack("<ffffff", f.read(24))
                    uv = [[uv_raw[0], uv_raw[1]],
                          [uv_raw[2], uv_raw[3]],
                          [uv_raw[4], uv_raw[5]]]
                    tris.append(tri)
                    uvs.append(uv)
                    if flags == 3:
                        colors.append(list(struct.unpack("BBBB", f.read(4))))
            else:
                is_rgba = (flags == 4)
                for _ in range(num_tris):
                    raw = struct.unpack("<fffffffff", f.read(36))
                    tri = [[raw[0], raw[1], raw[2]],
                           [raw[3], raw[4], raw[5]],
                           [raw[6], raw[7], raw[8]]]
                    color = list(struct.unpack("BBBB" if is_rgba else "BBB", f.read(4 if is_rgba else 3)))
                    tris.append(tri)
                    colors.append(color)

            meshes.append({
                "name":    name,
                "tris":    tris,
                "colors":  colors if colors else None,
                "texture": texture,
                "uvs":     uvs if texture is not None else None,
            })
    return meshes


def _prepare_single_mesh(mesh):
    verts = np.array(mesh["tris"], dtype=np.float32) if len(mesh.get("tris", [])) > 0 else np.zeros((0, 3, 3), dtype=np.float32)
    uv_arr = np.array(mesh["uvs"], dtype=np.float32) if mesh.get("uvs") else None

    if len(verts) >= 4 and not mesh.get("_winding_checked"):
        flat = verts.reshape(-1, 3)
        extents = flat.max(axis=0) - flat.min(axis=0)
        if np.all(extents > 1e-3):
            center = flat.mean(axis=0)
            tri_centers = verts.mean(axis=1)
            normals = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
            outward = ((tri_centers - center) * normals).sum(axis=1)
            if (outward < 0).sum() > (outward > 0).sum():
                verts = verts[:, [0, 2, 1], :].copy()
                mesh["tris"] = verts.tolist()
                if uv_arr is not None:
                    uv_arr = uv_arr[:, [0, 2, 1], :].copy()
                    mesh["uvs"] = uv_arr.tolist()
                mesh.pop("_gpu_tri_slice", None)
        mesh["_winding_checked"] = True

    mesh["verts"]  = verts
    mesh["uv_arr"] = uv_arr
    if _USE_GPU_COMPUTE and (_gpu_ctx is not None or _init_gpu_compute(640, 640)):
        _ensure_gpu_mesh(mesh)
        if mesh.get("texture") is not None and mesh["uv_arr"] is not None:
            _ensure_gpu_texture(mesh["texture"])
    if "submeshes" not in mesh:
        mesh["submeshes"] = [mesh]
    return mesh



def prepare_mesh(mesh_or_list):
    if isinstance(mesh_or_list, (list, tuple)):
        return combine_submeshes(mesh_or_list)
    if isinstance(mesh_or_list, dict):
        subs = mesh_or_list.get("submeshes")
        if subs and not (len(subs) == 1 and subs[0] is mesh_or_list):
            for sm in subs:
                _prepare_single_mesh(sm)
            if "verts" not in mesh_or_list or mesh_or_list["verts"] is None:
                vert_blocks = [sm["verts"] for sm in subs if len(sm["verts"]) > 0]
                mesh_or_list["verts"] = np.concatenate(vert_blocks, axis=0) if vert_blocks else np.zeros((0, 3, 3), dtype=np.float32)
            if "tris" not in mesh_or_list or mesh_or_list["tris"] is None:
                mesh_or_list["tris"] = [tri for sm in subs for tri in sm["tris"]]
            return mesh_or_list
        return _prepare_single_mesh(mesh_or_list)
    raise TypeError(f"Unsupported mesh type: {type(mesh_or_list)}")


def build_vp_matrix(eye, target, up, fov_degrees, aspect_ratio, near, far):
    f = target - eye
    f = f / np.linalg.norm(f)

    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-8:
        fallback = np.array([0.0, 0.0, 1.0]) if abs(f[1]) > 0.99 else np.array([0.0, 1.0, 0.0])
        r = np.cross(f, fallback)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)

    mat_view = np.eye(4)
    mat_view[0, :3], mat_view[0, 3] = r,  -np.dot(r, eye)
    mat_view[1, :3], mat_view[1, 3] = u,  -np.dot(u, eye)
    mat_view[2, :3], mat_view[2, 3] = -f,  np.dot(f, eye)

    g = 1.0 / np.tan(np.radians(fov_degrees) / 2.0)
    mat_proj = np.zeros((4, 4))
    mat_proj[0, 0] = g / aspect_ratio
    mat_proj[1, 1] = g
    mat_proj[2, 2] = -(far + near) / (far - near)
    mat_proj[2, 3] = -(2.0 * far * near) / (far - near)
    mat_proj[3, 2] = -1.0

    return np.dot(mat_proj, mat_view)


def build_view_matrix(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)

    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-8:
        fallback = np.array([0.0, 0.0, 1.0]) if abs(f[1]) > 0.99 else np.array([0.0, 1.0, 0.0])
        r = np.cross(f, fallback)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)

    mat_view = np.eye(4)
    mat_view[0, :3], mat_view[0, 3] = r,  -np.dot(r, eye)
    mat_view[1, :3], mat_view[1, 3] = u,  -np.dot(u, eye)
    mat_view[2, :3], mat_view[2, 3] = -f,  np.dot(f, eye)
    return mat_view


def _clip_axis(verts, axis, sign):
    def inside(v):
        return sign * v[axis] <= v[3]

    def intersect(a, b):
        da = a[3] - sign * a[axis]
        db = b[3] - sign * b[axis]
        t  = da / (da - db)
        return a + t * (b - a)

    out = []
    n   = len(verts)
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        if inside(a):
            out.append(a)
        if inside(a) != inside(b):
            out.append(intersect(a, b))
    return out


def _clip_near_plane(verts, near):
    def inside(v):
        return v[3] > near

    def intersect(a, b):
        t = (a[3] - near) / (a[3] - b[3])
        return a + t * (b - a)

    out = []
    n   = len(verts)
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        if inside(a):
            out.append(a)
        if inside(a) != inside(b):
            out.append(intersect(a, b))
    return out


def clip_to_frustum(verts, near):
    verts = _clip_near_plane(verts, near)
    if len(verts) < 3:
        return []
    # Clip against far plane (z <= w)
    verts = _clip_axis(verts, 2, +1)
    if len(verts) < 3:
        return []
    for axis in (0, 1):
        verts = _clip_axis(verts, axis, +1)
        verts = _clip_axis(verts, axis, -1)
        if len(verts) < 3:
            return []
    return verts


def to_pixel(clip_vert, screen_width, screen_height):
    w = clip_vert[3]
    x = (clip_vert[0] / w + 1.0) * 0.5 * screen_width
    y = (1.0 - clip_vert[1] / w) * 0.5 * screen_height
    return (x, y)


def rotation_matrix(angles):
    yaw, pitch, roll = (np.radians(a % 360.0) for a in angles)

    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)

    R_yaw   = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_pitch = np.array([[1, 0, 0],   [0, cp, -sp], [0, sp, cp]])
    R_roll  = np.array([[cr, -sr, 0],[sr, cr,  0], [0,  0,  1]])

    return R_yaw @ R_pitch @ R_roll


def camera_pose_from_angles(camera_pos, camera_angles):
    yaw, pitch, roll = (np.radians(a % 360.0) for a in camera_angles)

    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)

    R_yaw   = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_pitch = np.array([[1, 0, 0],   [0, cp, -sp], [0, sp, cp]])
    R_roll  = np.array([[cr, -sr, 0],[sr, cr,  0], [0,  0,  1]])

    R       = R_yaw @ R_pitch @ R_roll
    forward = R @ np.array([0.0, 0.0, -1.0])
    up      = R @ np.array([0.0, 1.0,  0.0])

    eye    = np.array(camera_pos, dtype=float)
    target = eye + forward
    return eye, target, up


def _compute_cpu_shading(unit_normals, centroids, eye_np, far):
    mode = _SHADER_MODE
    N = len(unit_normals)
    if mode == "unlit":
        return np.ones((N, 3), dtype=np.float32), np.zeros((N, 3), dtype=np.float32), np.zeros((N, 1), dtype=np.float32)
    if mode == "normals":
        norm_rgb = np.clip(unit_normals * 0.5 + 0.5, 0.0, 1.0)
        return np.zeros((N, 3), dtype=np.float32), norm_rgb, np.zeros((N, 1), dtype=np.float32)

    to_eye = eye_np[None, :] - centroids
    dist = np.linalg.norm(to_eye, axis=1, keepdims=True)
    V = to_eye / np.maximum(dist, 1e-6)
    L = _LIGHT_DIR[None, :]

    ndotl = np.maximum((unit_normals * L).sum(axis=1, keepdims=True), 0.0)
    ndotv = np.maximum((unit_normals * V).sum(axis=1, keepdims=True), 0.0)
    H = L + V
    H /= np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-6)
    ndoth = np.maximum((unit_normals * H).sum(axis=1, keepdims=True), 0.0)

    if mode == "toon":
        bands = np.where(ndotl > 0.65, 1.0, np.where(ndotl > 0.25, 0.62, 0.32)).astype(np.float32)
        spec_step = np.where((ndoth ** 48.0) > 0.5, 0.28, 0.0).astype(np.float32) * (ndotl > 0.0)
        rim = ((1.0 - ndotv) ** 3.0) * 0.16
        mult = _LIGHT_COLOR[None, :] * bands + rim
        add = _LIGHT_COLOR[None, :] * spec_step
        fog = np.zeros((N, 1), dtype=np.float32)
        return mult, add, fog

    # Default "lit" Blinn-Phong + hemisphere ambient + distance fog
    hemi = unit_normals[:, 1:2] * 0.5 + 0.5
    ambient = _AMBIENT_GROUND[None, :] * (1.0 - hemi) + _AMBIENT_SKY[None, :] * hemi
    diffuse = _LIGHT_COLOR[None, :] * (ndotl * 0.72)
    fill = ndotv * 0.12
    spec = (ndoth ** 32.0) * np.where(ndotl > 0.0, 0.22, 0.0)
    mult = ambient + diffuse + fill
    add = _LIGHT_COLOR[None, :] * spec
    fog = np.clip((dist - far * 0.45) / max(far * 0.55, 1e-4), 0.0, 1.0) ** 2 * 0.65
    return mult.astype(np.float32), add.astype(np.float32), fog.astype(np.float32)


def draw_textured_triangle(screen_arr, zbuf, texture, screen_pts, uvs, ws, zs,
                           shade_mult=None, shade_add=None, fog=0.0, tint_rgba=None):
    th, tw = texture.shape[:2]
    has_tex_alpha = (texture.ndim == 3 and texture.shape[2] == 4)
    sw, sh = screen_arr.shape[0], screen_arr.shape[1]
    p  = np.array(screen_pts, dtype=np.float32)
    uv = np.array(uvs,        dtype=np.float32)
    w  = np.array(ws,         dtype=np.float32)
    z  = np.array(zs,         dtype=np.float32)

    x0 = max(0,    int(np.floor(p[:, 0].min())))
    x1 = min(sw-1, int(np.ceil (p[:, 0].max())))
    y0 = max(0,    int(np.floor(p[:, 1].min())))
    y1 = min(sh-1, int(np.ceil (p[:, 1].max())))
    if x0 > x1 or y0 > y1:
        return

    xs, ys = np.meshgrid(np.arange(x0, x1+1, dtype=np.float32),
                         np.arange(y0, y1+1, dtype=np.float32))
    px = xs.ravel()
    py = ys.ravel()

    denom = (p[1,1]-p[2,1])*(p[0,0]-p[2,0]) + (p[2,0]-p[1,0])*(p[0,1]-p[2,1])
    if abs(denom) < 1e-10:
        return
    b0 = ((p[1,1]-p[2,1])*(px-p[2,0]) + (p[2,0]-p[1,0])*(py-p[2,1])) / denom
    b1 = ((p[2,1]-p[0,1])*(px-p[2,0]) + (p[0,0]-p[2,0])*(py-p[2,1])) / denom
    b2 = 1.0 - b0 - b1

    inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
    if not inside.any():
        return
    px = px[inside].astype(np.int32)
    py = py[inside].astype(np.int32)
    b0, b1, b2 = b0[inside], b1[inside], b2[inside]

    depth   = b0*z[0] + b1*z[1] + b2*z[2]
    visible = depth < zbuf[px, py]
    if not visible.any():
        return
    px, py  = px[visible], py[visible]
    b0, b1, b2 = b0[visible], b1[visible], b2[visible]
    depth_vis = depth[visible]

    inv_w        = 1.0 / np.maximum(w, 1e-6)
    interp_inv_w = b0*inv_w[0] + b1*inv_w[1] + b2*inv_w[2]
    u = (b0*uv[0,0]*inv_w[0] + b1*uv[1,0]*inv_w[1] + b2*uv[2,0]*inv_w[2]) / interp_inv_w
    v = (b0*uv[0,1]*inv_w[0] + b1*uv[1,1]*inv_w[1] + b2*uv[2,1]*inv_w[2]) / interp_inv_w

    u_wrap = np.where((u >= 0.0) & (u <= 1.0), u, np.mod(u, 1.0))
    v_wrap = np.where((v >= 0.0) & (v <= 1.0), v, np.mod(v, 1.0))

    tx = np.clip((u_wrap * tw).astype(np.int32), 0, tw-1)
    ty = np.clip((v_wrap * th).astype(np.int32), 0, th-1)

    sampled = texture[ty, tx]
    rgb = sampled[:, 2::-1].astype(np.float32)

    base_alpha = 1.0
    if tint_rgba is not None:
        rgb *= (np.asarray(tint_rgba[:3], dtype=np.float32) / 255.0)
        if len(tint_rgba) >= 4:
            base_alpha = float(tint_rgba[3]) / 255.0

    if shade_mult is not None:
        rgb = rgb * shade_mult + (shade_add * 255.0 if shade_add is not None else 0.0)
    if fog > 0.0:
        fog_col = np.array([20.4, 25.5, 35.7], dtype=np.float32)
        rgb = rgb * (1.0 - fog) + fog_col * fog
    rgb = np.clip(rgb, 0.0, 255.0)

    if has_tex_alpha or base_alpha < 0.995:
        if has_tex_alpha:
            alpha = (sampled[:, 3:4].astype(np.float32) / 255.0) * base_alpha
        else:
            alpha = np.full((len(px), 1), base_alpha, dtype=np.float32)
        keep = alpha[:, 0] >= 0.01
        if not keep.any():
            return
        px, py = px[keep], py[keep]
        rgb = rgb[keep]
        alpha = alpha[keep]
        depth_vis = depth_vis[keep]

        opaque_mask = alpha[:, 0] >= 0.995
        if opaque_mask.any():
            opx, opy = px[opaque_mask], py[opaque_mask]
            zbuf[opx, opy] = depth_vis[opaque_mask]
            screen_arr[opx, opy] = rgb[opaque_mask].astype(np.uint8)

        trans_mask = ~opaque_mask
        if trans_mask.any():
            tpx, tpy = px[trans_mask], py[trans_mask]
            ta = alpha[trans_mask]
            dst = screen_arr[tpx, tpy].astype(np.float32)
            screen_arr[tpx, tpy] = (rgb[trans_mask] * ta + dst * (1.0 - ta)).astype(np.uint8)
    else:
        zbuf[px, py] = depth_vis
        screen_arr[px, py] = rgb.astype(np.uint8)


def draw_flat_triangle(screen_arr, zbuf, screen_pts, zs, color):
    sw, sh = screen_arr.shape[0], screen_arr.shape[1]
    p = np.array(screen_pts, dtype=np.float32)
    z = np.array(zs,         dtype=np.float32)

    c_len = len(color) if color is not None else 0
    alpha = (float(color[3]) / 255.0) if c_len >= 4 else 1.0
    if alpha < 0.01:
        return

    x0 = max(0,    int(np.floor(p[:, 0].min())))
    x1 = min(sw-1, int(np.ceil (p[:, 0].max())))
    y0 = max(0,    int(np.floor(p[:, 1].min())))
    y1 = min(sh-1, int(np.ceil (p[:, 1].max())))
    if x0 > x1 or y0 > y1:
        return

    xs, ys = np.meshgrid(np.arange(x0, x1+1, dtype=np.float32),
                         np.arange(y0, y1+1, dtype=np.float32))
    px = xs.ravel()
    py = ys.ravel()

    denom = (p[1,1]-p[2,1])*(p[0,0]-p[2,0]) + (p[2,0]-p[1,0])*(p[0,1]-p[2,1])
    if abs(denom) < 1e-10:
        return
    b0 = ((p[1,1]-p[2,1])*(px-p[2,0]) + (p[2,0]-p[1,0])*(py-p[2,1])) / denom
    b1 = ((p[2,1]-p[0,1])*(px-p[2,0]) + (p[0,0]-p[2,0])*(py-p[2,1])) / denom
    b2 = 1.0 - b0 - b1

    inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
    if not inside.any():
        return
    px = px[inside].astype(np.int32)
    py = py[inside].astype(np.int32)
    b0, b1, b2 = b0[inside], b1[inside], b2[inside]

    depth   = b0*z[0] + b1*z[1] + b2*z[2]
    visible = depth < zbuf[px, py]
    if not visible.any():
        return
    px, py = px[visible], py[visible]

    rgb_u8 = np.clip(np.asarray(color[:3], dtype=np.float32), 0.0, 255.0).astype(np.uint8)
    if alpha >= 0.995:
        zbuf[px, py] = depth[visible]
        screen_arr[px, py] = rgb_u8
    else:
        src = rgb_u8.astype(np.float32)
        dst = screen_arr[px, py].astype(np.float32)
        screen_arr[px, py] = (src * alpha + dst * (1.0 - alpha)).astype(np.uint8)



# Stage 1 Compute Shader: 3D-to-2D Vertex Transform, Backface Culling,
# Near/Far-Plane Clipping, World-Space Normal/Position Interpolation, and Projection
_GPU_GEOM_SHADER_SRC = """#version 430
layout(local_size_x = 64) in;

struct RawTri {
    vec4 v0;    // v0.xyz, uv0.x
    vec4 v1;    // v1.xyz, uv0.y
    vec4 v2;    // v2.xyz, uv1.x
    vec4 uvs;   // uv1.y, uv2.x, uv2.y, 0.0
    vec4 color; // r, g, b, a
};

struct InstanceData {
    vec4 r0;    // R[0,0], R[0,1], R[0,2], pos.x
    vec4 r1;    // R[1,0], R[1,1], R[1,2], pos.y
    vec4 r2;    // R[2,0], R[2,1], R[2,2], pos.z
    ivec4 meta; // raw_tri_start, raw_tri_count, tex_id, cum_tri_start
};

struct ScreenTri {
    vec4 p0;    // p0_x, p0_y, ndc_z0, inv_w0
    vec4 p1;    // p1_x, p1_y, ndc_z1, inv_w1
    vec4 p2;    // p2_x, p2_y, ndc_z2, inv_w2
    vec4 uv0;   // u0, v0, w0.x, w0.y
    vec4 uv1;   // u1, v1, w1.x, w1.y
    vec4 uv2;   // u2, v2, w2.x, w2.y
    vec4 bb;    // min_x, min_y, max_x, max_y
    vec4 color; // r, g, b, a
    vec4 norm;  // normal.x, normal.y, normal.z, 0.0
    vec4 wz;    // w0.z, w1.z, w2.z, 0.0
    int tex_id;
    int pad0, pad1, pad2;
};

layout(std430, binding = 0) writeonly buffer TriBuffer {
    ScreenTri triangles[];
};

layout(std430, binding = 1) readonly buffer RawTriBuffer {
    RawTri raw_tris[];
};

layout(std430, binding = 2) readonly buffer InstanceBuffer {
    InstanceData instances[];
};

layout(std430, binding = 3) coherent buffer CounterBuffer {
    uint screen_tri_count;
};

uniform vec4 u_vp_r0;
uniform vec4 u_vp_r1;
uniform vec4 u_vp_r2;
uniform vec4 u_vp_r3;
uniform vec3 u_eye;
uniform float u_near;
uniform int u_width;
uniform int u_height;
uniform int u_total_input_tris;
uniform int u_num_instances;
uniform int u_max_screen_tris;

vec3 transform_world(InstanceData inst, vec3 v) {
    return vec3(
        dot(inst.r0.xyz, v) + inst.r0.w,
        dot(inst.r1.xyz, v) + inst.r1.w,
        dot(inst.r2.xyz, v) + inst.r2.w
    );
}

vec4 transform_clip(vec3 w) {
    vec4 hom = vec4(w, 1.0);
    return vec4(
        dot(u_vp_r0, hom),
        dot(u_vp_r1, hom),
        dot(u_vp_r2, hom),
        dot(u_vp_r3, hom)
    );
}

void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= u_total_input_tris || u_num_instances <= 0) return;

    int lo = 0;
    int hi = u_num_instances - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (instances[mid].meta.w <= gid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    InstanceData inst = instances[lo];
    int local_idx = gid - inst.meta.w;
    if (local_idx < 0 || local_idx >= inst.meta.y) return;

    RawTri raw = raw_tris[inst.meta.x + local_idx];
    if (raw.color.a < 0.01) return;

    vec3 w0 = transform_world(inst, raw.v0.xyz);
    vec3 w1 = transform_world(inst, raw.v1.xyz);
    vec3 w2 = transform_world(inst, raw.v2.xyz);

    vec3 raw_norm = cross(w1 - w0, w2 - w0);
    float norm_len = length(raw_norm);
    if (norm_len < 1e-10) return;
    if (dot(raw_norm, u_eye - w0) <= 0.0) return;
    vec3 unit_norm = raw_norm / norm_len;

    vec4 in_c[3];
    in_c[0] = transform_clip(w0);
    in_c[1] = transform_clip(w1);
    in_c[2] = transform_clip(w2);

    vec2 in_uv[3];
    in_uv[0] = vec2(raw.v0.w, raw.v1.w);
    in_uv[1] = vec2(raw.v2.w, raw.uvs.x);
    in_uv[2] = vec2(raw.uvs.y, raw.uvs.z);

    vec3 in_w[3];
    in_w[0] = w0;
    in_w[1] = w1;
    in_w[2] = w2;

    // 1. Clip against near plane (w > u_near)
    vec4 clip1_c[4];
    vec2 clip1_uv[4];
    vec3 clip1_w[4];
    int n1 = 0;

    for (int i = 0; i < 3; ++i) {
        int next = (i + 1) % 3;
        vec4 a_c = in_c[i];
        vec2 a_uv = in_uv[i];
        vec3 a_w = in_w[i];
        vec4 b_c = in_c[next];
        vec2 b_uv = in_uv[next];
        vec3 b_w = in_w[next];

        bool a_in = a_c.w > u_near;
        bool b_in = b_c.w > u_near;

        if (a_in) {
            clip1_c[n1] = a_c;
            clip1_uv[n1] = a_uv;
            clip1_w[n1] = a_w;
            n1++;
        }
        if (a_in != b_in) {
            float t = (a_c.w - u_near) / (a_c.w - b_c.w);
            clip1_c[n1] = mix(a_c, b_c, t);
            clip1_uv[n1] = mix(a_uv, b_uv, t);
            clip1_w[n1] = mix(a_w, b_w, t);
            n1++;
        }
    }

    if (n1 < 3) return;

    // 2. Clip against far plane (z <= w)
    vec4 clip_c[5];
    vec2 clip_uv[5];
    vec3 clip_w[5];
    int num_clipped = 0;

    for (int i = 0; i < n1; ++i) {
        int next = (i + 1) % n1;
        vec4 a_c = clip1_c[i];
        vec2 a_uv = clip1_uv[i];
        vec3 a_w = clip1_w[i];
        vec4 b_c = clip1_c[next];
        vec2 b_uv = clip1_uv[next];
        vec3 b_w = clip1_w[next];

        float da = a_c.w - a_c.z;
        float db = b_c.w - b_c.z;
        bool a_in = da >= 0.0;
        bool b_in = db >= 0.0;

        if (a_in) {
            clip_c[num_clipped] = a_c;
            clip_uv[num_clipped] = a_uv;
            clip_w[num_clipped] = a_w;
            num_clipped++;
        }
        if (a_in != b_in) {
            float t = da / (da - db);
            clip_c[num_clipped] = mix(a_c, b_c, t);
            clip_uv[num_clipped] = mix(a_uv, b_uv, t);
            clip_w[num_clipped] = mix(a_w, b_w, t);
            num_clipped++;
        }
    }

    if (num_clipped < 3) return;

    float sw = float(u_width);
    float sh = float(u_height);
    int tex_id = inst.meta.z;

    for (int k = 1; k < num_clipped - 1; ++k) {
        vec4 pa = clip_c[0];
        vec4 pb = clip_c[k];
        vec4 pc = clip_c[k + 1];

        vec2 uva = clip_uv[0];
        vec2 uvb = clip_uv[k];
        vec2 uvc = clip_uv[k + 1];

        vec3 wa = clip_w[0];
        vec3 wb = clip_w[k];
        vec3 wc = clip_w[k + 1];

        float inv_wa = 1.0 / max(pa.w, 1e-6);
        float inv_wb = 1.0 / max(pb.w, 1e-6);
        float inv_wc = 1.0 / max(pc.w, 1e-6);

        float p0_x = (pa.x * inv_wa + 1.0) * 0.5 * sw;
        float p0_y = (1.0 - pa.y * inv_wa) * 0.5 * sh;
        float p1_x = (pb.x * inv_wb + 1.0) * 0.5 * sw;
        float p1_y = (1.0 - pb.y * inv_wb) * 0.5 * sh;
        float p2_x = (pc.x * inv_wc + 1.0) * 0.5 * sw;
        float p2_y = (1.0 - pc.y * inv_wc) * 0.5 * sh;

        float min_x = max(0.0, min(p0_x, min(p1_x, p2_x)));
        float max_x = min(sw - 1.0, max(p0_x, max(p1_x, p2_x)));
        float min_y = max(0.0, min(p0_y, min(p1_y, p2_y)));
        float max_y = min(sh - 1.0, max(p0_y, max(p1_y, p2_y)));
        if (min_x > max_x || min_y > max_y) continue;

        float denom = (p1_y - p2_y) * (p0_x - p2_x) + (p2_x - p1_x) * (p0_y - p2_y);
        if (abs(denom) < 1e-10) continue;

        uint out_idx = atomicAdd(screen_tri_count, 1u);
        if (out_idx >= uint(u_max_screen_tris)) continue;

        ScreenTri st;
        st.p0 = vec4(p0_x, p0_y, pa.z * inv_wa, inv_wa);
        st.p1 = vec4(p1_x, p1_y, pb.z * inv_wb, inv_wb);
        st.p2 = vec4(p2_x, p2_y, pc.z * inv_wc, inv_wc);
        st.uv0 = vec4(uva, wa.x, wa.y);
        st.uv1 = vec4(uvb, wb.x, wb.y);
        st.uv2 = vec4(uvc, wc.x, wc.y);
        st.bb = vec4(min_x, min_y, max_x, max_y);
        st.color = raw.color;
        st.norm = vec4(unit_norm, 0.0);
        st.wz = vec4(wa.z, wb.z, wc.z, 0.0);
        st.tex_id = tex_id;
        st.pad0 = 0;
        st.pad1 = 0;
        st.pad2 = 0;
        triangles[out_idx] = st;
    }
}
"""

# Stage 2 Compute Shader: Per-Pixel Rasterization, Depth Buffering,
# Perspective-Correct UV & World-Pos Interpolation, Lighting Shaders, and Order-Independent Alpha Blending
_GPU_COMPUTE_SHADER_SRC = """#version 430
layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba8, binding = 0) uniform writeonly image2D u_color_img;

layout(binding = 0) uniform sampler2D u_tex0;
layout(binding = 1) uniform sampler2D u_tex1;
layout(binding = 2) uniform sampler2D u_tex2;
layout(binding = 3) uniform sampler2D u_tex3;
layout(binding = 4) uniform sampler2D u_tex4;
layout(binding = 5) uniform sampler2D u_tex5;
layout(binding = 6) uniform sampler2D u_tex6;
layout(binding = 7) uniform sampler2D u_tex7;

struct ScreenTri {
    vec4 p0;
    vec4 p1;
    vec4 p2;
    vec4 uv0;
    vec4 uv1;
    vec4 uv2;
    vec4 bb;
    vec4 color;
    vec4 norm;
    vec4 wz;
    int tex_id;
    int pad0, pad1, pad2;
};

layout(std430, binding = 0) readonly buffer TriBuffer {
    ScreenTri triangles[];
};

layout(std430, binding = 3) readonly buffer CounterBuffer {
    uint screen_tri_count;
};

layout(std430, binding = 4) readonly buffer TexDataBuffer {
    uint tex_pixels[];
};

layout(std430, binding = 5) readonly buffer TexMetaBuffer {
    ivec4 tex_meta[]; // offset, width, height, 0
};

uniform int u_width;
uniform int u_height;
uniform int u_max_screen_tris;
uniform int u_shader_mode; // 0 = lit, 1 = toon, 2 = normals, 3 = unlit
uniform vec3 u_eye;
uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform vec3 u_ambient_sky;
uniform vec3 u_ambient_ground;
uniform float u_far;

vec2 wrap_uv(vec2 uv) {
    float u = (uv.x >= 0.0 && uv.x <= 1.0) ? uv.x : fract(uv.x);
    float v = (uv.y >= 0.0 && uv.y <= 1.0) ? uv.y : fract(uv.y);
    return vec2(u, v);
}

vec4 sample_ssbo_tex(int id, vec2 wrapped_uv) {
    ivec4 meta = tex_meta[id];
    int offset = meta.x;
    int tw = meta.y;
    int th = meta.z;
    if (tw <= 0 || th <= 0) return vec4(1.0);

    float fx = wrapped_uv.x * float(tw - 1);
    float fy = wrapped_uv.y * float(th - 1);
    int x0 = clamp(int(floor(fx)), 0, tw - 1);
    int y0 = clamp(int(floor(fy)), 0, th - 1);
    int x1 = min(x0 + 1, tw - 1);
    int y1 = min(y0 + 1, th - 1);
    float dx = fx - float(x0);
    float dy = fy - float(y0);

    vec4 c00 = unpackUnorm4x8(tex_pixels[offset + y0 * tw + x0]);
    vec4 c10 = unpackUnorm4x8(tex_pixels[offset + y0 * tw + x1]);
    vec4 c01 = unpackUnorm4x8(tex_pixels[offset + y1 * tw + x0]);
    vec4 c11 = unpackUnorm4x8(tex_pixels[offset + y1 * tw + x1]);
    return mix(mix(c00, c10, dx), mix(c01, c11, dx), dy);
}

vec4 sample_mesh_tex(int id, vec2 uv) {
    vec2 w_uv = wrap_uv(uv);
    if (id == 0) return textureLod(u_tex0, w_uv, 0.0);
    if (id == 1) return textureLod(u_tex1, w_uv, 0.0);
    if (id == 2) return textureLod(u_tex2, w_uv, 0.0);
    if (id == 3) return textureLod(u_tex3, w_uv, 0.0);
    if (id == 4) return textureLod(u_tex4, w_uv, 0.0);
    if (id == 5) return textureLod(u_tex5, w_uv, 0.0);
    if (id == 6) return textureLod(u_tex6, w_uv, 0.0);
    if (id == 7) return textureLod(u_tex7, w_uv, 0.0);
    if (id >= 8) return sample_ssbo_tex(id, w_uv);
    return vec4(1.0);
}

vec4 shade_fragment(vec4 base_col, vec3 N, vec3 world_pos) {
    if (u_shader_mode == 3) {
        return base_col;
    }
    vec3 n = normalize(N);
    if (u_shader_mode == 2) {
        return vec4(n * 0.5 + 0.5, base_col.a);
    }

    vec3 to_eye = u_eye - world_pos;
    float dist = length(to_eye);
    vec3 V = dist > 1e-5 ? (to_eye / dist) : vec3(0.0, 0.0, 1.0);
    vec3 L = normalize(u_light_dir);
    vec3 H = normalize(L + V);

    float ndotl = max(dot(n, L), 0.0);
    float ndotv = max(dot(n, V), 0.0);
    float ndoth = max(dot(n, H), 0.0);

    if (u_shader_mode == 1) {
        float band = ndotl > 0.65 ? 1.0 : (ndotl > 0.25 ? 0.62 : 0.32);
        float spec_step = (ndotl > 0.0 && pow(ndoth, 48.0) > 0.5) ? 0.28 : 0.0;
        float rim = pow(1.0 - ndotv, 3.0) * 0.16;
        vec3 cel = base_col.rgb * (u_light_color * band + rim) + u_light_color * spec_step;
        return vec4(clamp(cel, 0.0, 1.0), base_col.a);
    }

    // Mode 0: Blinn-Phong + Hemisphere Ambient + Distance Fog
    float hemi = n.y * 0.5 + 0.5;
    vec3 ambient = mix(u_ambient_ground, u_ambient_sky, hemi);
    vec3 diffuse = u_light_color * (ndotl * 0.72);
    float fill = ndotv * 0.12;
    float spec = pow(ndoth, 32.0) * (ndotl > 0.0 ? 0.22 : 0.0);

    vec3 lit = base_col.rgb * (ambient + diffuse + fill) + u_light_color * spec;
    float fog = clamp((dist - u_far * 0.45) / max(u_far * 0.55, 1e-4), 0.0, 1.0);
    fog = fog * fog * 0.65;
    vec3 fog_col = vec3(0.08, 0.10, 0.14);
    lit = mix(lit, fog_col, fog);
    return vec4(clamp(lit, 0.0, 1.0), base_col.a);
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= u_width || pixel.y >= u_height) return;

    vec2 p = vec2(pixel) + 0.5;
    float best_depth = 1e30;
    vec4 best_base_col = vec4(0.0, 0.0, 0.0, 1.0);
    vec3 best_norm = vec3(0.0, 1.0, 0.0);
    vec3 best_world_pos = vec3(0.0);
    bool hit_opaque = false;

    // 4-layer k-buffer for order-independent translucent surfaces (sorted closest to furthest)
    float trans_depth[4];
    vec4 trans_col[4];
    int num_trans = 0;

    int num_tris = min(int(screen_tri_count), u_max_screen_tris);

    for (int i = 0; i < num_tris; ++i) {
        ScreenTri tri = triangles[i];

        if (p.x < tri.bb.x || p.x > tri.bb.z || p.y < tri.bb.y || p.y > tri.bb.w) continue;

        vec2 p0 = tri.p0.xy;
        vec2 p1 = tri.p1.xy;
        vec2 p2 = tri.p2.xy;

        float denom = (p1.y - p2.y) * (p0.x - p2.x) + (p2.x - p1.x) * (p0.y - p2.y);
        float inv_denom = 1.0 / denom;

        float b0 = ((p1.y - p2.y) * (p.x - p2.x) + (p2.x - p1.x) * (p.y - p2.y)) * inv_denom;
        float b1 = ((p2.y - p0.y) * (p.x - p2.x) + (p0.x - p2.x) * (p.y - p2.y)) * inv_denom;
        float b2 = 1.0 - b0 - b1;

        if (b0 >= 0.0 && b1 >= 0.0 && b2 >= 0.0) {
            float depth = b0 * tri.p0.z + b1 * tri.p1.z + b2 * tri.p2.z;
            if (depth >= best_depth) continue;

            float w0 = b0 * tri.p0.w;
            float w1 = b1 * tri.p1.w;
            float w2 = b2 * tri.p2.w;
            float inv_interp_w = 1.0 / max(w0 + w1 + w2, 1e-12);

            vec4 base_col;
            if (tri.tex_id >= 0) {
                float u = (w0 * tri.uv0.x + w1 * tri.uv1.x + w2 * tri.uv2.x) * inv_interp_w;
                float v = (w0 * tri.uv0.y + w1 * tri.uv1.y + w2 * tri.uv2.y) * inv_interp_w;
                vec4 tex_col = sample_mesh_tex(tri.tex_id, vec2(u, v));
                base_col = vec4(tex_col.rgb * tri.color.rgb, tex_col.a * tri.color.a);
            } else {
                base_col = tri.color;
            }

            if (base_col.a < 0.01) continue;

            vec3 world_pos = vec3(
                (w0 * tri.uv0.z + w1 * tri.uv1.z + w2 * tri.uv2.z) * inv_interp_w,
                (w0 * tri.uv0.w + w1 * tri.uv1.w + w2 * tri.uv2.w) * inv_interp_w,
                (w0 * tri.wz.x  + w1 * tri.wz.y  + w2 * tri.wz.z)  * inv_interp_w
            );

            if (base_col.a >= 0.995) {
                best_depth = depth;
                best_base_col = base_col;
                best_norm = tri.norm.xyz;
                best_world_pos = world_pos;
                hit_opaque = true;
            } else {
                vec4 lit_trans = shade_fragment(base_col, tri.norm.xyz, world_pos);
                int ins = num_trans;
                for (int k = 0; k < num_trans; ++k) {
                    if (depth < trans_depth[k]) {
                        ins = k;
                        break;
                    }
                }
                if (ins < 4) {
                    int cap = min(num_trans, 3);
                    for (int k = cap; k > ins; --k) {
                        trans_depth[k] = trans_depth[k - 1];
                        trans_col[k] = trans_col[k - 1];
                    }
                    trans_depth[ins] = depth;
                    trans_col[ins] = lit_trans;
                    if (num_trans < 4) num_trans++;
                }
            }
        }
    }

    vec4 final_color;
    if (hit_opaque) {
        final_color = shade_fragment(best_base_col, best_norm, best_world_pos);
    } else {
        final_color = vec4(0.0, 0.0, 0.0, 1.0);
    }

    // Composite translucent hits in front of best_depth from back to front
    for (int k = num_trans - 1; k >= 0; --k) {
        if (trans_depth[k] < best_depth) {
            vec4 tc = trans_col[k];
            final_color.rgb = mix(final_color.rgb, tc.rgb, tc.a);
        }
    }

    imageStore(u_color_img, pixel, vec4(final_color.rgb, 1.0));
}
"""

_GPU_DTYPE = np.dtype([
    ('p0', 'f4', 4), ('p1', 'f4', 4), ('p2', 'f4', 4),
    ('uv0', 'f4', 4), ('uv1', 'f4', 4), ('uv2', 'f4', 4),
    ('bb', 'f4', 4), ('color', 'f4', 4),
    ('norm', 'f4', 4), ('wz', 'f4', 4),
    ('tex_id', 'i4'), ('pad0', 'i4'), ('pad1', 'i4'), ('pad2', 'i4')
])

_GPU_RAW_TRI_DTYPE = np.dtype([
    ('v0', 'f4', 4),
    ('v1', 'f4', 4),
    ('v2', 'f4', 4),
    ('uvs', 'f4', 4),
    ('color', 'f4', 4),
])

_GPU_INSTANCE_DTYPE = np.dtype([
    ('r0', 'f4', 4),
    ('r1', 'f4', 4),
    ('r2', 'f4', 4),
    ('meta', 'i4', 4),
])


def _init_gpu_compute(width, height):
    global _gpu_ctx, _gpu_geom_shader, _gpu_compute_shader, _gpu_color_tex
    global _gpu_readback_buf, _gpu_counter_ssbo, _gpu_tex_data_ssbo, _gpu_tex_meta_ssbo, _gpu_w, _gpu_h
    if _gpu_ctx is None:
        try:
            _gpu_ctx = moderngl.create_context(standalone=True, require=430)
        except Exception as e:
            print(f"[Renderer] ModernGL GPU compute init failed: {e}. Falling back to CPU rasterizer.")
            return False

    if _gpu_w != width or _gpu_h != height:
        _gpu_w, _gpu_h = width, height
        _gpu_color_tex = _gpu_ctx.texture((width, height), 4, dtype='f1')
        _gpu_color_tex.bind_to_image(0, read=False, write=True)
        _gpu_readback_buf = bytearray(width * height * 4)

    if _gpu_geom_shader is None:
        try:
            _gpu_geom_shader = _gpu_ctx.compute_shader(_GPU_GEOM_SHADER_SRC)
        except Exception as e:
            print(f"[Renderer] Geometry compute shader compilation failed: {e}. Falling back to CPU rasterizer.")
            return False

    if _gpu_compute_shader is None:
        try:
            _gpu_compute_shader = _gpu_ctx.compute_shader(_GPU_COMPUTE_SHADER_SRC)
        except Exception as e:
            print(f"[Renderer] Raster compute shader compilation failed: {e}. Falling back to CPU rasterizer.")
            return False

    if _gpu_counter_ssbo is None:
        _gpu_counter_ssbo = _gpu_ctx.buffer(reserve=16)

    if _gpu_tex_data_ssbo is None:
        _gpu_tex_data_ssbo = _gpu_ctx.buffer(reserve=16)

    if _gpu_tex_meta_ssbo is None:
        _gpu_tex_meta_ssbo = _gpu_ctx.buffer(reserve=16)

    return True


def _ensure_gpu_mesh(mesh):
    global _gpu_raw_tri_ssbo, _gpu_raw_tri_cap, _gpu_raw_tri_chunks, _gpu_total_cached_tris
    mesh_alpha = float(mesh.get("alpha", 1.0))
    cached = mesh.get("_gpu_tri_slice")
    if cached is not None and cached[0] is _gpu_ctx and len(cached) >= 4 and abs(cached[3] - mesh_alpha) < 1e-6:
        return cached[1], cached[2]

    verts = mesh.get("verts")
    if verts is None:
        verts = np.array(mesh["tris"], dtype=np.float32) if len(mesh.get("tris", [])) > 0 else np.zeros((0, 3, 3), dtype=np.float32)
        mesh["verts"] = verts
    uv_arr = mesh.get("uv_arr")
    if uv_arr is None and mesh.get("uvs") is not None:
        uv_arr = np.array(mesh["uvs"], dtype=np.float32)
        mesh["uv_arr"] = uv_arr

    colors = mesh.get("colors")
    has_tex = mesh.get("texture") is not None
    N = int(verts.shape[0])
    start_tri = _gpu_total_cached_tris

    if N > 0:
        chunk = np.zeros(N, dtype=_GPU_RAW_TRI_DTYPE)
        chunk['v0'][:, :3] = verts[:, 0, :]
        chunk['v1'][:, :3] = verts[:, 1, :]
        chunk['v2'][:, :3] = verts[:, 2, :]

        if uv_arr is not None and len(uv_arr) == N:
            chunk['v0'][:, 3] = uv_arr[:, 0, 0]
            chunk['v1'][:, 3] = uv_arr[:, 0, 1]
            chunk['v2'][:, 3] = uv_arr[:, 1, 0]
            chunk['uvs'][:, 0] = uv_arr[:, 1, 1]
            chunk['uvs'][:, 1] = uv_arr[:, 2, 0]
            chunk['uvs'][:, 2] = uv_arr[:, 2, 1]

        if colors is not None and len(colors) == N:
            col_np = np.asarray(colors, dtype=np.float32) / 255.0
            chunk['color'][:, :3] = col_np[:, :3]
            if col_np.shape[1] >= 4:
                chunk['color'][:, 3] = np.clip(col_np[:, 3] * mesh_alpha, 0.0, 1.0)
            else:
                chunk['color'][:, 3] = np.clip(mesh_alpha, 0.0, 1.0)
        else:
            chunk['color'][:, :3] = 1.0 if has_tex else (200.0 / 255.0)
            chunk['color'][:, 3] = np.clip(mesh_alpha, 0.0, 1.0)

        _gpu_raw_tri_chunks.append(chunk)
        _gpu_total_cached_tris += N

        all_raw_bytes = np.concatenate(_gpu_raw_tri_chunks).tobytes()
        if _gpu_raw_tri_ssbo is None or len(all_raw_bytes) > _gpu_raw_tri_cap:
            if _gpu_raw_tri_ssbo is not None:
                _gpu_raw_tri_ssbo.release()
            _gpu_raw_tri_ssbo = _gpu_ctx.buffer(reserve=max(len(all_raw_bytes) * 2, 65536))
            _gpu_raw_tri_cap = _gpu_raw_tri_ssbo.size
        _gpu_raw_tri_ssbo.write(all_raw_bytes)

    mesh["_gpu_tri_slice"] = (_gpu_ctx, start_tri, N, mesh_alpha)
    return start_tri, N


def _ensure_gpu_texture(texture):
    global _gpu_ctx, _gpu_textures, _gpu_tex_slot_cache, _gpu_tex_refs, _gpu_tex_meta_list
    global _gpu_tex_data_chunks, _gpu_total_texels, _gpu_tex_data_ssbo, _gpu_tex_meta_ssbo
    if _gpu_ctx is None or texture is None:
        return -1

    tex_id = id(texture)
    if tex_id in _gpu_tex_slot_cache:
        return _gpu_tex_slot_cache[tex_id]

    _gpu_tex_refs.append(texture)
    slot = len(_gpu_tex_slot_cache)
    th, tw = texture.shape[:2]
    channels = texture.shape[2] if texture.ndim == 3 else 1

    rgba = np.empty((th, tw, 4), dtype=np.uint8)
    if channels == 4:
        # Convert BGRA (OpenCV) to RGBA
        rgba[:, :, 0] = texture[:, :, 2]
        rgba[:, :, 1] = texture[:, :, 1]
        rgba[:, :, 2] = texture[:, :, 0]
        rgba[:, :, 3] = texture[:, :, 3]
    else:
        rgba[:, :, :3] = texture[:, :, 2::-1]
        rgba[:, :, 3] = 255

    if slot < 8:
        gl_tex = _gpu_ctx.texture((tw, th), 4, rgba.tobytes())
        gl_tex.repeat_x = True
        gl_tex.repeat_y = True
        gl_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        gl_tex.build_mipmaps()
        _gpu_textures[slot] = gl_tex

    # Also upload packed RGBA8 texels to GPU SSBO so >8 textures work seamlessly
    packed_u32 = rgba.view(np.uint32).ravel().copy()

    offset = _gpu_total_texels
    _gpu_tex_data_chunks.append(packed_u32)
    _gpu_total_texels += packed_u32.size
    _gpu_tex_meta_list.append([offset, tw, th, 0])

    all_tex_bytes = np.concatenate(_gpu_tex_data_chunks).tobytes()
    if _gpu_tex_data_ssbo is not None:
        _gpu_tex_data_ssbo.release()
    _gpu_tex_data_ssbo = _gpu_ctx.buffer(all_tex_bytes)

    meta_bytes = np.array(_gpu_tex_meta_list, dtype=np.int32).tobytes()
    if _gpu_tex_meta_ssbo is not None:
        _gpu_tex_meta_ssbo.release()
    _gpu_tex_meta_ssbo = _gpu_ctx.buffer(meta_bytes)

    _gpu_tex_slot_cache[tex_id] = slot
    return slot


def _render_scene_gpu(pygame_surface, meshes, positions, camera_pos, camera_angles,
                      rotations=None, fov_degrees=60.0, near=0.1, far=100.0):
    global _gpu_ctx, _gpu_geom_shader, _gpu_compute_shader, _gpu_color_tex
    global _gpu_readback_buf, _gpu_ssbo, _gpu_ssbo_cap, _gpu_instance_ssbo, _gpu_instance_cap

    sw = pygame_surface.get_width()
    sh = pygame_surface.get_height()
    aspect = sw / sh

    eye, target, up = camera_pose_from_angles(camera_pos, camera_angles)
    mat_vp = build_vp_matrix(eye, target, up, fov_degrees, aspect, near, far).astype(np.float32)
    eye_np = np.asarray(eye, dtype=np.float32)

    instance_rows = []
    total_input_tris = 0

    for name, mesh_entry in meshes.items():
        base_pos = np.asarray(positions.get(name, [0.0, 0.0, 0.0]), dtype=np.float32)
        base_rot = rotations.get(name, [0.0, 0.0, 0.0]) if rotations else [0.0, 0.0, 0.0]
        base_R = rotation_matrix(base_rot).astype(np.float32)

        for submesh in get_submeshes(mesh_entry):
            tri_start, tri_count = _ensure_gpu_mesh(submesh)
            if tri_count <= 0:
                continue

            sub_pos = submesh.get("pos")
            sub_rot = submesh.get("rot")
            if sub_rot is not None:
                R = base_R @ rotation_matrix(sub_rot).astype(np.float32)
            else:
                R = base_R
            if sub_pos is not None:
                pos = base_pos + base_R @ np.asarray(sub_pos, dtype=np.float32)
            else:
                pos = base_pos

            texture = submesh.get("texture")
            has_uvs = (submesh.get("uv_arr") is not None) or (submesh.get("uvs") is not None)
            tex_slot = _ensure_gpu_texture(texture) if (texture is not None and has_uvs) else -1

            instance_rows.append((
                [R[0, 0], R[0, 1], R[0, 2], float(pos[0])],
                [R[1, 0], R[1, 1], R[1, 2], float(pos[1])],
                [R[2, 0], R[2, 1], R[2, 2], float(pos[2])],
                [tri_start, tri_count, tex_slot, total_input_tris]
            ))
            total_input_tris += tri_count

    num_instances = len(instance_rows)
    max_screen_tris = max(total_input_tris * 3, 1)
    required_ssbo_bytes = max_screen_tris * _GPU_DTYPE.itemsize
    if _gpu_ssbo is None or required_ssbo_bytes > _gpu_ssbo_cap:
        if _gpu_ssbo is not None:
            _gpu_ssbo.release()
        _gpu_ssbo = _gpu_ctx.buffer(reserve=max(required_ssbo_bytes * 2, 65536))
        _gpu_ssbo_cap = _gpu_ssbo.size
    max_screen_tris = _gpu_ssbo_cap // _GPU_DTYPE.itemsize

    # Reset atomic screen triangle counter on GPU
    _gpu_counter_ssbo.write(b'\x00\x00\x00\x00')

    if num_instances > 0 and total_input_tris > 0:
        inst_bytes = np.array(instance_rows, dtype=_GPU_INSTANCE_DTYPE).tobytes()
        if _gpu_instance_ssbo is None or len(inst_bytes) > _gpu_instance_cap:
            if _gpu_instance_ssbo is not None:
                _gpu_instance_ssbo.release()
            _gpu_instance_ssbo = _gpu_ctx.buffer(reserve=max(len(inst_bytes) * 2, 4096))
            _gpu_instance_cap = _gpu_instance_ssbo.size
        _gpu_instance_ssbo.write(inst_bytes)

        _gpu_ssbo.bind_to_storage_buffer(0)
        _gpu_raw_tri_ssbo.bind_to_storage_buffer(1)
        _gpu_instance_ssbo.bind_to_storage_buffer(2)
        _gpu_counter_ssbo.bind_to_storage_buffer(3)

        _gpu_geom_shader["u_vp_r0"].value = tuple(mat_vp[0])
        _gpu_geom_shader["u_vp_r1"].value = tuple(mat_vp[1])
        _gpu_geom_shader["u_vp_r2"].value = tuple(mat_vp[2])
        _gpu_geom_shader["u_vp_r3"].value = tuple(mat_vp[3])
        _gpu_geom_shader["u_eye"].value = tuple(eye_np)
        _gpu_geom_shader["u_near"].value = float(near)
        _gpu_geom_shader["u_width"].value = sw
        _gpu_geom_shader["u_height"].value = sh
        _gpu_geom_shader["u_total_input_tris"].value = total_input_tris
        _gpu_geom_shader["u_num_instances"].value = num_instances
        _gpu_geom_shader["u_max_screen_tris"].value = max_screen_tris

        _gpu_geom_shader.run((total_input_tris + 63) // 64, 1, 1)
        _gpu_ctx.memory_barrier()

    _gpu_ssbo.bind_to_storage_buffer(0)
    _gpu_counter_ssbo.bind_to_storage_buffer(3)
    _gpu_tex_data_ssbo.bind_to_storage_buffer(4)
    _gpu_tex_meta_ssbo.bind_to_storage_buffer(5)

    for slot, gl_t in _gpu_textures.items():
        gl_t.use(location=slot)

    shader_mode_id = SHADER_MODES.index(_SHADER_MODE) if _SHADER_MODE in SHADER_MODES else 0
    _gpu_compute_shader["u_width"].value = sw
    _gpu_compute_shader["u_height"].value = sh
    _gpu_compute_shader["u_max_screen_tris"].value = max_screen_tris
    _gpu_compute_shader["u_shader_mode"].value = shader_mode_id
    _gpu_compute_shader["u_eye"].value = tuple(eye_np)
    _gpu_compute_shader["u_light_dir"].value = tuple(_LIGHT_DIR)
    _gpu_compute_shader["u_light_color"].value = tuple(_LIGHT_COLOR)
    _gpu_compute_shader["u_ambient_sky"].value = tuple(_AMBIENT_SKY)
    _gpu_compute_shader["u_ambient_ground"].value = tuple(_AMBIENT_GROUND)
    _gpu_compute_shader["u_far"].value = float(far)

    _gpu_compute_shader.run((sw + 15) // 16, (sh + 15) // 16, 1)
    _gpu_ctx.memory_barrier()

    _gpu_color_tex.read_into(_gpu_readback_buf)
    surf = pygame.image.frombuffer(_gpu_readback_buf, (sw, sh), 'RGBA')
    pygame_surface.blit(surf, (0, 0))


def render_scene(pygame_surface, meshes, positions, camera_pos, camera_angles,
                 rotations=None, fov_degrees=60.0, near=0.1, far=100.0):
    sw = pygame_surface.get_width()
    sh = pygame_surface.get_height()

    if _USE_GPU_COMPUTE and _init_gpu_compute(sw, sh):
        _render_scene_gpu(pygame_surface, meshes, positions, camera_pos, camera_angles,
                          rotations, fov_degrees, near, far)
        return

    aspect = sw / sh
    eye, target, up = camera_pose_from_angles(camera_pos, camera_angles)
    mat_vp   = build_vp_matrix(eye, target, up, fov_degrees, aspect, near, far)

    zbuf = np.full((sw, sh), np.inf, dtype=np.float32)
    opaque_list = []
    trans_list  = []

    eye_np = np.asarray(eye, dtype=np.float32)

    for name, mesh_entry in meshes.items():
        base_pos = np.array(positions.get(name, [0.0, 0.0, 0.0]), dtype=np.float32)
        base_rot = rotations.get(name, [0.0, 0.0, 0.0]) if rotations else [0.0, 0.0, 0.0]
        base_R   = rotation_matrix(base_rot).astype(np.float32)

        for mesh in get_submeshes(mesh_entry):
            sub_pos = mesh.get("pos")
            sub_rot = mesh.get("rot")
            R   = (base_R @ rotation_matrix(sub_rot).astype(np.float32)) if sub_rot is not None else base_R
            pos = (base_pos + base_R @ np.asarray(sub_pos, dtype=np.float32)) if sub_pos is not None else base_pos

            texture    = mesh.get("texture")
            uvs        = mesh.get("uvs")
            colors     = mesh.get("colors")
            mesh_alpha = float(mesh.get("alpha", 1.0))
            if mesh_alpha < 0.01:
                continue

            if "verts" in mesh and mesh["verts"] is not None:
                verts  = mesh["verts"]
                uv_arr = mesh.get("uv_arr")
            else:
                verts  = np.array(mesh["tris"], dtype=np.float32) if len(mesh.get("tris", [])) > 0 else np.zeros((0, 3, 3), dtype=np.float32)
                uv_arr = np.array(uvs, dtype=np.float32) if uvs else None

            N = verts.shape[0]
            if N == 0:
                continue
            flat  = verts.reshape(-1, 3)
            world = flat @ R.T + pos

            ones = np.ones((N * 3, 1), dtype=np.float32)
            hom  = np.concatenate([world, ones], axis=1)
            clip = (mat_vp @ hom.T).T

            clip  = clip.reshape(N, 3, 4)
            world = world.reshape(N, 3, 3)

            edge0   = world[:, 1] - world[:, 0]
            edge1   = world[:, 2] - world[:, 0]
            normals = np.cross(edge0, edge1)
            norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
            unit_normals = normals / np.maximum(norm_len, 1e-10)
            to_cam  = eye_np - world[:, 0]
            facing  = (normals * to_cam).sum(axis=1) > 0

            centroids = world.mean(axis=1)
            shade_mults, shade_adds, fogs = _compute_cpu_shading(unit_normals, centroids, eye_np, far)

            clip_w   = np.maximum(clip[:, :, 3], 1e-6)
            ndc_z    = clip[:, :, 2] / clip_w
            centroid_ndc_z = ndc_z.mean(axis=1)

            has_tex_alpha = (texture is not None and texture.ndim == 3 and texture.shape[2] == 4)

            for idx in range(N):
                if not facing[idx]:
                    continue

                sort_z = float(centroid_ndc_z[idx])
                cv     = clip[idx]
                s_mult = shade_mults[idx]
                s_add  = shade_adds[idx]
                fog_v  = float(fogs[idx, 0])

                raw_col = colors[idx] if (colors is not None and idx < len(colors)) else None
                tri_alpha = int(round((raw_col[3] if (raw_col is not None and len(raw_col) >= 4) else 255) * mesh_alpha))
                if tri_alpha < 3:
                    continue

                if texture is not None and uv_arr is not None:
                    uv = uv_arr[idx]
                    c_verts = [np.append(cv[j], uv[j]) for j in range(3)]
                    clipped = clip_to_frustum(c_verts, near)
                    if len(clipped) < 3:
                        continue
                    tint_rgba = [raw_col[0], raw_col[1], raw_col[2], tri_alpha] if raw_col is not None else [255, 255, 255, tri_alpha]
                    is_trans = has_tex_alpha or (tri_alpha < 254)
                    target_list = trans_list if is_trans else opaque_list
                    for i in range(1, len(clipped) - 1):
                        pa, pb, pc  = clipped[0], clipped[i], clipped[i+1]
                        pts         = [to_pixel(pa, sw, sh),
                                       to_pixel(pb, sw, sh),
                                       to_pixel(pc, sw, sh)]
                        tri_uvs     = [[pa[4], pa[5]], [pb[4], pb[5]], [pc[4], pc[5]]]
                        tri_ws      = [pa[3], pb[3], pc[3]]
                        tri_zs      = [pa[2]/max(pa[3], 1e-6),
                                       pb[2]/max(pb[3], 1e-6),
                                       pc[2]/max(pc[3], 1e-6)]
                        target_list.append((sort_z, pts, texture, tri_uvs, tri_ws, tri_zs, None, s_mult, s_add, fog_v, tint_rgba))
                else:
                    c_verts = [cv[j] for j in range(3)]
                    clipped = clip_to_frustum(c_verts, near)
                    if len(clipped) < 3:
                        continue
                    base_rgb = np.asarray(raw_col[:3] if raw_col is not None else (200, 200, 200), dtype=np.float32)
                    lit_rgb = base_rgb * s_mult + s_add * 255.0
                    if fog_v > 0.0:
                        fog_col = np.array([20.4, 25.5, 35.7], dtype=np.float32)
                        lit_rgb = lit_rgb * (1.0 - fog_v) + fog_col * fog_v
                    lit_rgb = np.clip(lit_rgb, 0.0, 255.0)
                    shaded_color = [int(lit_rgb[0]), int(lit_rgb[1]), int(lit_rgb[2]), tri_alpha]
                    target_list = trans_list if tri_alpha < 254 else opaque_list
                    for i in range(1, len(clipped) - 1):
                        pa, pb, pc = clipped[0], clipped[i], clipped[i+1]
                        pts  = [to_pixel(pa, sw, sh),
                                to_pixel(pb, sw, sh),
                                to_pixel(pc, sw, sh)]
                        tri_zs = [pa[2]/max(pa[3], 1e-6),
                                   pb[2]/max(pb[3], 1e-6),
                                   pc[2]/max(pc[3], 1e-6)]
                        target_list.append((sort_z, pts, None, None, None, tri_zs, shaded_color, None, None, 0.0, None))

    # Draw opaque front-to-back, then translucent back-to-front
    opaque_list.sort(key=lambda x: x[0])
    trans_list.sort(key=lambda x: x[0], reverse=True)

    screen_arr = pygame.surfarray.pixels3d(pygame_surface)
    for sort_z, pts, texture, uv, ws, tri_zs, color, s_mult, s_add, fog_v, tint_rgba in opaque_list:
        if texture is not None and uv is not None:
            draw_textured_triangle(screen_arr, zbuf, texture, pts, uv, ws, tri_zs, s_mult, s_add, fog_v, tint_rgba)
        else:
            draw_flat_triangle(screen_arr, zbuf, pts, tri_zs, color or (200, 200, 200, 255))

    for sort_z, pts, texture, uv, ws, tri_zs, color, s_mult, s_add, fog_v, tint_rgba in trans_list:
        if texture is not None and uv is not None:
            draw_textured_triangle(screen_arr, zbuf, texture, pts, uv, ws, tri_zs, s_mult, s_add, fog_v, tint_rgba)
        else:
            draw_flat_triangle(screen_arr, zbuf, pts, tri_zs, color or (200, 200, 200, 255))
    del screen_arr


def draw_aabb_debug(pygame_surface, mn, mx, camera_pos, camera_angles,
                    color=(0, 255, 0), fov_degrees=60.0, near=0.1, far=100.0):
    sw = pygame_surface.get_width()
    sh = pygame_surface.get_height()
    aspect = sw / sh

    eye, target, up = camera_pose_from_angles(camera_pos, camera_angles)
    mat_vp = build_vp_matrix(eye, target, up, fov_degrees, aspect, near, far)

    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ], dtype=np.float32)

    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]

    ones = np.ones((8, 1), dtype=np.float32)
    hom  = np.concatenate([corners, ones], axis=1)
    clip = (mat_vp @ hom.T).T

    for a, b in edges:
        ca = clip[a].copy()
        cb = clip[b].copy()
        wa, wb = ca[3], cb[3]
        if wa <= near and wb <= near:
            continue
        if wa <= near:
            t = (near - wa) / (wb - wa)
            ca = ca + t * (cb - ca)
        elif wb <= near:
            t = (near - wb) / (wa - wb)
            cb = cb + t * (ca - cb)
        pa = to_pixel(ca, sw, sh)
        pb = to_pixel(cb, sw, sh)
        pygame.draw.line(pygame_surface, color,
                         (int(np.clip(pa[0], -10000, 10000)), int(np.clip(pa[1], -10000, 10000))),
                         (int(np.clip(pb[0], -10000, 10000)), int(np.clip(pb[1], -10000, 10000))), 1)


def draw_normals_debug(pygame_surface, mesh_entry, pos, rot=None,
                       camera_pos=(0.0, 0.0, 0.0), camera_angles=(0.0, 0.0, 0.0),
                       length=0.35, color=(255, 60, 220), tip_color=(80, 255, 255),
                       fov_degrees=60.0, near=0.1, far=100.0):
    sw = pygame_surface.get_width()
    sh = pygame_surface.get_height()
    aspect = sw / sh

    eye, target, up = camera_pose_from_angles(camera_pos, camera_angles)
    mat_vp = build_vp_matrix(eye, target, up, fov_degrees, aspect, near, far)

    base_pos = np.asarray(pos, dtype=np.float32)
    base_rot = rot if rot is not None else [0.0, 0.0, 0.0]
    base_R   = rotation_matrix(base_rot).astype(np.float32)

    for submesh in get_submeshes(mesh_entry):
        verts = submesh.get("verts")
        if verts is None:
            tris = submesh.get("tris", [])
            if not tris:
                continue
            verts = np.array(tris, dtype=np.float32)
        if len(verts) == 0:
            continue

        sub_pos = submesh.get("pos")
        sub_rot = submesh.get("rot")
        R = (base_R @ rotation_matrix(sub_rot).astype(np.float32)) if sub_rot is not None else base_R
        w_pos = (base_pos + base_R @ np.asarray(sub_pos, dtype=np.float32)) if sub_pos is not None else base_pos

        N = verts.shape[0]
        world = (verts.reshape(-1, 3) @ R.T + w_pos).reshape(N, 3, 3)

        edge0 = world[:, 1] - world[:, 0]
        edge1 = world[:, 2] - world[:, 0]
        normals = np.cross(edge0, edge1)
        norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
        valid = norm_len[:, 0] > 1e-10
        if not np.any(valid):
            continue

        world = world[valid]
        unit_normals = normals[valid] / norm_len[valid]
        centroids = world.mean(axis=1)
        tips = centroids + unit_normals * float(length)

        M = len(centroids)
        ones = np.ones((M, 1), dtype=np.float32)
        clip_start = (mat_vp @ np.concatenate([centroids, ones], axis=1).T).T
        clip_end   = (mat_vp @ np.concatenate([tips, ones], axis=1).T).T

        for i in range(M):
            ca = clip_start[i].copy()
            cb = clip_end[i].copy()
            wa, wb = ca[3], cb[3]
            if wa <= near and wb <= near:
                continue
            if wa > far and wb > far:
                continue
            if wa <= near:
                t = (near - wa) / (wb - wa)
                ca = ca + t * (cb - ca)
            elif wb <= near:
                t = (near - wb) / (wa - wb)
                cb = cb + t * (ca - cb)

            pa = to_pixel(ca, sw, sh)
            pb = to_pixel(cb, sw, sh)
            x0, y0 = int(np.clip(pa[0], -10000, 10000)), int(np.clip(pa[1], -10000, 10000))
            x1, y1 = int(np.clip(pb[0], -10000, 10000)), int(np.clip(pb[1], -10000, 10000))

            if (x0 < 0 and x1 < 0) or (x0 >= sw and x1 >= sw) or (y0 < 0 and y1 < 0) or (y0 >= sh and y1 >= sh):
                continue

            pygame.draw.line(pygame_surface, color, (x0, y0), (x1, y1), 1)
            if 0 <= x1 < sw and 0 <= y1 < sh:
                pygame.draw.circle(pygame_surface, tip_color, (x1, y1), 2)


