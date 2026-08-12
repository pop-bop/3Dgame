import struct
import numpy as np
import cv2
import pygame

_USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0

_MAGIC   = b"QWZX"
_VERSION = 1


def save_qwzx(path, meshes):
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("B", _VERSION))
        f.write(struct.pack("<I", len(meshes)))
        for mesh in meshes:
            name_bytes = mesh["name"].encode("utf-8")
            f.write(struct.pack("B", len(name_bytes)))
            f.write(name_bytes)

            texture = mesh.get("texture")
            has_tex = 1 if texture is not None else 0
            f.write(struct.pack("B", has_tex))

            tris = mesh["tris"]
            f.write(struct.pack("<I", len(tris)))

            if has_tex:
                h, w = texture.shape[:2]
                f.write(struct.pack("<II", w, h))
                f.write(texture.astype(np.uint8).tobytes())
                uvs = mesh["uvs"]
                for tri, uv in zip(tris, uvs):
                    for vx, vy, vz in tri:
                        f.write(struct.pack("<fff", vx, vy, vz))
                    for u, v in uv:
                        f.write(struct.pack("<ff", u, v))
            else:
                colors = mesh["colors"]
                for tri, color in zip(tris, colors):
                    for vx, vy, vz in tri:
                        f.write(struct.pack("<fff", vx, vy, vz))
                    f.write(struct.pack("BBB", int(color[0]), int(color[1]), int(color[2])))


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
            has_tex  = struct.unpack("B", f.read(1))[0]
            num_tris = struct.unpack("<I", f.read(4))[0]

            tris    = []
            colors  = []
            uvs     = []
            texture = None

            if has_tex:
                w, h    = struct.unpack("<II", f.read(8))
                tex_raw = f.read(w * h * 3)
                texture = np.frombuffer(tex_raw, dtype=np.uint8).reshape(h, w, 3).copy()
                for _ in range(num_tris):
                    raw = struct.unpack("<fffffffff", f.read(36))
                    tri = [[raw[0],raw[1],raw[2]],
                           [raw[3],raw[4],raw[5]],
                           [raw[6],raw[7],raw[8]]]
                    uv_raw = struct.unpack("<ffffff", f.read(24))
                    uv = [[uv_raw[0],uv_raw[1]],
                          [uv_raw[2],uv_raw[3]],
                          [uv_raw[4],uv_raw[5]]]
                    tris.append(tri)
                    uvs.append(uv)
            else:
                for _ in range(num_tris):
                    raw = struct.unpack("<fffffffff", f.read(36))
                    tri = [[raw[0],raw[1],raw[2]],
                           [raw[3],raw[4],raw[5]],
                           [raw[6],raw[7],raw[8]]]
                    color = list(struct.unpack("BBB", f.read(3)))
                    tris.append(tri)
                    colors.append(color)

            meshes.append({
                "name":    name,
                "tris":    tris,
                "colors":  colors if not has_tex else None,
                "texture": texture,
                "uvs":     uvs if has_tex else None,
            })
    return meshes


def prepare_mesh(mesh):
    """Call once after load_qwzx. Pre-converts geometry lists to numpy arrays
    so render_scene can transform all vertices with a single matrix multiply
    instead of a Python loop over every triangle."""
    mesh["verts"]  = np.array(mesh["tris"], dtype=np.float32)  # (N, 3, 3)
    mesh["uv_arr"] = np.array(mesh["uvs"],  dtype=np.float32) if mesh["uvs"] else None  # (N, 3, 2)
    return mesh


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


def draw_textured_triangle(screen_arr, texture, screen_pts, uvs, ws):

    th, tw = texture.shape[:2]
    sw, sh = screen_arr.shape[0], screen_arr.shape[1]
    p  = np.array(screen_pts, dtype=np.float32)
    uv = np.array(uvs,        dtype=np.float32)
    w  = np.array(ws,         dtype=np.float32)

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

    # perspective-correct UV interpolation
    inv_w        = 1.0 / w
    interp_inv_w = b0*inv_w[0] + b1*inv_w[1] + b2*inv_w[2]
    u = (b0*uv[0,0]*inv_w[0] + b1*uv[1,0]*inv_w[1] + b2*uv[2,0]*inv_w[2]) / interp_inv_w
    v = (b0*uv[0,1]*inv_w[0] + b1*uv[1,1]*inv_w[1] + b2*uv[2,1]*inv_w[2]) / interp_inv_w

    tx = np.clip((u * tw).astype(np.int32), 0, tw-1)
    ty = np.clip((v * th).astype(np.int32), 0, th-1)

    bgr = texture[ty, tx]
    screen_arr[px, py] = bgr[:, ::-1]   # BGR → RGB


def render_scene(pygame_surface, meshes, positions, camera_pos, camera_angles,
                 rotations=None, fov_degrees=60.0, near=0.1, far=100.0):
    sw = pygame_surface.get_width()
    sh = pygame_surface.get_height()
    aspect = sh / sw

    eye, target, up = camera_pose_from_angles(camera_pos, camera_angles)
    mat_view = build_view_matrix(eye, target, up)
    mat_vp   = build_vp_matrix(eye, target, up, fov_degrees, aspect, near, far)

    draw_list = []

    for name, mesh in meshes.items():
        pos     = np.array(positions.get(name, [0.0, 0.0, 0.0]), dtype=np.float32)
        rot     = rotations.get(name, [0.0, 0.0, 0.0]) if rotations else [0.0, 0.0, 0.0]
        R       = rotation_matrix(rot).astype(np.float32)
        texture = mesh.get("texture")
        uvs     = mesh.get("uvs")
        colors  = mesh.get("colors")


        if "verts" in mesh:
            verts = mesh["verts"]
            uv_arr = mesh.get("uv_arr")
        else:
            verts  = np.array(mesh["tris"], dtype=np.float32)
            uv_arr = np.array(uvs, dtype=np.float32) if uvs else None

        N = verts.shape[0]


        flat  = verts.reshape(-1, 3)
        world = flat @ R.T + pos

        ones  = np.ones((N * 3, 1), dtype=np.float32)
        hom   = np.concatenate([world, ones], axis=1)
        clip  = (mat_vp @ hom.T).T

        clip  = clip.reshape(N, 3, 4)
        world = world.reshape(N, 3, 3)

        centroids_w = world.mean(axis=1)
        ones_n      = np.ones((N, 1), dtype=np.float32)
        centroids_h = np.concatenate([centroids_w, ones_n], axis=1)
        cam_zs      = (mat_view @ centroids_h.T)[2]


        for idx in range(N):
            cam_z = float(cam_zs[idx])

            if texture is not None and uv_arr is not None:
                uv = uv_arr[idx]

                cv = clip[idx]
                c_verts = [np.append(cv[j], uv[j]) for j in range(3)]
                clipped = clip_to_frustum(c_verts, near)
                if len(clipped) < 3:
                    continue
                for i in range(1, len(clipped) - 1):
                    pa, pb, pc = clipped[0], clipped[i], clipped[i+1]
                    pts     = [to_pixel(pa, sw, sh), to_pixel(pb, sw, sh), to_pixel(pc, sw, sh)]
                    tri_uvs = [[pa[4], pa[5]], [pb[4], pb[5]], [pc[4], pc[5]]]
                    tri_ws  = [pa[3], pb[3], pc[3]]
                    draw_list.append((cam_z, pts, texture, tri_uvs, tri_ws, None))

            else:
                cv      = clip[idx]
                c_verts = [cv[j] for j in range(3)]
                clipped = clip_to_frustum(c_verts, near)
                if len(clipped) < 3:
                    continue
                color = colors[idx] if colors else None
                for i in range(1, len(clipped) - 1):
                    pts = [to_pixel(clipped[0],     sw, sh),
                           to_pixel(clipped[i],     sw, sh),
                           to_pixel(clipped[i + 1], sw, sh)]
                    draw_list.append((cam_z, pts, None, None, None, color))

    draw_list.sort(key=lambda x: x[0])


    screen_arr = pygame.surfarray.pixels3d(pygame_surface)
    for cam_z, pts, texture, uv, ws, color in draw_list:
        if texture is not None and uv is not None:
            draw_textured_triangle(screen_arr, texture, pts, uv, ws)
        else:
            del screen_arr
            pygame.draw.polygon(pygame_surface, color or (200, 200, 200), pts)
            screen_arr = pygame.surfarray.pixels3d(pygame_surface)
    del screen_arr


