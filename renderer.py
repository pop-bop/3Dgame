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


def draw_textured_triangle(screen_arr, zbuf, texture, screen_pts, uvs, ws, zs):
    """Rasterize a textured triangle with perspective-correct UVs and depth testing.

    zs : NDC z per vertex (clip_z / clip_w).  Smaller value = closer to camera.
    zbuf : (sw, sh) float32 array; pixel written only when depth < current entry.
    """
    th, tw = texture.shape[:2]
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

    # Depth test — linear interpolation of NDC z (sufficient for non-degenerate geo)
    depth   = b0*z[0] + b1*z[1] + b2*z[2]
    visible = depth < zbuf[px, py]
    if not visible.any():
        return
    px, py  = px[visible], py[visible]
    b0, b1, b2 = b0[visible], b1[visible], b2[visible]
    zbuf[px, py] = depth[visible]

    # Perspective-correct UV interpolation
    inv_w        = 1.0 / w
    interp_inv_w = b0*inv_w[0] + b1*inv_w[1] + b2*inv_w[2]
    u = (b0*uv[0,0]*inv_w[0] + b1*uv[1,0]*inv_w[1] + b2*uv[2,0]*inv_w[2]) / interp_inv_w
    v = (b0*uv[0,1]*inv_w[0] + b1*uv[1,1]*inv_w[1] + b2*uv[2,1]*inv_w[2]) / interp_inv_w

    tx = np.clip((u * tw).astype(np.int32), 0, tw-1)
    ty = np.clip((v * th).astype(np.int32), 0, th-1)

    bgr = texture[ty, tx]
    screen_arr[px, py] = bgr[:, ::-1]   # BGR → RGB


def draw_flat_triangle(screen_arr, zbuf, screen_pts, zs, color):
    """Rasterize a flat-shaded triangle with depth testing.

    Mirrors draw_textured_triangle but writes a single colour per pixel instead
    of sampling a texture.  Using the same rasterizer for both triangle types
    means the z-buffer is shared and painter's-algorithm sorting order no longer
    matters for correctness.
    """
    sw, sh = screen_arr.shape[0], screen_arr.shape[1]
    p = np.array(screen_pts, dtype=np.float32)
    z = np.array(zs,         dtype=np.float32)

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

    # Depth test
    depth   = b0*z[0] + b1*z[1] + b2*z[2]
    visible = depth < zbuf[px, py]
    if not visible.any():
        return
    px, py = px[visible], py[visible]
    zbuf[px, py] = depth[visible]

    # Write colour (convert RGB → BGR to match surfarray layout)
    screen_arr[px, py] = np.array(color, dtype=np.uint8)[::-1]


def render_scene(pygame_surface, meshes, positions, camera_pos, camera_angles,
                 rotations=None, fov_degrees=60.0, near=0.1, far=100.0):
    sw = pygame_surface.get_width()
    sh = pygame_surface.get_height()
    aspect = sh / sw

    eye, target, up = camera_pose_from_angles(camera_pos, camera_angles)
    mat_view = build_view_matrix(eye, target, up)
    mat_vp   = build_vp_matrix(eye, target, up, fov_degrees, aspect, near, far)

    # Z-buffer: one depth value per pixel, initialised to +∞ (nothing drawn yet).
    # NDC z ranges from -1 (near) to +1 (far), so any real triangle will be closer.
    zbuf = np.full((sw, sh), np.inf, dtype=np.float32)

    # draw_list entries: (ndc_z_centroid, pts, texture, uvs, ws, ndc_zs, color)
    # ndc_z_centroid is kept only for the sort; per-pixel depth comes from ndc_zs.
    draw_list = []

    for name, mesh in meshes.items():
        pos     = np.array(positions.get(name, [0.0, 0.0, 0.0]), dtype=np.float32)
        rot     = rotations.get(name, [0.0, 0.0, 0.0]) if rotations else [0.0, 0.0, 0.0]
        R       = rotation_matrix(rot).astype(np.float32)
        texture = mesh.get("texture")
        uvs     = mesh.get("uvs")
        colors  = mesh.get("colors")

        if "verts" in mesh:
            verts  = mesh["verts"]
            uv_arr = mesh.get("uv_arr")
        else:
            verts  = np.array(mesh["tris"], dtype=np.float32)
            uv_arr = np.array(uvs, dtype=np.float32) if uvs else None

        N = verts.shape[0]

        flat  = verts.reshape(-1, 3)
        world = flat @ R.T + pos

        ones = np.ones((N * 3, 1), dtype=np.float32)
        hom  = np.concatenate([world, ones], axis=1)
        clip = (mat_vp @ hom.T).T         # (N*3, 4) homogeneous clip coords

        clip  = clip.reshape(N, 3, 4)
        world = world.reshape(N, 3, 3)

        # Back-face cull (world space) — avoids painter's-sort instability on
        # double-sided geometry where front and back face share the same centroid.
        eye_np  = np.asarray(eye, dtype=np.float32)
        edge0   = world[:, 1] - world[:, 0]
        edge1   = world[:, 2] - world[:, 0]
        normals = np.cross(edge0, edge1)
        to_cam  = eye_np - world[:, 0]
        facing  = (normals * to_cam).sum(axis=1) > 0   # (N,) bool

        # NDC z per vertex: clip_z / clip_w.  Used for per-pixel depth testing.
        # Clamp clip_w away from zero to avoid divide-by-zero on the near plane.
        clip_w   = np.maximum(clip[:, :, 3], 1e-6)   # (N,3)
        ndc_z    = clip[:, :, 2] / clip_w             # (N,3)  range [-1, +1]

        # Centroid NDC z for the painter's sort (coarse ordering only).
        centroid_ndc_z = ndc_z.mean(axis=1)           # (N,)

        for idx in range(N):
            if not facing[idx]:
                continue

            sort_z = float(centroid_ndc_z[idx])
            cv     = clip[idx]          # (3,4)
            zv     = ndc_z[idx]         # (3,)  per-vertex NDC depths

            if texture is not None and uv_arr is not None:
                uv = uv_arr[idx]
                # Pack clip + uv into a single vector for frustum clipping
                c_verts = [np.append(cv[j], uv[j]) for j in range(3)]
                clipped = clip_to_frustum(c_verts, near)
                if len(clipped) < 3:
                    continue
                for i in range(1, len(clipped) - 1):
                    pa, pb, pc  = clipped[0], clipped[i], clipped[i+1]
                    pts         = [to_pixel(pa, sw, sh),
                                   to_pixel(pb, sw, sh),
                                   to_pixel(pc, sw, sh)]
                    tri_uvs     = [[pa[4], pa[5]], [pb[4], pb[5]], [pc[4], pc[5]]]
                    tri_ws      = [pa[3], pb[3], pc[3]]
                    # NDC z for clipped verts: clip_z / clip_w
                    tri_zs      = [pa[2]/max(pa[3], 1e-6),
                                   pb[2]/max(pb[3], 1e-6),
                                   pc[2]/max(pc[3], 1e-6)]
                    draw_list.append((sort_z, pts, texture, tri_uvs, tri_ws, tri_zs, None))
            else:
                c_verts = [cv[j] for j in range(3)]
                clipped = clip_to_frustum(c_verts, near)
                if len(clipped) < 3:
                    continue
                color = colors[idx] if colors else None
                for i in range(1, len(clipped) - 1):
                    pa, pb, pc = clipped[0], clipped[i], clipped[i+1]
                    pts  = [to_pixel(pa, sw, sh),
                            to_pixel(pb, sw, sh),
                            to_pixel(pc, sw, sh)]
                    tri_zs = [pa[2]/max(pa[3], 1e-6),
                               pb[2]/max(pb[3], 1e-6),
                               pc[2]/max(pc[3], 1e-6)]
                    draw_list.append((sort_z, pts, None, None, None, tri_zs, color))

    # Painter's sort is now only a hint — the z-buffer handles correctness.
    draw_list.sort(key=lambda x: x[0])

    screen_arr = pygame.surfarray.pixels3d(pygame_surface)
    for sort_z, pts, texture, uv, ws, tri_zs, color in draw_list:
        if texture is not None and uv is not None:
            draw_textured_triangle(screen_arr, zbuf, texture, pts, uv, ws, tri_zs)
        else:
            draw_flat_triangle(screen_arr, zbuf, pts, tri_zs, color or (200, 200, 200))
    del screen_arr


def draw_aabb_debug(pygame_surface, mn, mx, camera_pos, camera_angles,
                    color=(0, 255, 0), fov_degrees=60.0, near=0.1, far=100.0):
    """
    Project the 12 edges of an AABB into screen space and draw them as lines.
    mn, mx : array-like (3,) — world-space min/max corners.
    color  : RGB tuple.
    """
    sw = pygame_surface.get_width()
    sh = pygame_surface.get_height()
    aspect = sh / sw

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
    clip = (mat_vp @ hom.T).T   # (8, 4)

    for a, b in edges:
        wa, wb = clip[a, 3], clip[b, 3]
        if wa <= near or wb <= near:
            continue
        pa = to_pixel(clip[a], sw, sh)
        pb = to_pixel(clip[b], sw, sh)
        pygame.draw.line(pygame_surface, color,
                         (int(pa[0]), int(pa[1])),
                         (int(pb[0]), int(pb[1])), 1)


