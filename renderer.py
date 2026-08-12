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


# ── Clipping ─────────────────────────────────────────────────────────────────
#
# Vertices are numpy arrays that may carry extra data beyond [x, y, z, w].
# For textured triangles we pack UVs as [x, y, z, w, u, v] so that the linear
# interpolation done at clip boundaries also interpolates the texture coords —
# this is what prevents the glitch when a face straddles the FOV edge.

def _clip_axis(verts, axis, sign):
    def inside(v):
        return sign * v[axis] <= v[3]

    def intersect(a, b):
        da = a[3] - sign * a[axis]
        db = b[3] - sign * b[axis]
        t  = da / (da - db)
        return a + t * (b - a)   # works for any array length, including u,v

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


# ── Texture drawing ───────────────────────────────────────────────────────────
#
# Uses getAffineTransform (3-point direct mapping) instead of the old
# getPerspectiveTransform parallelogram trick — simpler and correct.

def _warp_triangle(texture, src_uv_tri, dst_screen_tri, out_w, out_h):
    h, w  = texture.shape[:2]
    src   = np.float32([[u * w, v * h] for u, v in src_uv_tri])
    dst   = np.float32(dst_screen_tri)

    M = cv2.getAffineTransform(src, dst)

    if _USE_CUDA:
        gpu_src    = cv2.cuda_GpuMat()
        gpu_src.upload(texture)
        gpu_warped = cv2.cuda.warpAffine(gpu_src, M, (out_w, out_h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
        warped = gpu_warped.download()
    else:
        warped = cv2.warpAffine(texture, M, (out_w, out_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))

    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    cv2.fillPoly(mask, np.int32([dst_screen_tri]), 255)
    return warped, mask


def _blit_onto_surface(pygame_surface, bgr_img, mask):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    px      = pygame.surfarray.pixels3d(pygame_surface)
    mask_t  = mask.T
    img_t   = np.transpose(rgb_img, (1, 0, 2))
    px[mask_t == 255] = img_t[mask_t == 255]
    del px


def render_scene(pygame_surface, meshes, positions, camera_pos, camera_angles,
                 fov_degrees=60.0, near=0.1, far=100.0):
    sw = pygame_surface.get_width()
    sh = pygame_surface.get_height()
    aspect = sh / sw

    eye, target, up = camera_pose_from_angles(camera_pos, camera_angles)
    mat_view = build_view_matrix(eye, target, up)
    mat_vp   = build_vp_matrix(eye, target, up, fov_degrees, aspect, near, far)

    draw_list = []

    for name, mesh in meshes.items():
        pos     = positions.get(name, [0.0, 0.0, 0.0])
        tris    = mesh["tris"]
        texture = mesh.get("texture")
        uvs     = mesh.get("uvs")
        colors  = mesh.get("colors")

        for idx, tri in enumerate(tris):
            v0 = np.array([tri[0][0]+pos[0], tri[0][1]+pos[1], tri[0][2]+pos[2]])
            v1 = np.array([tri[1][0]+pos[0], tri[1][1]+pos[1], tri[1][2]+pos[2]])
            v2 = np.array([tri[2][0]+pos[0], tri[2][1]+pos[1], tri[2][2]+pos[2]])

            centroid_world = (v0 + v1 + v2) / 3.0
            cam_z = (mat_view @ np.append(centroid_world, 1.0))[2]

            if texture is not None and uvs is not None:
                # Pack UVs into the clip vector as [x, y, z, w, u, v] so the
                # clipper linearly interpolates UVs at every boundary it creates.
                uv = uvs[idx]
                c0 = np.append(np.dot(mat_vp, np.append(v0, 1.0)), uv[0])
                c1 = np.append(np.dot(mat_vp, np.append(v1, 1.0)), uv[1])
                c2 = np.append(np.dot(mat_vp, np.append(v2, 1.0)), uv[2])
                clipped = clip_to_frustum([c0, c1, c2], near)
                if len(clipped) < 3:
                    continue

                for i in range(1, len(clipped) - 1):
                    pa = clipped[0]
                    pb = clipped[i]
                    pc = clipped[i + 1]
                    screen_pts = [to_pixel(pa, sw, sh),
                                  to_pixel(pb, sw, sh),
                                  to_pixel(pc, sw, sh)]
                    # Read the interpolated UVs directly — no divide by w needed.
                    # UVs are object-space attributes; dividing by clip-w would
                    # scale them with depth, which causes textures to scroll as
                    # the camera moves.
                    clipped_uvs = [[pa[4], pa[5]],
                                   [pb[4], pb[5]],
                                   [pc[4], pc[5]]]
                    draw_list.append((cam_z, screen_pts, texture, clipped_uvs, None))

            else:
                verts_4d = [np.dot(mat_vp, np.append(v, 1.0)) for v in (v0, v1, v2)]
                clipped  = clip_to_frustum(verts_4d, near)
                if len(clipped) < 3:
                    continue

                color = colors[idx] if colors else None
                for i in range(1, len(clipped) - 1):
                    pts = [to_pixel(clipped[0],     sw, sh),
                           to_pixel(clipped[i],     sw, sh),
                           to_pixel(clipped[i + 1], sw, sh)]
                    draw_list.append((cam_z, pts, None, None, color))

    draw_list.sort(key=lambda x: x[0])

    for cam_z, pts, texture, uv, color in draw_list:
        if texture is not None and uv is not None:
            warped, mask = _warp_triangle(texture, uv, pts, sw, sh)
            _blit_onto_surface(pygame_surface, warped, mask)
        else:
            pygame.draw.polygon(pygame_surface, color or (200, 200, 200), pts)


