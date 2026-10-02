import struct
import numpy as np
import cv2
import pygame

_MAGIC   = b"QWZX"
_VERSION = 1

_USE_GPU_COMPUTE = True
_gpu_ctx = None
_gpu_compute_shader = None
_gpu_color_tex = None
_gpu_readback_buf = None
_gpu_ssbo = None
_gpu_ssbo_cap = 0
_gpu_textures = {}
_gpu_w = 0
_gpu_h = 0


def set_gpu_acceleration(enabled: bool):
    global _USE_GPU_COMPUTE
    _USE_GPU_COMPUTE = enabled


def is_gpu_acceleration_enabled() -> bool:
    return _USE_GPU_COMPUTE and (_gpu_ctx is not None or _init_gpu_compute(640, 640))


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
    mesh["verts"]  = np.array(mesh["tris"], dtype=np.float32)
    mesh["uv_arr"] = np.array(mesh["uvs"],  dtype=np.float32) if mesh["uvs"] else None
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

    depth   = b0*z[0] + b1*z[1] + b2*z[2]
    visible = depth < zbuf[px, py]
    if not visible.any():
        return
    px, py  = px[visible], py[visible]
    b0, b1, b2 = b0[visible], b1[visible], b2[visible]
    zbuf[px, py] = depth[visible]

    inv_w        = 1.0 / w
    interp_inv_w = b0*inv_w[0] + b1*inv_w[1] + b2*inv_w[2]
    u = (b0*uv[0,0]*inv_w[0] + b1*uv[1,0]*inv_w[1] + b2*uv[2,0]*inv_w[2]) / interp_inv_w
    v = (b0*uv[0,1]*inv_w[0] + b1*uv[1,1]*inv_w[1] + b2*uv[2,1]*inv_w[2]) / interp_inv_w

    tx = np.clip((u * tw).astype(np.int32), 0, tw-1)
    ty = np.clip((v * th).astype(np.int32), 0, th-1)

    bgr = texture[ty, tx]
    screen_arr[px, py] = bgr[:, ::-1]


def draw_flat_triangle(screen_arr, zbuf, screen_pts, zs, color):
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

    depth   = b0*z[0] + b1*z[1] + b2*z[2]
    visible = depth < zbuf[px, py]
    if not visible.any():
        return
    px, py = px[visible], py[visible]
    zbuf[px, py] = depth[visible]

    screen_arr[px, py] = np.array(color, dtype=np.uint8)[::-1]


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

vec4 sample_mesh_tex(int id, vec2 uv) {
    vec2 clamped_uv = clamp(uv, 0.0, 1.0);
    if (id == 0) return textureLod(u_tex0, clamped_uv, 0.0);
    if (id == 1) return textureLod(u_tex1, clamped_uv, 0.0);
    if (id == 2) return textureLod(u_tex2, clamped_uv, 0.0);
    if (id == 3) return textureLod(u_tex3, clamped_uv, 0.0);
    if (id == 4) return textureLod(u_tex4, clamped_uv, 0.0);
    if (id == 5) return textureLod(u_tex5, clamped_uv, 0.0);
    if (id == 6) return textureLod(u_tex6, clamped_uv, 0.0);
    if (id == 7) return textureLod(u_tex7, clamped_uv, 0.0);
    return vec4(1.0);
}

struct ScreenTri {
    vec4 p0;
    vec4 p1;
    vec4 p2;
    vec4 uv0;
    vec4 uv1;
    vec4 uv2;
    vec4 bb;
    vec4 color;
    int tex_id;
    int pad0, pad1, pad2;
};

layout(std430, binding = 0) readonly buffer TriBuffer {
    ScreenTri triangles[];
};

uniform int u_num_tris;
uniform int u_width;
uniform int u_height;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= u_width || pixel.y >= u_height) return;

    vec2 p = vec2(pixel) + 0.5;
    float best_depth = 1.0;
    vec4 best_color = vec4(0.0, 0.0, 0.0, 1.0);

    for (int i = 0; i < u_num_tris; ++i) {
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
            if (depth < best_depth) {
                best_depth = depth;
                if (tri.tex_id >= 0) {
                    float interp_inv_w = b0 * tri.p0.w + b1 * tri.p1.w + b2 * tri.p2.w;
                    float u = (b0 * tri.uv0.x * tri.p0.w + b1 * tri.uv1.x * tri.p1.w + b2 * tri.uv2.x * tri.p2.w) / interp_inv_w;
                    float v = (b0 * tri.uv0.y * tri.p0.w + b1 * tri.uv1.y * tri.p1.w + b2 * tri.uv2.y * tri.p2.w) / interp_inv_w;
                    best_color = sample_mesh_tex(tri.tex_id, vec2(u, v));
                } else {
                    best_color = tri.color;
                }
            }
        }
    }

    imageStore(u_color_img, pixel, best_color);
}
"""

_GPU_DTYPE = np.dtype([
    ('p0', 'f4', 4), ('p1', 'f4', 4), ('p2', 'f4', 4),
    ('uv0', 'f4', 4), ('uv1', 'f4', 4), ('uv2', 'f4', 4),
    ('bb', 'f4', 4), ('color', 'f4', 4),
    ('tex_id', 'i4'), ('pad0', 'i4'), ('pad1', 'i4'), ('pad2', 'i4')
])


def _init_gpu_compute(width, height):
    global _gpu_ctx, _gpu_compute_shader, _gpu_color_tex, _gpu_readback_buf, _gpu_w, _gpu_h
    if _gpu_ctx is None:
        try:
            import moderngl
            _gpu_ctx = moderngl.create_context(standalone=True, require=430)
        except Exception as e:
            print(f"[Renderer] ModernGL GPU compute init failed: {e}. Falling back to CPU rasterizer.")
            return False

    if _gpu_w != width or _gpu_h != height:
        _gpu_w, _gpu_h = width, height
        _gpu_color_tex = _gpu_ctx.texture((width, height), 4, dtype='f1')
        _gpu_color_tex.bind_to_image(0, read=False, write=True)
        _gpu_readback_buf = bytearray(width * height * 4)

    if _gpu_compute_shader is None:
        try:
            _gpu_compute_shader = _gpu_ctx.compute_shader(_GPU_COMPUTE_SHADER_SRC)
        except Exception as e:
            print(f"[Renderer] Compute shader compilation failed: {e}. Falling back to CPU rasterizer.")
            return False

    return True


def _get_gpu_texture(texture):
    global _gpu_ctx, _gpu_textures
    if _gpu_ctx is None or texture is None:
        return None
    tex_id = id(texture)
    if tex_id not in _gpu_textures:
        import moderngl
        th, tw = texture.shape[:2]
        tex_rgb = np.ascontiguousarray(texture[:, :, ::-1])
        gl_tex = _gpu_ctx.texture((tw, th), 3, tex_rgb.tobytes())
        gl_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        gl_tex.build_mipmaps()
        _gpu_textures[tex_id] = gl_tex
    return _gpu_textures[tex_id]


def _render_scene_gpu(pygame_surface, meshes, positions, camera_pos, camera_angles,
                      rotations=None, fov_degrees=60.0, near=0.1, far=100.0):
    global _gpu_ctx, _gpu_compute_shader, _gpu_color_tex, _gpu_readback_buf, _gpu_ssbo, _gpu_ssbo_cap

    sw = pygame_surface.get_width()
    sh = pygame_surface.get_height()
    aspect = sh / sw

    eye, target, up = camera_pose_from_angles(camera_pos, camera_angles)
    mat_vp = build_vp_matrix(eye, target, up, fov_degrees, aspect, near, far)
    eye_np = np.asarray(eye, dtype=np.float32)

    screen_tris = []
    bound_textures = []
    texture_slot_map = {}

    for name, mesh in meshes.items():
        pos = np.array(positions.get(name, [0.0, 0.0, 0.0]), dtype=np.float32)
        rot = rotations.get(name, [0.0, 0.0, 0.0]) if rotations else [0.0, 0.0, 0.0]
        R = rotation_matrix(rot).astype(np.float32)

        verts = mesh.get("verts")
        if verts is None:
            verts = np.array(mesh["tris"], dtype=np.float32)
        uv_arr = mesh.get("uv_arr")
        texture = mesh.get("texture")
        colors = mesh.get("colors")

        tex_id = -1
        if texture is not None:
            raw_id = id(texture)
            if raw_id not in texture_slot_map:
                gl_t = _get_gpu_texture(texture)
                if gl_t is not None and len(bound_textures) < 8:
                    slot = len(bound_textures)
                    texture_slot_map[raw_id] = slot
                    bound_textures.append(gl_t)
            tex_id = texture_slot_map.get(raw_id, -1)

        N = verts.shape[0]
        flat = verts.reshape(-1, 3)
        world = flat @ R.T + pos

        ones = np.ones((N * 3, 1), dtype=np.float32)
        hom = np.concatenate([world, ones], axis=1)
        clip = (mat_vp @ hom.T).T.reshape(N, 3, 4)
        world = world.reshape(N, 3, 3)

        edge0 = world[:, 1] - world[:, 0]
        edge1 = world[:, 2] - world[:, 0]
        normals = np.cross(edge0, edge1)
        to_cam = eye_np - world[:, 0]
        facing = (normals * to_cam).sum(axis=1) > 0

        for idx in range(N):
            if not facing[idx]:
                continue

            cv = clip[idx]
            uv = uv_arr[idx] if uv_arr is not None else None
            col = colors[idx] if (colors and idx < len(colors)) else [200, 200, 200]
            col_f = [col[0] / 255.0, col[1] / 255.0, col[2] / 255.0, 1.0]

            if uv is not None and tex_id >= 0:
                c_verts = [np.append(cv[j], uv[j]) for j in range(3)]
                clipped = _clip_near_plane(c_verts, near)
                if len(clipped) < 3:
                    continue
                for k in range(1, len(clipped) - 1):
                    pa, pb, pc = clipped[0], clipped[k], clipped[k + 1]
                    inv_wa = 1.0 / max(pa[3], 1e-6)
                    inv_wb = 1.0 / max(pb[3], 1e-6)
                    inv_wc = 1.0 / max(pc[3], 1e-6)

                    p0_x = (pa[0] * inv_wa + 1.0) * 0.5 * sw
                    p0_y = (1.0 - pa[1] * inv_wa) * 0.5 * sh
                    p1_x = (pb[0] * inv_wb + 1.0) * 0.5 * sw
                    p1_y = (1.0 - pb[1] * inv_wb) * 0.5 * sh
                    p2_x = (pc[0] * inv_wc + 1.0) * 0.5 * sw
                    p2_y = (1.0 - pc[1] * inv_wc) * 0.5 * sh

                    min_x = max(0.0, min(p0_x, p1_x, p2_x))
                    max_x = min(float(sw - 1), max(p0_x, p1_x, p2_x))
                    min_y = max(0.0, min(p0_y, p1_y, p2_y))
                    max_y = min(float(sh - 1), max(p0_y, p1_y, p2_y))
                    if min_x > max_x or min_y > max_y:
                        continue

                    denom = (p1_y - p2_y) * (p0_x - p2_x) + (p2_x - p1_x) * (p0_y - p2_y)
                    if abs(denom) < 1e-10:
                        continue

                    screen_tris.append((
                        [p0_x, p0_y, pa[2] * inv_wa, inv_wa],
                        [p1_x, p1_y, pb[2] * inv_wb, inv_wb],
                        [p2_x, p2_y, pc[2] * inv_wc, inv_wc],
                        [pa[4], pa[5], 0.0, 0.0],
                        [pb[4], pb[5], 0.0, 0.0],
                        [pc[4], pc[5], 0.0, 0.0],
                        [min_x, min_y, max_x, max_y],
                        col_f, tex_id, 0, 0, 0
                    ))
            else:
                c_verts = [cv[j] for j in range(3)]
                clipped = _clip_near_plane(c_verts, near)
                if len(clipped) < 3:
                    continue
                for k in range(1, len(clipped) - 1):
                    pa, pb, pc = clipped[0], clipped[k], clipped[k + 1]
                    inv_wa = 1.0 / max(pa[3], 1e-6)
                    inv_wb = 1.0 / max(pb[3], 1e-6)
                    inv_wc = 1.0 / max(pc[3], 1e-6)

                    p0_x = (pa[0] * inv_wa + 1.0) * 0.5 * sw
                    p0_y = (1.0 - pa[1] * inv_wa) * 0.5 * sh
                    p1_x = (pb[0] * inv_wb + 1.0) * 0.5 * sw
                    p1_y = (1.0 - pb[1] * inv_wb) * 0.5 * sh
                    p2_x = (pc[0] * inv_wc + 1.0) * 0.5 * sw
                    p2_y = (1.0 - pc[1] * inv_wc) * 0.5 * sh

                    min_x = max(0.0, min(p0_x, p1_x, p2_x))
                    max_x = min(float(sw - 1), max(p0_x, p1_x, p2_x))
                    min_y = max(0.0, min(p0_y, p1_y, p2_y))
                    max_y = min(float(sh - 1), max(p0_y, p1_y, p2_y))
                    if min_x > max_x or min_y > max_y:
                        continue

                    denom = (p1_y - p2_y) * (p0_x - p2_x) + (p2_x - p1_x) * (p0_y - p2_y)
                    if abs(denom) < 1e-10:
                        continue

                    screen_tris.append((
                        [p0_x, p0_y, pa[2] * inv_wa, inv_wa],
                        [p1_x, p1_y, pb[2] * inv_wb, inv_wb],
                        [p2_x, p2_y, pc[2] * inv_wc, inv_wc],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [min_x, min_y, max_x, max_y],
                        col_f, -1, 0, 0, 0
                    ))

    num_tris = len(screen_tris)
    if num_tris > 0:
        tri_bytes = np.array(screen_tris, dtype=_GPU_DTYPE).tobytes()
        if _gpu_ssbo is None or len(tri_bytes) > _gpu_ssbo_cap:
            _gpu_ssbo = _gpu_ctx.buffer(reserve=max(len(tri_bytes) * 2, 65536))
            _gpu_ssbo_cap = _gpu_ssbo.size
        _gpu_ssbo.write(tri_bytes)
        _gpu_ssbo.bind_to_storage_buffer(0)

    for idx, gl_t in enumerate(bound_textures):
        gl_t.use(location=idx)

    _gpu_compute_shader["u_num_tris"].value = num_tris
    _gpu_compute_shader["u_width"].value = sw
    _gpu_compute_shader["u_height"].value = sh
    _gpu_compute_shader.run((sw + 15) // 16, (sh + 15) // 16)

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

    aspect = sh / sw
    eye, target, up = camera_pose_from_angles(camera_pos, camera_angles)
    mat_vp   = build_vp_matrix(eye, target, up, fov_degrees, aspect, near, far)

    zbuf = np.full((sw, sh), np.inf, dtype=np.float32)
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
        clip = (mat_vp @ hom.T).T

        clip  = clip.reshape(N, 3, 4)
        world = world.reshape(N, 3, 3)

        eye_np  = np.asarray(eye, dtype=np.float32)
        edge0   = world[:, 1] - world[:, 0]
        edge1   = world[:, 2] - world[:, 0]
        normals = np.cross(edge0, edge1)
        to_cam  = eye_np - world[:, 0]
        facing  = (normals * to_cam).sum(axis=1) > 0

        clip_w   = np.maximum(clip[:, :, 3], 1e-6)
        ndc_z    = clip[:, :, 2] / clip_w
        centroid_ndc_z = ndc_z.mean(axis=1)

        for idx in range(N):
            if not facing[idx]:
                continue

            sort_z = float(centroid_ndc_z[idx])
            cv     = clip[idx]

            if texture is not None and uv_arr is not None:
                uv = uv_arr[idx]
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
    clip = (mat_vp @ hom.T).T

    for a, b in edges:
        wa, wb = clip[a, 3], clip[b, 3]
        if wa <= near or wb <= near:
            continue
        pa = to_pixel(clip[a], sw, sh)
        pb = to_pixel(clip[b], sw, sh)
        pygame.draw.line(pygame_surface, color,
                         (int(pa[0]), int(pa[1])),
                         (int(pb[0]), int(pb[1])), 1)
