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
    if _USE_GPU_COMPUTE and (_gpu_ctx is not None or _init_gpu_compute(640, 640)):
        _ensure_gpu_mesh(mesh)
        if mesh.get("texture") is not None and mesh["uv_arr"] is not None:
            _ensure_gpu_texture(mesh["texture"])
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


# Stage 1 Compute Shader: 3D-to-2D Vertex Transform, Backface Culling,
# Near-Plane Clipping, UV Interpolation, and Screen-Space Projection
_GPU_GEOM_SHADER_SRC = """#version 430
layout(local_size_x = 64) in;

struct RawTri {
    vec4 v0;    // v0.xyz, uv0.x
    vec4 v1;    // v1.xyz, uv0.y
    vec4 v2;    // v2.xyz, uv1.x
    vec4 uvs;   // uv1.y, uv2.x, uv2.y, 0.0
    vec4 color; // r, g, b, 1.0
};

struct InstanceData {
    vec4 r0;    // R[0,0], R[0,1], R[0,2], pos.x
    vec4 r1;    // R[1,0], R[1,1], R[1,2], pos.y
    vec4 r2;    // R[2,0], R[2,1], R[2,2], pos.z
    ivec4 meta; // raw_tri_start, raw_tri_count, tex_id, cum_tri_start
};

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

    vec3 w0 = transform_world(inst, raw.v0.xyz);
    vec3 w1 = transform_world(inst, raw.v1.xyz);
    vec3 w2 = transform_world(inst, raw.v2.xyz);

    vec3 normal = cross(w1 - w0, w2 - w0);
    if (dot(normal, u_eye - w0) <= 0.0) return;

    vec4 in_c[3];
    in_c[0] = transform_clip(w0);
    in_c[1] = transform_clip(w1);
    in_c[2] = transform_clip(w2);

    vec2 in_uv[3];
    in_uv[0] = vec2(raw.v0.w, raw.v1.w);
    in_uv[1] = vec2(raw.v2.w, raw.uvs.x);
    in_uv[2] = vec2(raw.uvs.y, raw.uvs.z);

    vec4 clip_c[4];
    vec2 clip_uv[4];
    int num_clipped = 0;

    for (int i = 0; i < 3; ++i) {
        int next = (i + 1) % 3;
        vec4 a_c = in_c[i];
        vec2 a_uv = in_uv[i];
        vec4 b_c = in_c[next];
        vec2 b_uv = in_uv[next];

        bool a_in = a_c.w > u_near;
        bool b_in = b_c.w > u_near;

        if (a_in) {
            clip_c[num_clipped] = a_c;
            clip_uv[num_clipped] = a_uv;
            num_clipped++;
        }
        if (a_in != b_in) {
            float t = (a_c.w - u_near) / (a_c.w - b_c.w);
            clip_c[num_clipped] = mix(a_c, b_c, t);
            clip_uv[num_clipped] = mix(a_uv, b_uv, t);
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
        st.uv0 = vec4(uva, 0.0, 0.0);
        st.uv1 = vec4(uvb, 0.0, 0.0);
        st.uv2 = vec4(uvc, 0.0, 0.0);
        st.bb = vec4(min_x, min_y, max_x, max_y);
        st.color = raw.color;
        st.tex_id = tex_id;
        st.pad0 = 0;
        st.pad1 = 0;
        st.pad2 = 0;
        triangles[out_idx] = st;
    }
}
"""

# Stage 2 Compute Shader: Per-Pixel Rasterization, Depth Buffering,
# Perspective-Correct UV Interpolation, and GPU Texture Mapping
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

vec4 sample_ssbo_tex(int id, vec2 clamped_uv) {
    ivec4 meta = tex_meta[id];
    int offset = meta.x;
    int tw = meta.y;
    int th = meta.z;
    if (tw <= 0 || th <= 0) return vec4(1.0);

    float fx = clamped_uv.x * float(tw - 1);
    float fy = clamped_uv.y * float(th - 1);
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
    vec2 clamped_uv = clamp(uv, 0.0, 1.0);
    if (id == 0) return textureLod(u_tex0, clamped_uv, 0.0);
    if (id == 1) return textureLod(u_tex1, clamped_uv, 0.0);
    if (id == 2) return textureLod(u_tex2, clamped_uv, 0.0);
    if (id == 3) return textureLod(u_tex3, clamped_uv, 0.0);
    if (id == 4) return textureLod(u_tex4, clamped_uv, 0.0);
    if (id == 5) return textureLod(u_tex5, clamped_uv, 0.0);
    if (id == 6) return textureLod(u_tex6, clamped_uv, 0.0);
    if (id == 7) return textureLod(u_tex7, clamped_uv, 0.0);
    if (id >= 8) return sample_ssbo_tex(id, clamped_uv);
    return vec4(1.0);
}

uniform int u_width;
uniform int u_height;
uniform int u_max_screen_tris;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= u_width || pixel.y >= u_height) return;

    vec2 p = vec2(pixel) + 0.5;
    float best_depth = 1e30;
    vec4 best_color = vec4(0.0, 0.0, 0.0, 1.0);

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
    cached = mesh.get("_gpu_tri_slice")
    if cached is not None and cached[0] is _gpu_ctx:
        return cached[1], cached[2]

    verts = mesh.get("verts")
    if verts is None:
        verts = np.array(mesh["tris"], dtype=np.float32)
        mesh["verts"] = verts
    uv_arr = mesh.get("uv_arr")
    if uv_arr is None and mesh.get("uvs") is not None:
        uv_arr = np.array(mesh["uvs"], dtype=np.float32)
        mesh["uv_arr"] = uv_arr

    colors = mesh.get("colors")
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
            chunk['color'][:, 3] = 1.0
        else:
            chunk['color'][:, :3] = 200.0 / 255.0
            chunk['color'][:, 3] = 1.0

        _gpu_raw_tri_chunks.append(chunk)
        _gpu_total_cached_tris += N

        all_raw_bytes = np.concatenate(_gpu_raw_tri_chunks).tobytes()
        if _gpu_raw_tri_ssbo is None or len(all_raw_bytes) > _gpu_raw_tri_cap:
            if _gpu_raw_tri_ssbo is not None:
                _gpu_raw_tri_ssbo.release()
            _gpu_raw_tri_ssbo = _gpu_ctx.buffer(reserve=max(len(all_raw_bytes) * 2, 65536))
            _gpu_raw_tri_cap = _gpu_raw_tri_ssbo.size
        _gpu_raw_tri_ssbo.write(all_raw_bytes)

    mesh["_gpu_tri_slice"] = (_gpu_ctx, start_tri, N)
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
    tex_rgb = np.ascontiguousarray(texture[:, :, ::-1], dtype=np.uint8)

    if slot < 8:
        gl_tex = _gpu_ctx.texture((tw, th), 3, tex_rgb.tobytes())
        gl_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        gl_tex.build_mipmaps()
        _gpu_textures[slot] = gl_tex

    # Also upload packed RGBA8 texels to GPU SSBO so >8 textures work seamlessly
    rgba = np.empty((th, tw, 4), dtype=np.uint8)
    rgba[:, :, :3] = tex_rgb
    rgba[:, :, 3] = 255
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
    aspect = sh / sw

    eye, target, up = camera_pose_from_angles(camera_pos, camera_angles)
    mat_vp = build_vp_matrix(eye, target, up, fov_degrees, aspect, near, far).astype(np.float32)
    eye_np = np.asarray(eye, dtype=np.float32)

    instance_rows = []
    total_input_tris = 0

    for name, mesh in meshes.items():
        tri_start, tri_count = _ensure_gpu_mesh(mesh)
        if tri_count <= 0:
            continue

        pos = positions.get(name, [0.0, 0.0, 0.0])
        rot = rotations.get(name, [0.0, 0.0, 0.0]) if rotations else [0.0, 0.0, 0.0]
        R = rotation_matrix(rot).astype(np.float32)

        texture = mesh.get("texture")
        has_uvs = (mesh.get("uv_arr") is not None) or (mesh.get("uvs") is not None)
        tex_slot = _ensure_gpu_texture(texture) if (texture is not None and has_uvs) else -1

        instance_rows.append((
            [R[0, 0], R[0, 1], R[0, 2], float(pos[0])],
            [R[1, 0], R[1, 1], R[1, 2], float(pos[1])],
            [R[2, 0], R[2, 1], R[2, 2], float(pos[2])],
            [tri_start, tri_count, tex_slot, total_input_tris]
        ))
        total_input_tris += tri_count

    num_instances = len(instance_rows)
    max_screen_tris = max(total_input_tris * 2, 1)
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

    _gpu_compute_shader["u_width"].value = sw
    _gpu_compute_shader["u_height"].value = sh
    _gpu_compute_shader["u_max_screen_tris"].value = max_screen_tris
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
