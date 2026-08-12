import sys
import os
import struct
import numpy as np
import cv2

import renderer



def _parse_face_vertex(token):

    parts = token.split("/")
    vi = int(parts[0])
    ti = int(parts[1]) if len(parts) > 1 and parts[1] else None
    ni = int(parts[2]) if len(parts) > 2 and parts[2] else None
    return vi, ti, ni


def _load_mtl(mtl_path):


    materials = {}
    current   = None
    base_dir  = os.path.dirname(mtl_path)

    try:
        with open(mtl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                kw    = parts[0].lower()
                if kw == "newmtl":
                    current = parts[1]
                    materials[current] = {"map_Kd": None}
                elif kw == "map_kd" and current:
                    tex_file = " ".join(parts[1:])
                    tex_path = os.path.join(base_dir, tex_file)
                    materials[current]["map_Kd"] = tex_path if os.path.exists(tex_path) else None
    except FileNotFoundError:
        pass

    return materials


def parse_obj(obj_path):

    base_dir = os.path.dirname(os.path.abspath(obj_path))

    positions_raw = []
    uvs_raw       = []
    materials     = {}


    groups        = {}
    current_mat   = "__default__"
    groups[current_mat] = {"tris": [], "uvs": []}

    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            kw    = parts[0].lower()

            if kw == "v":
                positions_raw.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif kw == "vt":
                u = float(parts[1])
                v = float(parts[2]) if len(parts) > 2 else 0.0
                uvs_raw.append([u, v])

            elif kw == "mtllib":
                mtl_file = " ".join(parts[1:])
                mtl_path = os.path.join(base_dir, mtl_file)
                materials.update(_load_mtl(mtl_path))

            elif kw == "usemtl":
                current_mat = parts[1]
                if current_mat not in groups:
                    groups[current_mat] = {"tris": [], "uvs": []}

            elif kw == "f":
                face_verts = [_parse_face_vertex(t) for t in parts[1:]]

                for i in range(1, len(face_verts) - 1):
                    corners = [face_verts[0], face_verts[i], face_verts[i + 1]]
                    tri     = []
                    uv_tri  = []
                    for vi, ti, _ni in corners:

                        idx = vi - 1 if vi > 0 else len(positions_raw) + vi
                        tri.append(positions_raw[idx])
                        if ti is not None:
                            tidx = ti - 1 if ti > 0 else len(uvs_raw) + ti
                            uv_tri.append(uvs_raw[tidx])
                        else:
                            uv_tri.append([0.0, 0.0])
                    groups[current_mat]["tris"].append(tri)
                    groups[current_mat]["uvs"].append(uv_tri)


    meshes = []
    for mat_name, group in groups.items():
        if not group["tris"]:
            continue

        mat_info = materials.get(mat_name, {})
        tex_path = mat_info.get("map_Kd")
        texture  = None

        if tex_path and os.path.exists(tex_path):
            texture = cv2.imread(tex_path)
            if texture is None:
                print(f"Warning: could not load texture {tex_path}")

        mesh = {
            "name":    mat_name,
            "tris":    group["tris"],
            "colors":  [[180, 180, 180]] * len(group["tris"]) if texture is None else None,
            "texture": texture,
            "uvs":     group["uvs"] if texture is not None else None,
        }
        meshes.append(mesh)

    return meshes


def main():
    if len(sys.argv) < 2:
        print("Usage: python translate.py input.obj [output.qwzx]")
        sys.exit(1)

    obj_path = sys.argv[1]
    if not os.path.exists(obj_path):
        print(f"Error: {obj_path} not found")
        sys.exit(1)

    if len(sys.argv) >= 3:
        qwzx_path = sys.argv[2]
    else:
        qwzx_path = os.path.splitext(obj_path)[0] + ".qwzx"

    print(f"Parsing {obj_path} ...")
    meshes = parse_obj(obj_path)

    if not meshes:
        print("No geometry found in the .obj file.")
        sys.exit(1)

    print(f"Writing {qwzx_path} ({len(meshes)} mesh(es)) ...")
    renderer.save_qwzx(qwzx_path, meshes)
    print("Done.")


if __name__ == "__main__":
    main()
