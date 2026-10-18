import pygame
import math
import json
import os
import numpy as np

import renderer
import physics

S_WIDTH  = 640
S_HEIGHT = 640
ROOT     = os.path.dirname(os.path.abspath(__file__))

pygame.init()
screen = pygame.display.set_mode((S_WIDTH, S_HEIGHT))
pygame.display.set_caption("3D Game")
clock = pygame.time.Clock()

pygame.event.set_grab(True)
pygame.mouse.set_visible(False)

_CX, _CY   = S_WIDTH // 2, S_HEIGHT // 2
_skip_warp = False
pygame.mouse.set_pos(_CX, _CY)


def _json_to_mesh(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    name = os.path.splitext(os.path.basename(json_path))[0]
    return {"name": name, "tris": data[0], "colors": data[1],
            "texture": None, "uvs": None}


ASSETS_DIR  = os.path.join(ROOT, "assets")
PHYSICS_DIR = os.path.join(ROOT, "physics")


def load_or_migrate(name):
    qwzx_path = os.path.join(ASSETS_DIR, f"{name}.qwzx")
    if not os.path.exists(qwzx_path):
        root_qwzx = os.path.join(ROOT, f"{name}.qwzx")
        if os.path.exists(root_qwzx):
            qwzx_path = root_qwzx

    json_path = os.path.join(ASSETS_DIR, f"{name}.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(ROOT, f"{name}.json")

    if os.path.exists(qwzx_path):
        loaded = renderer.load_qwzx(qwzx_path)
        return renderer.combine_submeshes(loaded, name=name)
    if os.path.exists(json_path):
        mesh = _json_to_mesh(json_path)
        os.makedirs(ASSETS_DIR, exist_ok=True)
        save_path = os.path.join(ASSETS_DIR, f"{name}.qwzx")
        renderer.save_qwzx(save_path, [mesh])
        return renderer.prepare_mesh(mesh)
    raise FileNotFoundError(f"Neither {qwzx_path} nor {json_path} found")


def _clone_with_alpha(mesh_dict, alpha):
    subs = renderer.get_submeshes(mesh_dict)
    cloned_subs = []
    for sm in subs:
        sm_copy = dict(sm)
        sm_copy.pop("_gpu_tri_slice", None)
        sm_copy["alpha"] = float(sm.get("alpha", 1.0)) * float(alpha)
        cloned_subs.append(sm_copy)
    return renderer.combine_submeshes(cloned_subs, name=mesh_dict.get("name"))


def load_map(path):
    with open(path) as f:
        data = json.load(f)

    base_meshes  = {}
    meshes       = {}
    positions    = {}
    rotations    = {}
    objects_meta = {}

    for i, obj in enumerate(data["objects"]):
        mesh_name = obj["mesh"]
        key       = f"{mesh_name}_{i}"

        if mesh_name not in base_meshes:
            base_meshes[mesh_name] = renderer.prepare_mesh(load_or_migrate(mesh_name))

        obj_mesh = base_meshes[mesh_name]
        if "alpha" in obj:
            obj_mesh = _clone_with_alpha(obj_mesh, obj["alpha"])

        meshes[key]       = obj_mesh
        positions[key]    = obj["pos"]
        rotations[key]    = obj.get("rot", [0, 0, 0])
        objects_meta[key] = obj

    return meshes, positions, rotations, data["player_start"], objects_meta


meshes, positions, rotations, player_start, objects_meta = load_map(os.path.join(ROOT, "map.json"))

for key, mesh in meshes.items():
    obj_info = objects_meta.get(key, {})
    is_rigid = obj_info.get("rigid_body", False)
    if is_rigid:
        physics_file = (
            obj_info.get("physics_file") or
            obj_info.get("physics") or
            obj_info.get("physics_json") or
            f"{obj_info.get('mesh')}.physics.json"
        )
        physics.add_rigid_body(key, mesh, positions[key], rotations[key], physics_file=physics_file, root=ROOT)
    else:
        physics.add_mesh_static(key, mesh, positions[key], rotations[key])

physics.player["pos"][:] = player_start

_mesh_aabbs = {}
for key, mesh in meshes.items():
    verts = mesh["verts"].reshape(-1, 3).astype(np.float32)
    R     = renderer.rotation_matrix(rotations.get(key, [0, 0, 0])).astype(np.float32)
    wpos  = verts @ R.T + np.array(positions[key], dtype=np.float32)
    _mesh_aabbs[key] = (wpos.min(axis=0), wpos.max(axis=0))


my_font  = pygame.font.SysFont("consolas", 14)
hint_col = (100, 100, 100)
hud_col  = (200, 200, 200)
flag_col = (255, 220, 60)

debug_hitboxes = False
no_collision   = False
fly_mode       = False
gpu_mode       = True
shader_mode    = renderer.get_shader_mode()

yaw   = 0.0
pitch = 0.0
SENS  = 0.15

held_body = None
hold_distance = 2.2

running = True

while running:
    dt = clock.tick(60) / 1000.0

    mouse_dx = 0
    mouse_dy = 0

    camera_pos    = physics.player_camera_pos()
    camera_angles = (yaw, pitch, 0)
    cam_eye, cam_target, _ = renderer.camera_pose_from_angles(camera_pos, camera_angles)
    cam_fwd = cam_target - cam_eye
    fwd_norm = np.linalg.norm(cam_fwd)
    if fwd_norm > 1e-6:
        cam_fwd = cam_fwd / fwd_norm

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEMOTION:
            if _skip_warp:
                _skip_warp = False
                continue
            mouse_dx += event.rel[0]
            mouse_dy += event.rel[1]

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if held_body is not None:
                    physics.throw_held_object(held_body, cam_fwd, physics.player["vel"], throw_speed=18.0)
                    held_body = None
                else:
                    for body in physics.rigid_bodies.values():
                        dist = np.linalg.norm(body.get_center() - np.array(camera_pos))
                        if dist < 12.0:
                            body.velocity += cam_fwd * 6.0
                            body.angular_velocity += np.array([0.5, 1.2, -0.3])

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_e:
                if held_body is not None:
                    physics.drop_held_object(held_body, physics.player["vel"])
                    held_body = None
                else:
                    target = physics.find_targeted_body(cam_eye, cam_fwd, max_dist=4.5)
                    if target is not None:
                        held_body = target
                        hold_distance = float(np.clip(np.linalg.norm(target.get_center() - cam_eye), 1.8, 2.6))
            elif event.key == pygame.K_4:
                debug_hitboxes = not debug_hitboxes
            elif event.key == pygame.K_5:
                no_collision              = not no_collision
                physics.player["no_clip"] = no_collision
            elif event.key == pygame.K_6:
                fly_mode                   = not fly_mode
                physics.player["fly_mode"] = fly_mode
                physics.player["vel"][1]   = 0.0
            elif event.key == pygame.K_7:
                gpu_mode = not gpu_mode
                renderer.set_gpu_acceleration(gpu_mode)
            elif event.key == pygame.K_8:
                shader_mode = renderer.cycle_shader_mode()

    pygame.mouse.set_pos(_CX, _CY)
    _skip_warp = True

    yaw   = (yaw   - mouse_dx * SENS) % 360.0
    pitch = max(-89.0, min(89.0, pitch - mouse_dy * SENS))

    rad   = math.radians(yaw)
    fwd_x = -math.sin(rad)
    fwd_z = -math.cos(rad)
    str_x =  fwd_z
    str_z = -fwd_x

    keys = pygame.key.get_pressed()
    wx   = 0.0
    wz   = 0.0
    if keys[pygame.K_w]: wx += fwd_x; wz += fwd_z
    if keys[pygame.K_s]: wx -= fwd_x; wz -= fwd_z
    if keys[pygame.K_a]: wx += str_x; wz += str_z
    if keys[pygame.K_d]: wx -= str_x; wz -= str_z

    jump = bool(keys[pygame.K_SPACE])

    if fly_mode:
        if   keys[pygame.K_SPACE]:  physics.player["vel"][1] =  physics.FLY_SPEED
        elif keys[pygame.K_LSHIFT]: physics.player["vel"][1] = -physics.FLY_SPEED
        else:                       physics.player["vel"][1] =  0.0

    if held_body is not None:
        target_pos = cam_eye + cam_fwd * hold_distance
        if not physics.update_held_object(held_body, target_pos, dt):
            held_body = None

    physics.move_player(wx, wz, jump, dt)
    physics.update_rigid_bodies(dt)

    for key, body in physics.rigid_bodies.items():
        positions[key] = body.position.tolist()
        rotations[key] = body.get_rotation_euler()

    camera_pos    = physics.player_camera_pos()
    camera_angles = (yaw, pitch, 0)

    screen.fill((0, 0, 0))

    renderer.render_scene(screen, meshes, positions,
                          camera_pos, camera_angles,
                          rotations=rotations)

    # Crosshair
    ch_col = (180, 180, 180)
    pygame.draw.line(screen, ch_col, (_CX - 4, _CY), (_CX + 4, _CY), 1)
    pygame.draw.line(screen, ch_col, (_CX, _CY - 4), (_CX, _CY + 4), 1)

    # Pickup interaction HUD prompt
    aimed_body = None if held_body is not None else physics.find_targeted_body(cam_eye, cam_fwd, max_dist=4.5)
    if held_body is not None:
        prompt_surf = my_font.render("[E] Drop   [LMB] Throw", True, flag_col)
        screen.blit(prompt_surf, (_CX - prompt_surf.get_width() // 2, _CY + 20))
    elif aimed_body is not None:
        prompt_surf = my_font.render("[E] Pick up", True, flag_col)
        screen.blit(prompt_surf, (_CX - prompt_surf.get_width() // 2, _CY + 20))

    if debug_hitboxes:
        pmn, pmx = physics.player_aabb()
        renderer.draw_aabb_debug(screen, pmn, pmx,
                                 camera_pos, camera_angles,
                                 color=(0, 255, 0))
        for key, mesh in meshes.items():
            if key in physics.rigid_bodies:
                verts = np.array(physics.rigid_bodies[key].get_vertices())
                renderer.draw_aabb_debug(screen, verts.min(axis=0), verts.max(axis=0),
                                         camera_pos, camera_angles,
                                         color=(255, 160, 60))
            elif key in _mesh_aabbs:
                mn, mx = _mesh_aabbs[key]
                renderer.draw_aabb_debug(screen, mn, mx,
                                         camera_pos, camera_angles,
                                         color=(64, 128, 255))
            renderer.draw_normals_debug(screen, mesh,
                                        positions[key], rotations.get(key, [0, 0, 0]),
                                        camera_pos, camera_angles)


    flags = [
        f"RENDERER: {'GPU (RX 7900 XTX)' if gpu_mode else 'CPU (Software)'}",
        f"SHADER: {shader_mode.upper()}"
    ]
    if debug_hitboxes: flags.append("HITBOX")
    if no_collision:   flags.append("NOCLIP")
    if fly_mode:       flags.append("FLY")

    p = physics.player["pos"]
    line1 = (f"yaw={yaw:6.1f}  pitch={pitch:6.1f}"
             f"  pos=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})")
    line2 = "  ".join(flags) if flags else ""
    line3 = "E=pickup/drop  LMB=throw  4=hitbox  5=noclip  6=fly  7=GPU/CPU  8=shader"
    line4 = str(clock)

    screen.blit(my_font.render(line1, True, hud_col),  (6, 6))
    if line2:
        screen.blit(my_font.render(line2, True, flag_col), (6, 22))
    screen.blit(my_font.render(line3, True, hint_col), (6, 38))
    screen.blit(my_font.render(line4, True, hint_col), (6, 62))

    pygame.display.flip()

pygame.quit()
