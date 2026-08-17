import pygame
import math
import json
import os

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

# Warp mouse to window center so get_rel never hits the border.
# A _skip_warp flag discards the one synthetic MOUSEMOTION pygame fires
# as a result of the warp itself, preventing a corrupt delta.
_CX, _CY   = S_WIDTH // 2, S_HEIGHT // 2
_skip_warp = False
pygame.mouse.set_pos(_CX, _CY)


# ── mesh loading ──────────────────────────────────────────────────────────────

def _json_to_mesh(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    name = os.path.splitext(os.path.basename(json_path))[0]
    return {"name": name, "tris": data[0], "colors": data[1],
            "texture": None, "uvs": None}


def load_or_migrate(name):
    qwzx_path = os.path.join(ROOT, f"{name}.qwzx")
    json_path  = os.path.join(ROOT, f"{name}.json")
    if os.path.exists(qwzx_path):
        return renderer.load_qwzx(qwzx_path)[0]
    if os.path.exists(json_path):
        mesh = _json_to_mesh(json_path)
        renderer.save_qwzx(qwzx_path, [mesh])
        return mesh
    raise FileNotFoundError(f"Neither {qwzx_path} nor {json_path} found")


meshes = {}
for _name in ("cube", "cylinder", "concrete_box"):
    try:
        meshes[_name] = renderer.prepare_mesh(load_or_migrate(_name))
    except FileNotFoundError:
        pass

positions = {
    "cube":         [0.0,  0.0,  0.0],
    "cylinder":     [2.0,  0.0,  0.0],
    "concrete_box": [0.0,  0.0, -8.0],
}

rotations = {
    "cube":         [0.0, 0.0, 0.0],
    "cylinder":     [0.0, 0.0, 0.0],
    "concrete_box": [0.0, 0.0, 0.0],
}


# ── physics setup ─────────────────────────────────────────────────────────────

# Infinite flat ground plane — fast AABB path
physics.add_static("ground", [-500, -0.2, -500], [500, 0.0, 500])

# Register each mesh for triangle-normal collision
for name, mesh in meshes.items():
    pos = positions.get(name, [0.0, 0.0, 0.0])
    rot = rotations.get(name, [0.0, 0.0, 0.0])
    physics.add_mesh_static(name, mesh, pos, rot)

# Player start position
physics.player["pos"][:] = [0.0, 2.0, 5.0]

# Pre-compute mesh AABBs for the debug hitbox overlay (static, computed once)
import numpy as np
_mesh_aabbs = {}
for name, mesh in meshes.items():
    verts = mesh["verts"].reshape(-1, 3).astype(np.float32)
    wpos  = verts + np.array(positions[name], dtype=np.float32)
    _mesh_aabbs[name] = (wpos.min(axis=0), wpos.max(axis=0))


# ── HUD ───────────────────────────────────────────────────────────────────────

my_font  = pygame.font.SysFont("consolas", 14)
hint_col = (100, 100, 100)
hud_col  = (200, 200, 200)
flag_col = (255, 220, 60)


# ── debug / mode flags ────────────────────────────────────────────────────────

debug_hitboxes = False   # key 4 — draw AABB wireframes
no_collision   = False   # key 5 — noclip
fly_mode       = False   # key 6 — fly


# ── camera look ───────────────────────────────────────────────────────────────

yaw       = 0.0
pitch     = 0.0
SENS      = 0.15


running = True

while running:
    dt = clock.tick(60) / 1000.0


    mouse_dx = 0
    mouse_dy = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEMOTION:
            if _skip_warp:

                _skip_warp = False
                continue
            mouse_dx += event.rel[0]
            mouse_dy += event.rel[1]

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_4:
                debug_hitboxes = not debug_hitboxes
            elif event.key == pygame.K_5:
                no_collision              = not no_collision
                physics.player["no_clip"] = no_collision
            elif event.key == pygame.K_6:
                fly_mode                     = not fly_mode
                physics.player["fly_mode"]   = fly_mode
                physics.player["vel"][1]     = 0.0


    pygame.mouse.set_pos(_CX, _CY)
    _skip_warp = True


    yaw   = (yaw   - mouse_dx * SENS) % 360.0
    pitch = max(-89.0, min(89.0, pitch - mouse_dy * SENS))


    rad   = math.radians(yaw)
    fwd_x = -math.sin(rad)
    fwd_z = -math.cos(rad)
    str_x =  fwd_z
    str_z = -fwd_x

    keys  = pygame.key.get_pressed()
    wx    = 0.0
    wz    = 0.0
    if keys[pygame.K_w]:     wx += fwd_x; wz += fwd_z
    if keys[pygame.K_s]:     wx -= fwd_x; wz -= fwd_z
    if keys[pygame.K_a]:     wx += str_x; wz += str_z
    if keys[pygame.K_d]:     wx -= str_x; wz -= str_z

    jump = bool(keys[pygame.K_SPACE])


    if fly_mode:
        if   keys[pygame.K_SPACE]:   physics.player["vel"][1] =  physics.FLY_SPEED
        elif keys[pygame.K_LSHIFT]:  physics.player["vel"][1] = -physics.FLY_SPEED
        else:                        physics.player["vel"][1] =  0.0


    physics.move_player(wx, wz, jump, dt)


    camera_pos    = physics.player_camera_pos()
    camera_angles = (yaw, pitch, 0)

    screen.fill((0, 0, 0))

    renderer.render_scene(screen, meshes, positions,
                          camera_pos, camera_angles,
                          rotations=rotations)


    if debug_hitboxes:
        pmn, pmx = physics.player_aabb()
        renderer.draw_aabb_debug(screen, pmn, pmx,
                                 camera_pos, camera_angles,
                                 color=(0, 255, 0))
        for name, (mn, mx) in _mesh_aabbs.items():
            renderer.draw_aabb_debug(screen, mn, mx,
                                     camera_pos, camera_angles,
                                     color=(64, 128, 255))


    flags = []
    if debug_hitboxes: flags.append("HITBOX")
    if no_collision:   flags.append("NOCLIP")
    if fly_mode:       flags.append("FLY")

    p = physics.player["pos"]
    line1 = (f"yaw={yaw:6.1f}  pitch={pitch:6.1f}"
             f"  pos=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})")
    line2 = "  ".join(flags) if flags else ""
    line3 = "4=hitbox  5=noclip  6=fly"

    screen.blit(my_font.render(line1, True, hud_col),  (6, 6))
    if line2:
        screen.blit(my_font.render(line2, True, flag_col), (6, 22))
    screen.blit(my_font.render(line3, True, hint_col), (6, 38))

    pygame.display.flip()

pygame.quit()
