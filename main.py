import pygame
import math
import json
import os

import renderer

S_WIDTH  = 640
S_HEIGHT = 640

pygame.init()
screen = pygame.display.set_mode((S_WIDTH, S_HEIGHT))
pygame.display.set_caption("3D Game")
clock = pygame.time.Clock()

pygame.event.set_grab(True)
pygame.mouse.set_visible(False)


def _json_to_mesh(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    name = os.path.splitext(os.path.basename(json_path))[0]
    return {"name": name, "tris": data[0], "colors": data[1],
            "texture": None, "uvs": None}


def load_or_migrate(name):
    qwzx_path = f"{name}.qwzx"
    json_path  = f"{name}.json"
    if os.path.exists(qwzx_path):
        return renderer.load_qwzx(qwzx_path)[0]
    if os.path.exists(json_path):
        mesh = _json_to_mesh(json_path)
        renderer.save_qwzx(qwzx_path, [mesh])
        return mesh
    raise FileNotFoundError(f"Neither {qwzx_path} nor {json_path} found")


meshes = {
    "cube":         renderer.prepare_mesh(load_or_migrate("cube")),
    "cylinder":     renderer.prepare_mesh(load_or_migrate("cylinder")),
    "concrete_box": renderer.prepare_mesh(load_or_migrate("concrete_box")),
}

positions = {
    "cube":         [0.0,  0.0,  0.0],
    "cylinder":     [2.0,  0.0,  0.0],
    "concrete_box": [0.0,  0.0, -8.0],
}

# yaw, pitch, roll in degrees for each object
rotations = {
    "cube":         [0.0, 0.0, 0.0],
    "cylinder":     [0.0, 0.0, 0.0],
    "concrete_box": [0.0, 0.0, 0.0],
}

camera_pos = [0.0, 0.0, 5.0]
yaw   = 0.0
pitch = 0.0
vel   = 0.5

my_font = pygame.font.SysFont(None, 48)
running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    dx, dy      = pygame.mouse.get_rel()
    sensitivity = 0.15
    yaw   = (yaw - dx * sensitivity) % 360.0
    pitch = max(-89.0, min(89.0, pitch - dy * sensitivity))

    rad   = math.radians(yaw)
    fwd_x = -math.sin(rad)
    fwd_z = -math.cos(rad)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        camera_pos[0] += fwd_x * vel
        camera_pos[2] += fwd_z * vel
    if keys[pygame.K_s]:
        camera_pos[0] -= fwd_x * vel
        camera_pos[2] -= fwd_z * vel
    if keys[pygame.K_a]:
        camera_pos[0] += fwd_z * vel
        camera_pos[2] -= fwd_x * vel
    if keys[pygame.K_d]:
        camera_pos[0] -= fwd_z * vel
        camera_pos[2] += fwd_x * vel
    if keys[pygame.K_SPACE]:
        camera_pos[1] += vel
    if keys[pygame.K_LSHIFT]:
        camera_pos[1] -= vel

    screen.fill("black")
    screen.blit(my_font.render(f"yaw={yaw:.1f}", True, "white"), (50, 50))

    renderer.render_scene(screen, meshes, positions,
                          camera_pos, (yaw, pitch, 0),
                          rotations=rotations)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
