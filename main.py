import pygame
import numpy as np
import math
import json

S_WIDTH = 640
S_HEIGHT = 640
ASPECT_RATIO = S_HEIGHT / S_WIDTH

pygame.init()
screen = pygame.display.set_mode((S_WIDTH, S_HEIGHT))
clock = pygame.time.Clock()
running = True

data = []
obj_load = ["cube.json", "cylinder.json"]
for obj in obj_load:
    with open(obj, "r") as file:
        data.append(json.load(file))
print(data)

def compute_3d_vertex(v0, v1, v2,
                      eye=(0.0, 0.0, 5.0),
                      target=(0.0, 0.0, 0.0),
                      up=(0.0, 1.0, 0.0),
                      fov_degrees=60.0,
                      aspect_ratio= ASPECT_RATIO,
                      near=0.1,
                      far=100.0):

    vertices = [np.array(v0, dtype=float), np.array(v1, dtype=float), np.array(v2, dtype=float)]
    e = np.array(eye, dtype=float)
    t = np.array(target, dtype=float)
    u_up = np.array(up, dtype=float)

    fwd_vec = t - e
    fwd_norm = np.linalg.norm(fwd_vec)
    if fwd_norm < 1e-8:
        raise ValueError("eye and target must not coincide")
    f = fwd_vec / fwd_norm

    r = np.cross(f, u_up)
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-8:
        fallback_up = np.array([0.0, 0.0, 1.0]) if abs(f[1]) > 0.99 else np.array([0.0, 1.0, 0.0])
        r = np.cross(f, fallback_up)
        r_norm = np.linalg.norm(r)
    r = r / r_norm
    u = np.cross(r, f)

    mat_view = np.eye(4)
    mat_view[0, :3], mat_view[0, 3] = r, -np.dot(r, e)
    mat_view[1, :3], mat_view[1, 3] = u, -np.dot(u, e)
    mat_view[2, :3], mat_view[2, 3] = -f, np.dot(f, e)

    if not (0.0 < fov_degrees < 180.0):
        raise ValueError("fov_degrees must be between 0 and 180")
    if near <= 0.0 or far <= near:
        raise ValueError("require 0 < near < far")

    g = 1.0 / np.tan(np.radians(fov_degrees) / 2.0)
    mat_proj = np.zeros((4, 4))
    mat_proj[0, 0] = g / aspect_ratio
    mat_proj[1, 1] = g
    mat_proj[2, 2] = -(far + near) / (far - near)
    mat_proj[2, 3] = -(2.0 * far * near) / (far - near)
    mat_proj[3, 2] = -1.0

    mat_vp = np.dot(mat_proj, mat_view)

    screen_coordinates = []
    for vertex in vertices:
        v_4d = np.append(vertex, 1.0)
        clip_space = np.dot(mat_vp, v_4d)
        w_c = clip_space[3]

        if w_c <= near:
            screen_coordinates.append(np.array([np.nan, np.nan]))
            continue

        x_screen = clip_space[0] / w_c
        y_screen = clip_space[1] / w_c
        screen_coordinates.append(np.array([x_screen, y_screen]))

    return screen_coordinates[0], screen_coordinates[1], screen_coordinates[2]


def project_to_screen(v0, v1, v2, screen_width, screen_height, **kwargs):
    p0, p1, p2 = compute_3d_vertex(v0, v1, v2, **kwargs)
    pixel_points = []
    for p in (p0, p1, p2):
        if np.isnan(p).any():
            return None
        px = (p[0] + 1.0) * 0.5 * screen_width
        py = (1.0 - p[1]) * 0.5 * screen_height
        pixel_points.append((px, py))
    return pixel_points


def _camera_pose_from_angles(camera_pos, camera_angles):

    yaw, pitch, roll = (np.radians(a % 360.0) for a in camera_angles)

    cy, sy = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[cy, 0, sy],
                      [0, 1, 0],
                      [-sy, 0, cy]])

    cp, sp = np.cos(pitch), np.sin(pitch)
    R_pitch = np.array([[1, 0, 0],
                        [0, cp, -sp],
                        [0, sp, cp]])

    cr, sr = np.cos(roll), np.sin(roll)
    R_roll = np.array([[cr, -sr, 0],
                       [sr, cr, 0],
                       [0, 0, 1]])

    R = R_yaw @ R_pitch @ R_roll

    forward = R @ np.array([0.0, 0.0, -1.0])
    up = R @ np.array([0.0, 1.0, 0.0])

    eye = np.array(camera_pos, dtype=float)
    target = eye + forward
    return eye, target, up


def project_to_screen_camera(v0, v1, v2, screen_width, screen_height,
                             camera_pos=(0.0, 0.0, 5.0),
                             camera_angles=(0.0, 0.0, 0.0),
                             fov_degrees=60.0,
                             aspect_ratio=1.7778,
                             near=0.1,
                             far=100.0):
    eye, target, up = _camera_pose_from_angles(camera_pos, camera_angles)
    return project_to_screen(v0, v1, v2, screen_width, screen_height,
                             eye=eye, target=target, up=up,
                             fov_degrees=fov_degrees, aspect_ratio=aspect_ratio,
                             near=near, far=far)


camera_pos = [0, 0, 5.0]
yaw, pitch = 0, 0
vel = 0.1

my_font = pygame.font.SysFont(None, 48)


while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_w]:
        camera_pos[2] += math.cos(math.radians(yaw)) * vel
        camera_pos[0] += math.sin(math.radians(yaw)) * vel

    if keys[pygame.K_s]:
        camera_pos[2] -= math.cos(math.radians(yaw)) * vel
        camera_pos[0] -= math.sin(math.radians(yaw)) * vel

    if keys[pygame.K_a]:
        camera_pos[2] += math.cos(math.radians(yaw - 90)) * vel
        camera_pos[0] += math.sin(math.radians(yaw - 90)) * vel

    if keys[pygame.K_d]:
        camera_pos[2] -= math.cos(math.radians(yaw - 90)) * vel
        camera_pos[0] -= math.sin(math.radians(yaw - 90)) * vel

    mouse_x, mouse_y = pygame.mouse.get_pos()
    pygame.mouse.set_visible(False)

    yaw = (mouse_x / 640) * 360.0
    pitch = (mouse_y / 640) * -360.0

    text_surface = my_font.render(str(yaw), True, "white")

    screen.fill("black")

    screen.blit(text_surface, (50, 50))

    for obj in data:
        for i in data[0]:
            v0, v1, v2 = i

            points = project_to_screen_camera(np.array(v0), np.array(v1), np.array(v2),
                                          S_WIDTH, S_HEIGHT,
                                          camera_pos = camera_pos, camera_angles=(yaw, pitch, 0))

            if points is not None:
                pygame.draw.polygon(screen, (255, 255, 255), points)

    pygame.display.flip()

    clock.tick(60)

pygame.quit()