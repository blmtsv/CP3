# 4 вариант
import numpy as np
from vispy import app, scene
import imageio
from vispy.geometry import Rect
from vispy.scene.visuals import Text
from funcs import init_boids, directions, propagate, flocking, wall_avoidance, periodic_walls
app.use_app('pyglet')


# variables
w, h = 1280, 960     # window parameters
N = 5000         # boids number
dt = 0.1         # time step
asp = w / h      # aspect ratio
perception = 1/20   # radius
better_walls_w = 0.05    # wall parameter
vrange = (0, 0.1)   # tuple of speed limitations
arange = (0, 0.05)  # tuple of acceleration limitations
ang = -0.5            # cos of angle visability
t = 60*30   # count of frames
alpha = 2*np.arccos(ang)   #  angle of visability to  draw


#                  c    a    s    w   wm
coeffs = np.array([0.7, 0.1,  0.3,  1, 0.1])


# boids initialization
boids = np.zeros((N, 6), dtype=np.float64)   # create array
init_boids(boids, asp, vrange=vrange)        # fill array
boids[:, 4:6] = 0.1                          # initial speed


# for video
canvas = scene.SceneCanvas(show=True, size=(w, h), resizable=False)
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))


# boids coloring

n = boids.shape[0]
colors = np.array([[1, 1, 1]] * n)
colors[0] = [0, 1, 1]

# arrows
arrows = scene.Arrow(arrows=directions(boids, dt), arrow_color=colors,
                     arrow_size=10, connect='segments', parent=view.scene)


#  find angle of observing boid to find his  visable area
beta = np.sign(boids[0, 3]) * np.arccos(boids[0, 2] / np.linalg.norm(boids[0, 2:4]))


if alpha <= np.pi:      # first case
    look = scene.Ellipse(center=boids[0, :2], color=(0, 1, 0.9, 0.2), parent=view.scene,
                         radius=(perception, perception), span_angle=alpha*180/np.pi,
                         start_angle=0)
# draw for first case
if alpha < np.pi:
    coords = [boids[0, :2], boids[0, :2]+[perception*np.cos(beta-alpha/2), perception*np.sin(beta-alpha/2)],
    boids[0, :2]+[perception*np.cos(beta+alpha/2), perception*np.sin(beta+alpha/2)]]
    poly = scene.Polygon(pos=coords, color=(0,1,1, 0.2), parent=view.scene)

if alpha > np.pi:   # second case
    look1 = scene.Ellipse(center=boids[0, :2], color=(0, 1, 1, 0.2), parent=view.scene,
                         radius=(perception, perception), span_angle=alpha * 90 / np.pi,
                         start_angle=beta*180 / np.pi)
    look2 = scene.Ellipse(center=boids[0, :2], color=(0, 1, 1, 0.2), parent=view.scene,
                         radius=(perception, perception), span_angle=-alpha*90 / np.pi,
                         start_angle=beta*180 / np.pi)
    coords = [boids[0, :2], boids[0, :2]+[perception*np.cos(beta-alpha/2), perception*np.sin(beta-alpha/2), ],
                  boids[0, :2] + [perception * np.cos(beta), perception * np.sin(beta)],
    boids[0, :2]+[perception*np.cos(beta+alpha/2), perception*np.sin(beta+alpha/2)]]
    poly = scene.Polygon(pos=coords, color=(0, 1, 1, 0.2), parent=view.scene)

# вывод данных

fps_legend = Text(parent=canvas.scene, color='pink', font_size=7)
coeffs_legend = Text(parent=canvas.scene, color='pink', font_size=7)
fps_legend.pos = canvas.size[0]*0.05, canvas.size[1]*0.025
coeffs_legend.pos = canvas.size[0]*0.052, canvas.size[1]*0.1


legend = f"Число агентов N={N}\n"
legend += f"Параметры:\n"
legend += f"alighment: {coeffs[0]} \n"
legend += f"cohesion: {coeffs[1]} \n"
legend += f"separation: {coeffs[2]} \n"
legend += f"walls: {coeffs[3]} \n"
legend += f"noise: {coeffs[4]} \n"
coeffs_legend.text = legend

arrows.set_data(arrows=directions(boids, dt), color=colors)

# make video
writer = imageio.get_writer(f'animation_5000.mp4', fps=60)
fr = 0       # start


def make_video(event):
    '''makes video of the process'''
    global boids, fr, fps_legend, colors        # variables
    if fr % 60 == 0:
        fps_legend.text = "fps =" +f"{canvas.fps:0.1f}"     # print fps changes
    fr += 1  # increases amount of frames

    fps_legend.text = f"FPS: {canvas.fps:0.3f}"
    mask = flocking(boids, perception, coeffs, asp, vrange, ang)
    propagate(boids, dt, vrange, arange)
    periodic_walls(boids, asp)
    wall_avoidance(boids, asp)

    beta = np.sign(boids[0, 3]) * np.arccos(boids[0, 2] / np.linalg.norm(boids[0, 2:4]))
    if alpha <= np.pi:
        look.center = boids[0, :2]  # center change
        look.start_angle = (beta - alpha / 2) * 180 / np.pi
    if alpha < np.pi:
        coords = boids[0, :2] + [[0, 0], [perception * np.cos(beta - alpha / 2), perception * np.sin(beta - alpha / 2)],
                                      [perception * np.cos(beta + alpha / 2), perception * np.sin(beta + alpha / 2)]]
        poly.pos = coords
    if alpha > np.pi:
        look1.center = boids[0, :2]
        look1.start_angle = (beta) * 180 / np.pi
        look2.center = boids[0, :2]
        look2.start_angle = (beta) * 180 / np.pi
        coords = [boids[0, :2], boids[0, :2] + [perception * np.cos(beta - alpha / 2), perception * np.sin(beta - alpha / 2), ],
                       boids[0, :2] + [perception * np.cos(beta), perception * np.sin(beta)],
                       boids[0, :2] + [perception * np.cos(beta + alpha / 2), perception * np.sin(beta + alpha / 2)]]
        poly.pos = coords

    for i in range(1, N):
        if mask[i] == 0:
            colors[i] = [1, 1, 1]
        else:
            colors[i] = [0, 0.4, 1]

    arrows.set_data(arrows=directions(boids, dt), color=colors)

    if fr <= t:  # while number of existing frames less than should be
        # new frame in video
        frame = canvas.render(alpha=False)
        writer.append_data(frame)
    else:   # all frames in video
         # stop making video
        writer.close()
        app.quit()
    canvas.update()


if __name__ == '__main__':
    timer = app.Timer(interval=0, start=True, connect=make_video)
    canvas.measure_fps()    # fps changes
    app.run()

