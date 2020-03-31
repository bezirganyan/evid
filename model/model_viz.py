from .model import CoronaModel
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class Visualizer():
    def __init__(self, model, figsize=(5, 5)):
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.model = model
        self.frames = []
        self.positions = None
        self.colors = None
        self.init_plot()
        self.frame_per_step = 4

    def init_plot(self):
        step = self.do_step()
        self.colors = step[1]
        self.positions = np.array(step[0])
        self.scat = self.ax.scatter(
            self.positions[0], self.positions[1], color=self.colors, s=2.5)
        box_bounds = self.model.isol_boxes
        for b in box_bounds:
            self.ax.axvline(b[0], color='gray')
            self.ax.axhline(b[1], color='gray')

    def do_step(self):
        self.model.step()
        xpos, ypos, colors = [], [], []
        for a in self.model.schedule.agents:
            xpos.append(a.pos[0])
            ypos.append(a.pos[1])

            if a.dead:
                color = (0, 0, 0)
            elif a.infected:
                color = (1, 0, 0)
            elif a.healed:
                color = (0, 1, 0)
            else:
                color = (0, 0, 1)
            colors.append(color)
        return (xpos, ypos), colors

    def animate(self, val):
        if not len(self.frames):
            step = self.do_step()
            npos = np.array(step[0])
            self.colors = step[1]
            self.frames = np.linspace(
                self.positions, npos, self.frame_per_step)
            self.positions = npos
        self.scat.set_offsets(
            np.c_[self.frames[0, 0, :], self.frames[0, 1, :]])
        self.scat.set_color(c=self.colors)
        self.frames = self.frames[1:, :, :]

    def create_animation(self, show=True, save_path=False, interval=1, save_count=1000, frame_per_step=4):
        self.frame_per_step = frame_per_step
        anim = FuncAnimation(self.fig, self.animate,
                             interval=1, save_count=save_count)
        if save_path:
            anim.save(save_path, writer='imagemagick', fps=20)
        if show:
            return anim
