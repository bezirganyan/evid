from .model import EpiModel
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class Visualizer():
    def __init__(self, model, figsize=(5, 5)):
        self.fig = plt.figure(constrained_layout=True)
        self.gs = self.fig.add_gridspec(5, 3)
        self.model = model
        self.frames = []
        self.positions = None
        self.colors = None
        self.init_plot()
        self.frame_per_step = 4

    def init_plot(self):
        self.ax1 = self.fig.add_subplot(self.gs[:3, :])
        self.ax2 = self.fig.add_subplot(self.gs[3:, :])
        step = self.do_step()
        self.colors = step[1]
        self.positions = np.array(step[0])
        self.scat = self.ax1.scatter(
            self.positions[0], self.positions[1], color=self.colors, s=2.5)
        box_bounds = self.model.isol_boxes
        for b in box_bounds:
            self.ax1.axvline(b[0], color='gray')
            self.ax1.axhline(b[1], color='gray')

            
        self.stack_chart = self.ax2.stackplot(range(len(self.model.datacollector.get_model_vars_dataframe())), self.model.datacollector.get_model_vars_dataframe().to_numpy().T, labels=['Infected', 'Dead', 'Healed', 'Not Infected'], alpha=1, colors=['red', 'black', 'limegreen', 'white'])

        self.ax2.set_title('Stacked chart')
        plt.legend(loc='upper right')
        self.ax2.set_ylabel('People')
        self.ax2.margins(0, 0) # Set margins to avoid "whitespace"

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

        # updated_stacked_data = self.model.datacollector.get_model_vars_dataframe().to_numpy().T
        # self.stack_chart[0].set_offsets(updated_stacked_data[0])
        # self.stack_chart[1].set_offsets(updated_stacked_data[1])
        # self.stack_chart[2].set_offsets(updated_stacked_data[2])
        # self.stack_chart[3].set_offsets(updated_stacked_data[3])
        self.ax2.clear()
        self.ax2.stackplot(range(len(self.model.datacollector.get_model_vars_dataframe())), self.model.datacollector.get_model_vars_dataframe().to_numpy().T, labels=['Infected', 'Dead', 'Healed', 'Not Infected'], alpha=1, colors=['red', 'black', 'limegreen', 'white'])
        self.ax2.set_title('Stacked chart')
        plt.legend(loc='upper left')
        self.ax2.set_ylabel('People')
        self.ax2.margins(0, 0) # Set margins to avoid "whitespace"

    def create_animation(self, show=True, save_path=False, interval=1, save_count=1000, frame_per_step=4):
        self.frame_per_step = frame_per_step
        anim = FuncAnimation(self.fig, self.animate,
                             interval=1, save_count=save_count)
        if save_path:
            anim.save(save_path, writer='imagemagick', fps=20)
        if show:
            return anim
