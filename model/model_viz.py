from .model import EpiModel, Condition
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class Visualizer():
    def __init__(self, model, figsize=(5, 6)):
        self.fig = plt.figure(constrained_layout=True, figsize=figsize)
        self.gs = self.fig.add_gridspec(5, 3)
        self.model = model
        self.frames = []
        self.positions = None
        self.colors = None
        self.init_plot()
        self.frame_per_step = 4

    def init_plot(self):
        self.ax1 = self.fig.add_subplot(self.gs[:3, :])
        self.ax1.axis('off')
        self.ax2 = self.fig.add_subplot(self.gs[3:, :])
        self.ax1.margins(0, 0) # Set margins to avoid "whitespace"
        step = self.do_step()
        self.colors = step[1]
        self.positions = np.array(step[0])
        self.scat = self.ax1.scatter(
            self.positions[0], self.positions[1], color=self.colors, s=2.5)
        box_bounds = self.model.isol_boxes
        for b in box_bounds:
            self.ax1.axvline(b[0], color='whitesmoke')
            self.ax1.axhline(b[1], color='whitesmoke')

            
        df = self.model.datacollector.get_model_vars_dataframe()
        self.ax2.stackplot(range(len(df)),
                            df.loc[:, ['Infected', 'Not_infected', 'Dead', 'Healed']].to_numpy().T,
                            labels=['Infected', 'Not Infected', 'Dead', 'Healed' ],
                            alpha=1,
                            colors=['red', 'whitesmoke', 'gray', 'limegreen'])
        
        self.ax2.plot(df['Healthcare_potential'].values, color='black')

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

            if a.condition == Condition.Dead:
                color = (0, 0, 0)
            elif a.condition == Condition.Infected:
                color = (1, 0, 0)
            elif a.condition == Condition.Healed:
                color = (0, 1, 0)
            elif a.condition == Condition.Healed:
                color = (0, 1, 0)
            elif a.condition == Condition.Quaranteened:
                color = (1, 0.5, 0)
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

        self.ax2.clear()
        df = self.model.datacollector.get_model_vars_dataframe()
        self.ax2.stackplot(range(len(df)),
                            df.loc[:, ['Infected', 'Not_infected', 'Dead', 'Healed']].to_numpy().T,
                            labels=['Infected', 'Not Infected', 'Dead', 'Healed' ],
                            alpha=1,
                            colors=['red', 'whitesmoke', 'gray', 'limegreen'])
        
        self.ax2.plot(df['Healthcare_potential'].values, color='black')
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
