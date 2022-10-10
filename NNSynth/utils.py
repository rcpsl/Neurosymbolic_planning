import numpy as np
from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


def plot_traj_robot2d(x_all_trajs):
    obstacles = [[-2., -2., 4., 4.]]
    goals = [[5., 5., 2., 2.]]
    fig = plt.figure(figsize=(10., 10.))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-10., 10.), ylim=(-10., 10.))
    width, height = 0.5, 0.5
    x_min = -10.
    while x_min < 10.:
        y_min = -10.
        while y_min < 10.:
            ax.add_patch(
                patches.Rectangle(
                    (x_min, y_min),  
                    width,  
                    height,
                    facecolor='white',
                    linestyle='dotted',
                    linewidth=0.5,
                    edgecolor='black',
                    label='Label'
                )
            )  
            y_min += height       
        x_min += width       
    for obst in obstacles:
        ax.add_patch(
            patches.Rectangle(
                (obst[0], obst[1]),  
                obst[2],  
                obst[3],
                facecolor='gray',
                linestyle='solid',
                linewidth=0.5,
                edgecolor='black'
            )
        )
    for g in goals:
        ax.add_patch(
            patches.Rectangle(
                (g[0], g[1]),  
                g[2],  
                g[3],
                facecolor='green',
                linestyle='solid',
                linewidth=0.5,
                edgecolor='black'
            )
        )    
        plt.text(g[0]+0.5, g[1]+1., 'Goal', fontsize=16)
    # Trajectory
    for traj in x_all_trajs:
    #if traj is not None:
        traj_x = [state[0] for state in traj]
        traj_y = [state[1] for state in traj]
        plt.plot(traj_x, traj_y, '-o') 
    plt.show()


