import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


def ccw(a, b, c):
	return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])

def is_seg_intersect(a, b, c, d):
    # Return true if line segments ab and cd are intersected
    # TODO: add a check for colinear case. 
	return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

def dict_remove_empty(d):
    """ Delete items whose value is an empty list or empty set from a dictionary """
    remove_keys = []
    for k, v in d.items():
        if not v:
            remove_keys.append(k)
    for k in remove_keys:
        del d[k]     

def print_traj():
    """ Print trajectory for Latex plot """
    traj_dir = './data/traj_c4'
    
    fin_name = traj_dir
    with open(fin_name, 'rb') as fin:
        traj_dict = pickle.load(fin)
    fin.close()
    for state in traj_dict['traj']:
        print("%.3f" % state[0], end='    ')
        print("%.3f" % state[1])

    #print(len(traj_dict['traj']))

def plot_traj(config, traj=None):
    """ 
    Plot x, y dimensions of a trajectrory 
    Reserved config:
    -1: only static obstacles (no runtime obstacles).
     0: config used in CAV (name of abstract states are different, since two obstacles are added in runtime)
    """
    # wksp is consistent with that in Workspace. 
    wksp = 1
    
    if wksp==1:
        fig = plt.figure(figsize=(13., 13.))
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(0., 5.), ylim=(0., 5.))
        width, height = 0.5, 0.5
        x_min = 0.
        idx = 0
        while x_min < 5.:
            y_min = 0.
            while y_min < 5.:
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
                # Goal and static obstacles.
                # NOTE: Name of abstract states do not depend on runtime obstacles. 
                if (x_min==4. or x_min==4.5) and y_min==4.5:
                    #plt.text(x_min+0.15, y_min+0.25, 'Goal')
                    pass
                elif y_min==0.:
                    plt.text(x_min+0.15, y_min+0.25, 'Obst')
                elif 1.<=x_min<=3. and y_min==4.5:
                    plt.text(x_min+0.15, y_min+0.25, 'Obst')
                else:     
                    plt.text(x_min+0.3, y_min+0.35, idx+1, fontsize=9)
                    plt.text(x_min+0.3, y_min+0.25, idx+0, fontsize=9)
                    plt.text(x_min+0.3, y_min+0.15, idx+7, fontsize=9)
                    plt.text(x_min+0.3, y_min+0.05, idx+6, fontsize=9)
                    plt.text(x_min+0.1, y_min+0.35, idx+2, fontsize=9)
                    plt.text(x_min+0.1, y_min+0.25, idx+3, fontsize=9)
                    plt.text(x_min+0.1, y_min+0.15, idx+4, fontsize=9)
                    plt.text(x_min+0.1, y_min+0.05, idx+5, fontsize=9)
                    idx += 8
                
                # Runtime obstacles.    
                if config==-1:
                    pass

                elif config==0:
                    if (x_min==0. or x_min==0.5) and (y_min==2. or y_min==2.5):
                        plt.text(x_min+0.15, y_min+0.25, 'Obst')
                    elif x_min>=2.5 and (y_min==2. or y_min==2.5):
                        plt.text(x_min+0.15, y_min+0.25, 'Obst')

                elif config==1:
                    if x_min==0. and (y_min==2. or y_min==2.5):
                        plt.text(x_min+0.15, y_min+0.25, 'Obst')
                    elif x_min>=2. and (y_min==2. or y_min==2.5):
                        plt.text(x_min+0.15, y_min+0.25, 'Obst')

                elif config==2:
                    if x_min<=2.5 and (y_min==2. or y_min==2.5):
                        plt.text(x_min+0.15, y_min+0.25, 'Obst')
                    elif x_min==4.5 and (y_min==2. or y_min==2.5):
                        plt.text(x_min+0.15, y_min+0.25, 'Obst')

                elif config==3:
                    if 1.<=x_min<=3. and (y_min==2. or y_min==2.5):
                        plt.text(x_min+0.15, y_min+0.25, 'Obst')
                   
                elif config==4:     
                    if 1.<=x_min<=3. and (y_min==2. or y_min==2.5):
                        plt.text(x_min+0.15, y_min+0.25, 'Obst')
                    elif x_min==3. and (3.<=y_min<=4.):
                        plt.text(x_min+0.15, y_min+0.25, 'Obst')     

                else:
                    assert False, 'config is undefined'
                
                y_min += height       
            x_min += width       
        
        # Obstacles
        if config==-1:
            obstacles = [[0., 0., 5., 0.5], [1, 4.5, 2.5, 0.5]]
        
        elif config==0:
            obstacles = [[0., 0., 5., 0.5], [0., 2., 1., 1.], [2.5, 2., 2.5, 1.], [1, 4.5, 2.5, 0.5]]

        elif config==1:
            obstacles = [[0., 0., 5., 0.5], [0., 2., 0.5, 1.], [2., 2., 3., 1.], [1, 4.5, 2.5, 0.5]]

        elif config==2:
            obstacles = [[0., 0., 5., 0.5], [0., 2., 3., 1.], [4.5, 2., 0.5, 1.], [1, 4.5, 2.5, 0.5]]

        elif config==3:
            obstacles = [[0., 0., 5., 0.5], [1., 2., 2.5, 1.], [1, 4.5, 2.5, 0.5]]

        elif config==4:
            obstacles = [[0., 0., 5., 0.5], [1., 2., 2.5, 1.], [3., 3., 0.5, 1.5], [1, 4.5, 2.5, 0.5]]     

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

        # Goal
        goals = [[4., 4.5, 1., 0.5]]
        for g in goals:
            ax.add_patch(
                patches.Rectangle(
                    (g[0], g[1]),  
                    g[2],  
                    g[3],
                    facecolor='white',
                    linestyle='solid',
                    linewidth=0.5,
                    edgecolor='black'
                )
            )    
            plt.text(g[0]+0.4, g[1]+0.25, 'Goal', fontsize=16)
        
        # Trajectory
        if traj is not None:
            traj_x = [state[0] for state in traj]
            traj_y = [state[1] for state in traj]
            plt.plot(traj_x, traj_y, '-o') 
        plt.show()

    else:
        assert False, 'wksp is undefined'


if __name__ == "__main__":
    #config = 4
    #plot_traj(config)       

    print_traj() 