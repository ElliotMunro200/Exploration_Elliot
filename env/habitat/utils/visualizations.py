import sys

import matplotlib
import matplotlib.cm as cm
import matplotlib.cbook as cbook
import numpy as np

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import skimage

flag = True

def visualize(option, fig, ax, img, grid, fmm_dist, num_explored, pos, goal, dump_dir, rank, ep_no, t,
              visualize, print_images, vis_style):
    for i in range(len(ax)):
        ax[i].clear()
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

    ax[0].imshow(img)
    ax[0].set_title("Observation", family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)

    if vis_style == 1:
        title = "Map"
    else:
        title = "Ground-Truth Map and Pose"

    ax[1].imshow(grid)
    ax[1].set_title(title, family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)
    if option==1:
        ax[1].set(xlabel='Rotation')
    else:
        ax[1].set(xlabel='Navigation')

    '''
    # Draw GT agent pose
    agent_size = 8
    x, y, o = gt_pos
    x, y = x * 100.0 / 5.0, grid.shape[1] - y * 100.0 / 5.0

    dx = 0
    dy = 0
    fc = 'Grey'
    dx = np.cos(np.deg2rad(o))
    dy = -np.sin(np.deg2rad(o))
    ax[1].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * (agent_size * 1.25),
                head_width=agent_size, head_length=agent_size * 1.25,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.9)
    '''

    # Draw predicted agent pose
    agent_size = 8
    x, y, o = pos
    x, y = x * 100.0 / 5.0, grid.shape[1] - y * 100.0 / 5.0

    dx = 0
    dy = 0
    fc = 'Red'
    dx = np.cos(np.deg2rad(o))
    dy = -np.sin(np.deg2rad(o))
    ax[1].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * agent_size * 1.25,
                head_width=agent_size, head_length=agent_size * 1.25,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.6)


    ax[2].imshow(fmm_dist, cmap=cm.RdYlGn,
                 origin='lower', 
                 vmax=abs(fmm_dist).max(), vmin=-abs(fmm_dist).max())
    ax[2].arrow(x - 1 * dx, grid.shape[1] - y - 1 * -dy, dx * agent_size, -dy * agent_size * 1.25,
                head_width=agent_size, head_length=agent_size * 1.25,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.6)


    ax[3].plot(num_explored)
    ax[3].set(xlim=(0, 1000), ylim=(0, 1.2),
       xlabel='timesteps', ylabel='explored',
       title='#explored');
    ax[3].set_aspect(1.0/ax[3].get_data_ratio())
    
    for i in range(0,1000,250):
        ax[3].axvline(x=i, color='r', ls = '--')

    for _ in range(4):
        plt.tight_layout()

    if visualize:
        plt.gcf().canvas.flush_events()
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()

    if print_images:
        fn = '{}/episodes/{}/{}/{}-{}-Vis-{}.png'.format(
            dump_dir, (rank + 1), ep_no, rank, ep_no, t)
        plt.savefig(fn)


def insert_circle(mat, x, y, value):
    mat[x - 2: x + 3, y - 2:y + 3] = value
    mat[x - 3:x + 4, y - 1:y + 2] = value
    mat[x - 1:x + 2, y - 3:y + 4] = value
    return mat


def fill_color(colored, mat, color):
    for i in range(3):
        colored[:, :, 2 - i] *= (1 - mat)
        colored[:, :, 2 - i] += (1 - color[i]) * mat
    return colored


def get_colored_map(mat, collision_map, visited, goal, goal_arbitrary,
                    explored, gt_map, frontier, change_goal_flag):
    m, n = mat.shape
    colored = np.zeros((m, n, 3))
    pal = sns.color_palette("Paired")

    #explorable map
    current_palette = [(0.9, 0.9, 0.9)]
    colored = fill_color(colored, gt_map, current_palette[0]) 

    #explored map
    current_palette = [(235. / 255., 243. / 255., 1.)]
    colored = fill_color(colored, explored, current_palette[0])

    #occupancy map
    colored = fill_color(colored, mat, pal[3])

    #explored map
    #current_palette = [(0.6, 0.6, 0.6)]
    #colored = fill_color(colored, gt_map_explored, current_palette[0])

    #colored = fill_color(colored, gt_map_explored, pal[3])

    #visited trajectory
    colored = fill_color(colored, visited, pal[5])
    #colored = fill_color(colored, visited * visited_gt, pal[5])

    #frontier
    colored = fill_color(colored, frontier, pal[6])

    #collision map
    colored = fill_color(colored, collision_map, pal[2])

    current_palette = sns.color_palette()

    #frontier goal point (closest to the arbitrary goal point)
    selem = skimage.morphology.disk(8)
    goal_mat = np.zeros((m, n))
    goal_mat[goal[0], goal[1]] = 1
    goal_mat = 1 - skimage.morphology.binary_dilation(
        goal_mat, selem) != True

    #arbitrary goal point 
    goal_mat2 = np.zeros((m, n))
    goal_mat2[goal_arbitrary[0], goal_arbitrary[1]] = 1
    goal_mat2 = 1 - skimage.morphology.binary_dilation(
        goal_mat2, selem) != True

    #goal map
    global flag

    if change_goal_flag:
       flag = not flag

    if flag:
        colored = fill_color(colored, goal_mat, current_palette[0])
    else:
        colored = fill_color(colored, goal_mat, current_palette[1])

    colored = fill_color(colored, goal_mat2, current_palette[2])

    colored = 1 - colored
    colored *= 255
    colored = colored.astype(np.uint8)

    return colored
