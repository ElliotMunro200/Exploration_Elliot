import scipy.io
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='MATLAB .mat file reader args')

    ## General Arguments
    parser.add_argument('-exp_output', '--exp_output', type=str, default="exp_data/",
                        help='path to dump experiments data')
    parser.add_argument('-exps', '--exp_names', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-eps', '--eps', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--num_scenes', type=int, default=4, help='number of parallel executed scenes')
    parser.add_argument('--num_episodes', type=int, default=1000000,
                        help='number of training episodes (default: 1000000)')

    args = parser.parse_args()
    return args

def plot(full_data):
    t = range(len_data)
    for method in range(len(full_data)):
        #print(full_data[method,-10:])
        plt.plot(t, full_data[method], format_strings[method], label=f"{exp_names[method]}")
    plt.legend()
    plt.tick_params(labeltop=False, labelright=True)
    plt.axis([0,1000,0,1.2])
    plt.xlabel('Timesteps')
    plt.ylabel('Coverage')
    plt.title('Exploration')
    for i in np.arange(0,1.2,0.25):
        plt.axhline(y=i, color='k', ls = ':')
    for i in range(0,1000,250):
        plt.axvline(x=i, color='r', ls = '--')

    fn = f"{read_dir}TestVis.png"
    plt.savefig(fn)
    #print(f"SAVED IMAGE TO: {fn}")

# will for a given episode, average the explored props by timestep across the scenes --> one average plot over the timesteps per episode,
# then will similarly do the same for other episodes, and then average these too across the episodes by timestep --> one average plot across scenes and episodes by timestep.
def mat_reader(exp_dir, num_scenes):
    exp_dir = f"{exp_dir}/"
    by_timestep_all = np.zeros((num_eps, len_data))
    for ep in range(ep0,ep1+1):
        by_timestep_scene = np.zeros((num_scenes, len_data))
        for scene in range(num_scenes):    
            mat_file = f"{scene}-{ep}.mat"
            full_dir = read_dir+exp_dir+mat_file
            mat = scipy.io.loadmat(full_dir)
            #print(f"LOADED FROM: {full_dir}")
            explored_prop_by_timestep = mat["num_explored"][0]
            if ep == 1 and len(explored_prop_by_timestep) == (len_data - 1): # to account for 999 timesteps of ep=1 (instead of 1000 timesteps) of each run.
                explored_prop_by_timestep = np.append(explored_prop_by_timestep, explored_prop_by_timestep[-1])
            print(explored_prop_by_timestep[-1])
            by_timestep_scene[scene,:] = explored_prop_by_timestep
        by_timestep_scene_avg = by_timestep_scene.mean(axis=0)
        by_timestep_all[ep-ep0,:] = by_timestep_scene_avg
    by_timestep_all_avg = by_timestep_all.mean(axis=0)
    return by_timestep_all_avg

def len_finder():
    exp_dir = f"{exp_names[0]}/"
    mat_file = f"{0}-{1}.mat"
    mat = scipy.io.loadmat(read_dir+exp_dir+mat_file)
    explored_prop_by_timestep = mat["num_explored"][0]
    #print(f"EXPLORED PROPS, LEN FINDER: {explored_prop_by_timestep}")
    return len(explored_prop_by_timestep)

args = get_args()
exp_names = args.exp_names
read_dir = f"{args.exp_output}"
num_methods = len(exp_names)
num_scenes = args.num_scenes
format_strings = ['b','r-.','g--','m-.']

eps = args.eps
ep0 = int(eps[0])
if len(eps) == 2:
    ep1 = int(eps[1])
else:
    ep1 = ep0
num_eps = ep1+1-ep0

len_data = len_finder()
len_data += 1 #999 -> 1000
assert len_data == 1000
print(f"CORRECTED TIMESTEPS OF DATA: {len_data}")

full_data = np.zeros((num_methods,len_data))
for method in range(num_methods):
    print(f"METHOD FROM: {read_dir}{exp_names[method]}")
    data = mat_reader(exp_names[method], num_scenes)
    full_data[method,:] = data

plot(full_data)

# 1) demonstrate correct plot over all timesteps (from one episode, from one scene) and saving figure. ~
# 2) generalize code to averaging coverage across test scenes. ~
# 3) train normally for both FO and FO+clustering (saving necessary data correctly). ~
# 4) edit code to generalize for method comparisons. ~
# 5) test FO and FO+clustering models on test set (saving mat files).
# 6) use mat_reader to analyze data and save figure. 
