import scipy.io
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='MATLAB .mat file reader args')

    ## General Arguments
    parser.add_argument('-f', '--function', type=str, default="short")
    parser.add_argument('-exps', '--exp_names', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-exp_output', '--exp_output', type=str, default="exp_data/",
                        help='path to dump experiments data')
    parser.add_argument('-eps', '--eps', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-n', '--num_scenes', type=int, default=4, help='number of parallel executed scenes')

    args = parser.parse_args()
    return args

def plot_short(full_data):
    t = range(len_data)
    for method in range(len(full_data)):
        #print(full_data[method,-10:])
        plt.plot(t, full_data[method], format_strings[method], label=f"{exp_names[method]}")
    plt.legend()
    plt.tick_params(labeltop=False, labelright=True)
    plt.axis([0,len_data,0,1.2])
    plt.xlabel('Timesteps')
    plt.ylabel('Coverage')
    plt.title('Exploration')
    for i in np.arange(0,1.2,0.25):
        plt.axhline(y=i, color='k', ls = ':')
    for i in np.arange(0,len_data,len_data/5):
        plt.axvline(x=i, color='r', ls = '--')

    fn = f"{read_dir}TestVis.png"
    plt.savefig(fn)
    #print(f"SAVED IMAGE TO: {fn}")

def plot_long(full_data):
    t = range(1,num_eps+1)
    for method in range(len(full_data)):
        #print(full_data[method,-10:])
        plt.plot(t, full_data[method], format_strings[method], label=f"{exp_names[method]}")
    plt.legend()
    plt.tick_params(labeltop=False, labelright=True)
    plt.axis([1,num_eps,0,1.2])
    plt.xlabel('Episode')
    plt.ylabel('Coverage')
    plt.title('Exploration')
    for i in np.arange(0,1.2,0.25):
        plt.axhline(y=i, color='k', ls = ':')
    #for i in np.arange(0,num_eps,num_eps/2):
    #    plt.axvline(x=i, color='r', ls = '--')

    fn = f"{read_dir}TestVis.png"
    plt.savefig(fn)
    #print(f"SAVED IMAGE TO: {fn}")

# will for a given episode, average the explored props by timestep across the scenes --> one average plot over the timesteps per episode,
# then will similarly do the same for other episodes, and then average these too across the episodes by timestep --> one average plot across scenes and episodes by timestep.
def mat_reader_short(exp_dir, num_scenes):
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
            #print(explored_prop_by_timestep)
            by_timestep_scene[scene,:] = explored_prop_by_timestep
        by_timestep_scene_avg = by_timestep_scene.mean(axis=0)
        #print(by_timestep_scene_avg)
        by_timestep_all[ep-ep0,:] = by_timestep_scene_avg    
    by_timestep_all_avg = by_timestep_all.mean(axis=0)
    print(by_timestep_all_avg)
    return by_timestep_all_avg

def mat_reader_long(exp_dir, num_scenes):
    exp_dir = f"{exp_dir}/"
    by_episode_all = np.zeros((num_eps))
    for ep in range(ep0,ep1+1):
        by_episode_scene = np.zeros((num_scenes))
        for scene in range(num_scenes):    
            mat_file = f"{scene}-{ep}.mat"
            full_dir = read_dir+exp_dir+mat_file
            mat = scipy.io.loadmat(full_dir)
            #print(f"LOADED FROM: {full_dir}")
            explored_props = mat["num_explored"][0]
            #print(explored_props)
            by_episode_scene[scene] = explored_props[-1]
        by_episode_scene_avg = by_episode_scene.mean(axis=0)
        by_episode_all[ep-ep0] = by_episode_scene_avg
    return by_episode_all

def len_finder():
    exp_dir = f"{exp_names[0]}/"
    mat_file = f"{0}-{1}.mat"
    mat = scipy.io.loadmat(read_dir+exp_dir+mat_file)
    explored_prop_by_timestep = mat["num_explored"][0]
    #print(f"EXPLORED PROPS, LEN FINDER: {explored_prop_by_timestep}")
    return len(explored_prop_by_timestep)

#getting the args to build the desired plot
args = get_args()

#methods being comapred (names of respective directories as strings e.g. "exp_local" or "exp11")
exp_names = args.exp_names
num_methods = len(exp_names)

#directory that the method data is in
read_dir = f"{args.exp_output}"

#the eval episodes in question
eps = args.eps
ep0 = int(eps[0])
if len(eps) == 2:
    ep1 = int(eps[1])
else:
    ep1 = ep0
num_eps = ep1+1-ep0

#finding the timestep length of the episodes in question from the 1st episode (always 1 timestep shorter than the rest)
len_data = len_finder()

#correcting for the first episode being one timestep shorter, i.e. either 999 -> 1000 or 24 -> 25.
len_data += 1
print(f"CORRECTED TIMESTEPS OF DATA: {len_data}")

#the number of scenes/processes iterated through
num_scenes = args.num_scenes

#format list for plotting of different methods
format_strings = ['b','r-.','g--','m-.','yx:','co-']

#plotting each method on the one graph with the mat_reader for the desired function (i.e. "short" or "long")
print(f"PLOTTING WITH FUNCTION: {args.function}")

if args.function == "short":
    full_data = np.zeros((num_methods,len_data))
    for method in range(num_methods):
        print(f"METHOD FROM: {read_dir}{exp_names[method]}")
        data = mat_reader_short(exp_names[method], num_scenes)
        full_data[method,:] = data
    plot_short(full_data)

elif args.function == "long":
    full_data = np.zeros((num_methods,num_eps))
    for method in range(num_methods):
        print(f"METHOD FROM: {read_dir}{exp_names[method]}")
        data = mat_reader_long(exp_names[method], num_scenes)
        full_data[method,:] = data
    plot_long(full_data)



# 1) demonstrate correct plot over all timesteps (from one episode, from one scene) and saving figure. ~
# 2) generalize code to averaging coverage across test scenes. ~
# 3) train normally for both FO and FO+clustering (saving necessary data correctly). ~
# 4) edit code to generalize for method comparisons. ~
# 5) test FO and FO+clustering models on test set (saving mat files).
# 6) use mat_reader to analyze data and save figure. 
