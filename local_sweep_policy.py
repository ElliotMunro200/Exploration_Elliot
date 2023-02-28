# TRAIN A LOCAL SWEEP POLICY: same as main_clustering.py except for change in scenes, desired saving locations, and the nature of the policy (normal A2C with PPO, no longer OC).
# 1) load scenes: only single room ones (probably only from Gibson), need to check how scenes are used in FO: number of episodes per scene, how initial positions are given.
# 2) make a policy: uses full state, outputting atomic actions. Adjusted model (type, shape, size). Explores the entire room within 25 timesteps (reward is the normal reward).
# 3) evaluate the policy - execute it in the scenes, save to rollouts? need to learn the proper learning logic from FO, optionally saving .mat files.
# 4) train the policy - same general training logic (but different model).
# 5) save the model - once at the end of training.
# 6) load and execute the model on new scenes, saving .mat files.
# 7) execute the testing visualization + save.

#To do:
#how to set up and execute using the proper scenes? - use the correct args. scene parameters are all defined in the .json files which are pointed to by the relevant .yaml files.
#CORRECT ARGS: 
#TRAINING GIBSON: --num_maps=5/6, --num_episodes=100 (per process), --eval=0 (don't save policy evaluations, but save model), --exp_name="exp#" (where to load from and log to), --load_global=0 (or path to load from e.g. "./tmp/models/exp#/) --task_config="tasks/pointnav_gibson.yaml", --split="train", --num_global_steps=40 (=number of macros), 
#--num_local_steps=25 (=number of steps per macro). 
#TESTING GIBSON: --num_maps=5/6, --num_episodes=100000 (or default large number), --eval=1 (save .mat files for plotting), --exp_name="exp#" (for model loading), --load_global=path (of model to test), --task_config="tasks/pointnav_gibson.yaml", --split="val", --num_global_steps=40 (=number of macros), --num_local_steps=25 (=number of steps per macro).
#TRAINING MP3D: same as TRAINING GIBSON except different: --exp_name="exp#", --task_config="tasks/pointnav_mp3d.yaml", --split="train", --num_global_steps=20 (=number of macros), 
#--num_local_steps=50 (=number of steps per macro). 
#TESTING MP3D: same as TESTING GIBSON but correct: --exp_name="exp#", --load_global=path, --task_config="tasks/pointnav_mp3d.yaml", --split="test", --num_global_steps=20 (=number of macros), --num_local_steps=50 (=number of steps per macro).    

#how to properly set up the model? - done (I think). 
#entropies? - used in the PPO update loss function to maintain entropy in the final distributions we sample the actions from.
#how to perform the training? 

#if loading a model for continued training does it know what step it is up to? if not, the training as determined by the .json will be messed up.

import time
from pytictoc import TicToc
from collections import deque

import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from torch.distributions import Bernoulli
import torch
import torch.nn as nn
from torch.nn import functional as F


import gym
import logging
from local_sweep_policy_args import get_args
from env import make_vec_envs

from utils.local_sweep_storage import GlobalRolloutStorage #, FIFOMemory
#from utils.optimization import get_optimizer

from algo import local_sweep_ppo
from local_sweep_model import RL_Policy
from clustering import frontier_clustering

import sys
import matplotlib
import time

from scipy.ndimage.morphology import binary_dilation
from skimage import morphology

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
import matplotlib.pyplot as plt

matplotlib.use('tkagg')

#Getting args
args = get_args()

#Seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#This function is called after every macro completes, in order to update the frontier map. The maps are moved to the cpu for the computations.
def get_frontier_map(map_gt, explored_gt, visited_gt, bad_frontier_map):
    
    map_gt = map_gt.detach().cpu().numpy() # obstacle map
    explored_gt = explored_gt.detach().cpu().numpy() # explored map
    visited_gt = visited_gt.detach().cpu().numpy() # visited/trajectory map

    selem = morphology.disk(5) # 5 is the radius of the flat, disk-shaped footprint
    # binary dilation expands the non-zero values of an image by the structuring element (selem) as its center passes through the non-zero elements
    contour = binary_dilation(explored_gt==0, selem) & (explored_gt==1) # frontier points (dilated by selem)
    contour = contour & (binary_dilation(map_gt, selem)==0) # same as above, but removing points near obstacles with selem
    contour = contour & (binary_dilation(visited_gt, morphology.disk(1))==0) # removing points previously visited with disc(rad=1)
    contour = contour & (binary_dilation(bad_frontier_map, morphology.disk(1))==0) # removing points in the bad frontier map (the current goal point).
    contour = morphology.remove_small_objects(contour, 2) # removing clusters (square connectivity) of size less than 2

    map_gt = torch.from_numpy(map_gt).float().to(args.device)
    explored_gt = torch.from_numpy(explored_gt).float().to(args.device)
    visited_gt = torch.from_numpy(visited_gt).float().to(args.device)
    return torch.from_numpy(contour.astype(float)).float().to(args.device)

# Since global_downscaling=1, it returns [0, 512, 0, 512], the ultimate full map boundaries (in number of map units). 
# map_size = args.map_size_cm // args.map_resolution = 2560 / 5 = 512.
def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes

    if args.global_downscaling > 1:
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

    return [gx1, gx2, gy1, gy2]

# Executes the whole FO(+clustering) algorithm, with policy evaluation and training options.
def main():

    # Timing runtimes of parts of the algorithm execution.
    timer = TicToc()
    timer.tic()
    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name) #is: "./tmp//models/exp#/" 
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name) #is: "./tmp//dump/exp#/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists("{}/images/".format(dump_dir)):
        os.makedirs("{}/images/".format(dump_dir))
    #Setting up the logging at the level of INFO (logs at this level or above).
    logging.basicConfig(
        filename=log_dir + 'train.log', #is: "./tmp//models/exp#/train.log"
        level=logging.INFO) 
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args) #logging the arguments on the root logger.

    # Logging and loss variables
    num_scenes = args.num_processes #number of simultaneous scenes executing (just in training? - don't think so.)
    num_episodes = int(args.num_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    policy_loss = 0

    g_masks = torch.ones(num_scenes).float().to(device)

    best_g_reward = -np.inf

    #plots
    plt.ion() #interactive (mode) on, figures automatically shown.
    fig, ax = plt.subplots(3, num_scenes, figsize=(10, 2.5), facecolor="whitesmoke") # plotting terminations, values, options and their losses.
    if len(np.shape(ax))==1:
        ax = np.expand_dims(ax, axis=1)

    plt.pause(0.001) # for crude animation. figure is updated before pause, GUI event loop runs during pause.

    if args.eval:
        traj_lengths = args.max_episode_length // args.num_local_steps #1000//25=40
        explored_area_log = np.zeros((num_scenes, num_episodes, traj_lengths)) #these never seem to be updated?
        explored_ratio_log = np.zeros((num_scenes, num_episodes, traj_lengths))

    g_episode_rewards = deque(maxlen=1000)
    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))
    
    
    # Starting environments
    torch.set_num_threads(1) #num threads for intraop parallelism on CPU
    envs = make_vec_envs(args) # I don't know right now how this works?
    obs, infos = envs.reset() # the obs here is rgb

    # Initialize map variables
    ### Full map consists of 5 or 6 channels containing the following:
    ### 0. Obstacle Map
    ### 1. Exploread Area
    ### 2. Current Agent Location
    ### 3. Past Agent Locations
    ### 4. Frontier Map
    ### 5. [Frontier Map cluster centroids]
    num_maps = args.num_maps

    torch.set_grad_enabled(False) #context manager that sets gradient calculation to on or off. 

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution #2560 / 5 = 512
    full_w, full_h = map_size, map_size #512, 512
    local_w, local_h = int(full_w / args.global_downscaling), int(full_h / args.global_downscaling) #512,512

    # Initializing full, local (same size), bad frontier map
    full_map = torch.zeros(num_scenes, num_maps, full_w, full_h).float().to(device) 
    local_map = torch.zeros(num_scenes, num_maps, local_w, local_h).float().to(device)
    bad_frontier_map = np.zeros((num_scenes, 1, local_w, local_h)) # the current goal frontier point

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device) #(x,y,o)
    local_pose = torch.zeros(num_scenes, 3).float().to(device) #(x,y,o)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int) #always [0, 512, 0, 512]

    ### Planner pose inputs has 7 dimensions: 1-3 store continuous global agent location (x,y,o); 4-7 store local map boundaries [0, 512, 0, 512].
    #planner_pose_inputs = np.zeros((num_scenes, 7))

    # initializes the maps (local=full) and the pose as the centre of the full map boundaries (=lmb).
    def init_map_and_pose():
        full_map.fill_(0.)
        bad_frontier_map.fill(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0 # map_size half-breadth = 12.8m (x,y), middle of the full area in m.

        locs = full_pose.cpu().numpy() #full pose is only used in this function.
        #planner_pose_inputs[:, :3] = locs # = [12.8, 12.8, 0.0] (x,y,o)
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0] # row of location in map in m, column of location in map in m.
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution), # map co-ordinates (blocks in map) = 256, 256.
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0 # size of the agent is 3x3=9 blocks.
            full_map[e, 4, :] = get_frontier_map(full_map[e, 0, :, :].detach(), \
                                        full_map[e, 1, :, :].detach(), full_map[e, 3, :, :].detach(), bad_frontier_map[e,0,:,:])

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            #planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0, # [0.0, 0.0, 0.0]
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] # local map = full map, since lmb=full_w, full_h.
            local_pose[e] = full_pose[e] - \
                            torch.from_numpy(origins[e]).to(device).float() # local_pose = full_pose, since origins = 0.

    init_map_and_pose()

    # Occupancy map observation space
    map_observation_space = gym.spaces.Box(0, 1,
                                         (num_maps,
                                          local_w, #512, 512
                                          local_h), dtype='uint8')

    # Policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=(2,), dtype=np.float32)

    # RGB observation space
    rgb_observation_space = gym.spaces.Box(0, 255,
                                         (3,
                                          args.frame_width, #256, 256
                                          args.frame_width), dtype='uint8')

    # Global policy recurrent layer sizes
    l_hidden_size = args.local_hidden_size #visual encoder hidden vector, 512.
    g_hidden_size = args.global_hidden_size # global feature hidden vector, 256.


    # Global policy (option RL)
    g_policy = RL_Policy(map_observation_space.shape, envs.venv.action_space, g_action_space,
                         rgb_observation_space.shape,
                         hidden_size=l_hidden_size, #512
                         use_deterministic_local=args.use_deterministic_local, #0
                         device=args.device, #cuda:0
                         base_kwargs={'recurrent': args.use_recurrent_global, #1 - can test with 0 if any erosion of rewards.
                                      'hidden_size': g_hidden_size, #256
                                      'downscaling': args.global_downscaling #1
                                      }).to(device)

    # wrapper that does PPO update on the actor critic.
    g_agent = local_sweep_ppo.PPO(g_policy, args.clip_param, args.ppo_epoch,
                       args.num_mini_batch, args.value_loss_coef,
                       args.entropy_coef, lr=args.global_lr, eps=args.eps,
                       max_grad_norm=args.max_grad_norm)


    # Storage
    g_rollouts = GlobalRolloutStorage(args.num_global_steps, #25
                                          num_scenes, map_observation_space.shape,
                                          g_action_space, g_policy.rec_state_size, rgb_observation_space.shape,
                                          1).to(device)

    # Loading model

    if args.load_global != "0": # default is false, since load_global=0. Otherwise, must be a path to a model .global for loading.
        print("Loading global {}".format(args.load_global))
        state_dict = torch.load(args.load_global, # loads an object saved with torch.save() from a file.
                                map_location=lambda storage, loc: storage) # default deserialization to cpu is used.
        g_policy.load_state_dict(state_dict) #Loading deserialized state_dict into the model. 
        #saving the model parameters as a serialized state_dict. Default location is: "./tmp//models/exp1/last_loaded_gibson_version.global".
        torch.save(g_policy.state_dict(), os.path.join(log_dir, "model_best_last_loaded_version.global"), _use_new_zipfile_serialization=False)

    if not args.train_global: #def=1
        g_policy.eval() # dropout and batchnorm and maybe other layers behave differently when in eval mode (self.training=False).

    # Pose found from infos (x,y,0) as produced by "obs, infos = envs.reset()".
    poses = torch.from_numpy(np.asarray( #should be local_pose?
        [infos[env_idx]['sensor_pose'] for env_idx
         in range(num_scenes)])
    ).float().to(device)

    # Compute Global policy input (maps)
    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, num_maps, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)

    global_input[:, 0:num_maps, :, :] = local_map.detach()

    g_rollouts.obs[0].copy_(global_input)
    g_rollouts.rgb[0].copy_(obs)
    g_rollouts.extras[0].copy_(global_orientation)

    # Run Policy (initial values found)

    g_action, g_value, g_action_log_prob, g_rec_states = \
        g_policy.act(
            g_rollouts.obs[0],
            num_scenes,
            g_rollouts.rec_states[0],
            g_rollouts.rgb[0],
            g_rollouts.masks[0],
            extras=g_rollouts.extras[0],
            deterministic=False
        )

    start = time.time()
    
    total_num_steps = -1

    print("Timer for phase 1")
    timer.toc()
    timer.tic()
    
    current_option = np.zeros((num_scenes,1))
    
    # The algo (policy eval+training loop): for each episode, for each atomic timestep, determine an atomic action (for each scene) according to the current macro. 
    # Save rewards, etc. in the GlobalRolloutStorage
    for ep_num in range(num_episodes):
        
        g_reward = torch.from_numpy(np.zeros((num_scenes))).float().to(device) #one scalar value for each scene on gpu
        g_reward_all = np.zeros((1,num_scenes)) #one scalar value for each scene on cpu

        torch.set_grad_enabled(False)
        
        g_value_all = [[g_value[i].cpu().numpy()] for i in range(num_scenes)] #maybe should be just [i]?

        for step in range(args.max_episode_length):
            total_num_steps += 1

            action = g_action.detach().cpu().squeeze(dim=1).numpy().astype(np.float64) # putting actions on cpu.

            print("Timer for phase 2")
            timer.toc()
            timer.tic()
            # ------------------------------------------------------------------
            # Env step
            action = torch.from_numpy(action)
            print(action)
            
            # returns the new: obs['rgb'/'depth'] infos - maps/pose.
            obs, rew, done, infos = envs.step(action)

            #what are these? "not dones".
            g_masks = torch.FloatTensor([0 if x else 1
                                         for x in done]).to(device)
            # execute visualize(): make + save the 4 subplot figures (rgb, colored_map=get_colored_map(), distance_plot, explored_prop), update self.num_explored (explored_prop).
            envs.update_visualize(current_option) #all outputted figures will have "Navigation" on them.

            print("Timer for phase 3")
            timer.toc()
            timer.tic()

            # ------------------------------------------------------------------
            # Reinitialize variables when episode ends
            if step == args.max_episode_length - 1:  # Last episode step
                init_map_and_pose()
            # ------------------------------------------------------------------

            # Update maps 0-3 (not frontier). They come directly from the habitat envs, through infos with line just above: "obs, rew, done, infos = envs.step(action)". 
            for env_idx in range(num_scenes):
                env_obs = obs[env_idx].to("cpu")
                env_poses = torch.from_numpy(np.asarray(
                    infos[env_idx]['sensor_pose']
                )).float().to("cpu")
                env_gt_fp_projs = torch.from_numpy(np.asarray(
                    infos[env_idx]['fp_proj']
                )).unsqueeze(0).float().to("cpu")
                env_gt_fp_explored = torch.from_numpy(np.asarray(
                   infos[env_idx]['fp_explored']
                )).unsqueeze(0).float().to("cpu")
                env_gt_pose_err = torch.from_numpy(np.asarray(
                   infos[env_idx]['pose_err']
                )).float().to("cpu")
                local_map[env_idx, 0, :, :] = env_gt_fp_projs #occupancy
                local_map[env_idx, 1, :, :] = env_gt_fp_explored #explored
                local_pose[env_idx] = env_poses


            locs = local_pose.cpu().numpy()
            local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
            for e in range(num_scenes):
                r, c = locs[e, 1], locs[e, 0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

            # ------------------------------------------------------------------
            print("Timer for phase 4")
            timer.toc()
            timer.tic()

            for e in range(num_scenes):
                g_value_all[e].append(g_value[e].cpu().numpy().copy())


            print("Timer for phase 5")
            timer.toc()
            timer.tic()
            # ------------------------------------------------------------------

            #Update frontier map
            locs = local_pose.cpu().numpy()
            for e in range(num_scenes):
                global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
                local_map[e, 4] = get_frontier_map(local_map[e, 0, :, :].detach(), \
                    local_map[e, 1, :, :].detach(), local_map[e, 3, :, :].detach(), bad_frontier_map[e,0,:,:])
                if num_maps >= 6:
                    frontier_map = local_map[e, 4, :, :].detach().cpu().numpy()
                    num_frontier_points = len(np.nonzero(frontier_map)[0])
                    if num_frontier_points < 5:
                        print(f"ONLY {num_frontier_points} FRONTIER POINTS, SO NOT ENOUGH FOR CLUSTERING")
                        frontier_clusters = np.zeros(np.shape(frontier_map))
                    else:
                        frontier_clusters = frontier_clustering(frontier_map, step=0, algo="AGNES", metric=None, save_freq=None)
                    local_map[e, 5] = torch.from_numpy(frontier_clusters.astype(float)).float().to(device)
            global_input = local_map
                
            # Get exploration reward and metrics, args.num_local_steps = 1 so getting reward every step.
            g_reward = torch.from_numpy(np.asarray(
                [infos[env_idx]['exp_reward'] for env_idx
                 in range(num_scenes)])
            ).float().to(device)

            print("Rewards:")
            print(g_reward)  

            g_reward_all = np.concatenate((g_reward_all, np.expand_dims(g_reward.cpu().numpy(), axis=0)), axis=0) # shape(n_steps, n_scenes), on cpu

            #Plot curves
            for e in range(num_scenes):
                #if num_scenes == 1:
                #    e = None

                ax[0,e].clear()
                ax[1,e].clear()
                ax[2,e].clear()

                ax[0,e].plot(g_value_all[e])
                ax[2,e].plot(g_reward_all[:,e])

            #ax[1,0].plot(g_value_losses)
            #ax[1,1].plot(g_action_losses)
            #ax[1,2].plot(g_dist_entropies)

            plt.gcf().canvas.flush_events()
            fig.canvas.start_event_loop(0.001)
            plt.pause(0.001)


            g_process_rewards += g_reward.cpu().numpy() # g_process_rewards = np.zeros((num_scenes))
            # is a dynamically updated array of total rewards for the episode.
            g_total_rewards = g_process_rewards * (1 - g_masks.cpu().numpy()) # g_masks = [0 if done, 1 not done, for each e]
            g_process_rewards *= g_masks.cpu().numpy() # when the episode is done, the masks will be 0, the process rewards will go to 0, and the total rewards go to the full amount.
            per_step_g_rewards.append(np.mean(g_reward.cpu().numpy())) # mean reward for the step across all actions.

            # appending the nonzero episode rewards to the g_episode_rewards deque (every step).
            if np.sum(g_total_rewards) != 0:
                for tr in g_total_rewards:
                    g_episode_rewards.append(tr) if tr != 0 else None

            # Add samples to global policy storage. this is fine
            g_rollouts.insert(
                global_input, obs, g_rec_states,
                g_action, g_action_log_prob, g_value, current_option,
                g_reward.detach(), g_masks, global_orientation
            )                
            # need to see how this works.
            g_action, g_value, g_action_log_prob, g_rec_states = \
                g_policy.act(
                    g_rollouts.obs[step + 1],
                    num_scenes,
                    g_rollouts.rec_states[step + 1],
                    g_rollouts.rgb[step + 1],
                    g_rollouts.masks[step + 1],
                    extras=g_rollouts.extras[step + 1],
                    deterministic=False
                )

            #print("g_value")
            #print(g_value)

            print("Timer for phase 6")
            timer.toc()
            timer.tic()
            # ------------------------------------------------------------------
            if step == args.max_episode_length - 1: #end of (25) step episode.
                # print rewards
                ep_rew_by_scene = np.sum(g_reward_all, axis=0)
                for e, rew in enumerate(ep_rew_by_scene):
                    print(f"Scene {e}, Reward {rew}")

                ### TRAINING
                torch.set_grad_enabled(True)

                print("#######Training Global Policy#######")
                    
                g_next_value = g_policy.get_value(
                    g_rollouts.obs[-1],
                    g_rollouts.rec_states[-1],
                    g_rollouts.rgb[-1],
                    g_rollouts.masks[-1],
                    extras=g_rollouts.extras[-1]
                )
                #
                g_rollouts.compute_returns(args.gamma, g_next_value.detach())
                # average values from across the 4 PPO training epochs
                g_value_loss, g_action_loss, g_dist_entropy = g_agent.update(g_rollouts) #the training happens here.
                g_value_losses.append(g_value_loss * args.value_loss_coef)
                g_action_losses.append(g_action_loss)
                g_dist_entropies.append(g_dist_entropy * args.entropy_coef)
                print("#######Finish Training Global Policy#######")
                g_rollouts.after_update()
                # ------------------------------------------------------------------

                # Finish Training
                torch.set_grad_enabled(False)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Logging+printing (once per episode)
            if total_num_steps % args.max_episode_length == 0:
                end = time.time()
                time_elapsed = time.gmtime(end - start)
                log = " ".join([
                    "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                    "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                    "num timesteps {},".format(total_num_steps *
                                               num_scenes),
                    "FPS {},".format(int(total_num_steps * num_scenes \
                                         / (end - start)))
                ])

                log += "\n\tRewards:"

                if len(g_episode_rewards) > 0:
                    log += " ".join([
                        " Global step mean/med rew:",
                        "{:.4f}/{:.4f},".format(
                            np.mean(per_step_g_rewards), #mean over the per_step_g_rewards deque. Isn't helpful for me.
                            np.median(per_step_g_rewards)),
                        " Global eps mean/med/min/max eps rew:",
                        "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_episode_rewards),
                            np.median(g_episode_rewards),
                            np.min(g_episode_rewards),
                            np.max(g_episode_rewards))
                    ])

                log += "\n\tLosses:"


                if args.train_global and len(g_value_losses) > 0:
                    log += " ".join([
                        " Global Loss value/action/dist:",
                        "{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_value_losses),
                            np.mean(g_action_losses),
                            np.mean(g_dist_entropies))
                    ])


                print(log)
                logging.info(log)
            # ------------------------------------------------------------------
            
            # ------------------------------------------------------------------
            # End of training run model saving
            if (step == args.max_episode_length - 1) and not args.eval:
                # Once after last step of last episode
                if ep_num == num_episodes - 1:
                    #is: "./tmp//models/exp#/model_last.global"
                    model_save_name = f"model_last.global" 
                    torch.save(g_policy.state_dict(),
                               os.path.join(log_dir, model_save_name))
                # If 5th of way through episodes
                if (ep_num+1) % (num_episodes/5) == 0:
                    model_save_name = f"model_ep_{ep_num+1}.global"
                    torch.save(g_policy.state_dict(),
                               os.path.join(log_dir, model_save_name)) 
                # If model is at the best performance of the minimum 100 episodes trained this run
                if len(g_episode_rewards) >= num_scenes*25 and (np.mean(g_episode_rewards) >= best_g_reward):
                    model_save_name = f"model_best.global"
                    torch.save(g_policy.state_dict(),
                               os.path.join(log_dir, model_save_name)) 
            
                best_g_reward = np.mean(g_episode_rewards)
            # ------------------------------------------------------------------
    def plot(data):
        len_data = len(data)
        max_rew = max(data)
        t = range(len_data)
        plt.plot(t, data, 'b', label=f"episodic reward")
        plt.legend()
        plt.axis([0,len_data,0,max_rew])
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episodic Reward')
        plt.axhline(y=1.0, color='k', ls = '--')
        for i in range(0,len_data,50):
            plt.axvline(x=i, color='r', ls = '--')
        fn = f"{log_dir}EpRews_local.png"
        plt.savefig(fn)
        print(f"SAVED EPISODIC REWARDS TO: {fn}")

    if np.sum(g_episode_rewards) != 0:
        plot(g_episode_rewards)
    
    # If evaluating the model, save the explored_area_log (not updated or useful? since .mat files save every episode the explored prop which is collected every step). 
    if args.eval:
        logfile = open("{}/explored_area.txt".format(dump_dir), "w+") #is: "./tmp//dump/exp#/explored_area.txt"
        for e in range(num_scenes):
            for i in range(explored_area_log[e].shape[0]):
                logfile.write(str(explored_area_log[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        logfile = open("{}/explored_ratio.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(explored_ratio_log[e].shape[0]):
                logfile.write(str(explored_ratio_log[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        log = "Final Exp Area: \n"
        for i in range(explored_area_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(explored_area_log[:, :, i]))

        log += "\nFinal Exp Ratio: \n"
        for i in range(explored_ratio_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(explored_ratio_log[:, :, i]))

        print(log)
        logging.info(log)
    
    end = time.time()
    time_elapsed = time.gmtime(end - start)
    print_time = time.strftime("%Hh %Mm %Ss", time_elapsed)
    print(f"end: {end}")
    print(f"start: {start}")
    print(f"time_gm: {time_elapsed}")
    print(f"TOTAL TIME: {print_time}")


if __name__ == "__main__":
    main()






