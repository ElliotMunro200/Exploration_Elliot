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
from arguments import get_args
from env import make_vec_envs

from utils.storage import GlobalRolloutStorage #, FIFOMemory
#from utils.optimization import get_optimizer

import algo
from model import RL_Policy

import sys
import matplotlib
import time

from scipy.ndimage.morphology import binary_dilation
from skimage import morphology

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
import matplotlib.pyplot as plt

matplotlib.use('tkagg')

args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


def get_frontier_map(map_gt, explored_gt, visited_gt, bad_frontier_map):

    map_gt = map_gt.detach().cpu().numpy()
    explored_gt = explored_gt.detach().cpu().numpy()
    visited_gt = visited_gt.detach().cpu().numpy()

    selem = morphology.disk(5)

    contour = binary_dilation(explored_gt==0, selem) & (explored_gt==1)
    contour = contour & (binary_dilation(map_gt, selem)==0)
    contour = contour & (binary_dilation(visited_gt, morphology.disk(1))==0)
    contour = contour & (binary_dilation(bad_frontier_map, morphology.disk(1))==0)
    contour = morphology.remove_small_objects(contour, 2)
    
    '''
    global counter
    if counter % 4 == 0:
        plt.imshow(binary_dilation(visited_gt, morphology.disk(10)))
        plt.gcf().canvas.flush_events()
        plt.pause(0.1)
    counter += 1
    '''

    map_gt = torch.from_numpy(map_gt).float().to(args.device)
    explored_gt = torch.from_numpy(explored_gt).float().to(args.device)
    visited_gt = torch.from_numpy(visited_gt).float().to(args.device)
    return torch.from_numpy(contour.astype(float)).float().to(args.device)

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


def main():

    timer = TicToc()
    timer.tic()
    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists("{}/images/".format(dump_dir)):
        os.makedirs("{}/images/".format(dump_dir))

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    policy_loss = 0

    best_cost = 100000
    costs = deque(maxlen=1000)
    exp_costs = deque(maxlen=1000)
    pose_costs = deque(maxlen=1000)

    g_masks = torch.ones(num_scenes).float().to(device)

    best_local_loss = np.inf
    best_g_reward = -np.inf

    # plots
    plt.ion()
    fig, ax = plt.subplots(4, num_scenes + 1, figsize=(12, 10), facecolor="whitesmoke")
    fig.subplots_adjust(hspace=0.5, wspace=0.5, top=0.9, bottom=0.1, left=0.1, right=0.9)
    fig.suptitle("Metrics by macro")
    ax[0, 0].set_title("g_terminations")
    ax[0, num_scenes].set_title("g_termination_losses")
    ax[1, 0].set_title("g_values")
    ax[1, num_scenes].set_title("g_value_losses")
    ax[2, 0].set_title("current_option")
    ax[2, num_scenes].set_title("g_dist_entropies")
    ax[3, 0].set_title("g_rewards")
    ax[3, num_scenes].set_title("g_action_losses")
    plt.pause(0.001)

    if args.eval:
        traj_lengths = args.max_episode_length // args.num_local_steps
        explored_area_log = np.zeros((num_scenes, num_episodes, traj_lengths))
        explored_ratio_log = np.zeros((num_scenes, num_episodes, traj_lengths))

    g_episode_rewards = deque(maxlen=1000)
    g_value_losses = deque(maxlen=1000)
    g_termination_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    l_episode_rewards = deque(maxlen=1000)
    l_value_losses = deque(maxlen=1000)
    l_action_losses = deque(maxlen=1000)
    l_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))
    
    
    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()

    # Initialize map variables
    ### Full map consists of 5 channels containing the following:
    ### 0. Obstacle Map
    ### 1. Exploread Area
    ### 2. Current Agent Location
    ### 3. Past Agent Locations
    ### 4. Frontier Map

    torch.set_grad_enabled(False)

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w, local_h = int(full_w / args.global_downscaling), \
                       int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, 5, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, 5, local_w, local_h).float().to(device)
    bad_frontier_map = np.zeros((num_scenes, 1, local_w, local_h))

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    ### Planner pose inputs has 7 dimensions
    ### 1-3 store continuous global agent location
    ### 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def init_map_and_pose():
        full_map.fill_(0.)
        bad_frontier_map.fill(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0
            full_map[e, 4, :] = get_frontier_map(full_map[e, 0, :, :].detach(), \
                                        full_map[e, 1, :, :].detach(), full_map[e, 3, :, :].detach(), bad_frontier_map[e,0,:,:])

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                            torch.from_numpy(origins[e]).to(device).float()

    init_map_and_pose()

    # Occupancy map observation space
    map_observation_space = gym.spaces.Box(0, 1,
                                         (5,
                                          local_w,
                                          local_h), dtype='uint8')

    # Policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=(2,), dtype=np.float32)

    # RGB observation space
    rgb_observation_space = gym.spaces.Box(0, 255,
                                         (3,
                                          args.frame_width,
                                          args.frame_width), dtype='uint8')

    # Global policy recurrent layer sizes
    l_hidden_size = args.local_hidden_size #visual encoder hidden vector
    g_hidden_size = args.global_hidden_size # global feature hidden vector


    # Global policy (option RL)
    g_policy = RL_Policy(map_observation_space.shape, envs.venv.action_space, g_action_space,
                         rgb_observation_space.shape,
                         hidden_size=l_hidden_size,
                         use_deterministic_local=args.use_deterministic_local,
			 device=args.device,
                         base_kwargs={'recurrent': args.use_recurrent_global,
                                      'hidden_size': g_hidden_size,
                                      'downscaling': args.global_downscaling
                                      }).to(device)

    g_agent = algo.PPO(g_policy, args.clip_param, args.ppo_epoch,
                       args.num_mini_batch, args.value_loss_coef, args.termination_loss_coef,
                       args.entropy_coef, lr=args.global_lr, eps=args.eps,
                       max_grad_norm=args.max_grad_norm)


    # Storage
    g_rollouts = GlobalRolloutStorage(args.num_global_steps,
                                          num_scenes, map_observation_space.shape,
                                          g_action_space, g_policy.rec_state_size, rgb_observation_space.shape,
                                          1).to(device)

    # Loading model

    if args.load_global != "0":
        print("Loading global {}".format(args.load_global))
        state_dict = torch.load(args.load_global,
                                map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict)
        torch.save(g_policy.state_dict(), os.path.join(log_dir, "model_best_gibson_version.global"), _use_new_zipfile_serialization=False)

    if not args.train_global:
        g_policy.eval()

    # Predict map from frame 1:
    #print(infos.size())
    #print(num_scenes)
    poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['sensor_pose'] for env_idx
         in range(num_scenes)])
    ).float().to(device)

    # Compute Global policy input
    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, 5, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)

    global_input[:, 0:5, :, :] = local_map.detach()
    #global_input[:, 4:, :, :] = nn.MaxPool2d(args.global_downscaling)(full_map)


    g_rollouts.obs[0].copy_(global_input)
    g_rollouts.rgb[0].copy_(obs)
    g_rollouts.extras[0].copy_(global_orientation)
    g_rollouts.option[0].fill_(1)
    
    current_option = np.ones((num_scenes,1))
    #g_value = torch.zeros(num_scenes,2).to(device)
    #g_termination = torch.zeros(num_scenes,2).to(device)
    #g_option = [0 for i in range(num_scenes)]
    #g_action = torch.zeros(num_scenes,3).to(device)
    #g_action_log_prob = torch.zeros(num_scenes,1).to(device)
    #g_rec_states = torch.zeros(num_scenes, g_hidden_size).to(device)

    # Run Policy
    

    g_value, g_termination, g_option, g_rec_states = \
        g_policy.predict_option_termination(
            g_rollouts.obs[0],
            g_rollouts.option[0],
            g_rollouts.rec_states[0],
            g_rollouts.rgb[0],
            g_rollouts.masks[0],
            extras=g_rollouts.extras[0],
            deterministic=False
        )


    g_action, g_action_log_prob, g_rec_states = \
        g_policy.act(
            g_rollouts.obs[0],
            g_rollouts.option[0],
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

    for ep_num in range(num_episodes):
        
        g_reward = torch.from_numpy(np.zeros((num_scenes))).float().to(device)
        g_reward_all = np.zeros((1,num_scenes))
        #change_goal_flag = [True for i in range(num_scenes)]
        local_step_count = [0 for i in range(num_scenes)]
        local_step_count_rot = [0 for i in range(num_scenes)]
        
        
        torch.set_grad_enabled(False)
        global_goals = [[0,0] for i in range(num_scenes)]
        frontier_goals = [[0,0] for i in range(num_scenes)]

        g_termination_all1 = [[g_termination[i,0].cpu().numpy()] for i in range(num_scenes)]
        g_termination_all2 = [[g_termination[i,1].cpu().numpy()] for i in range(num_scenes)]
        g_value_all1 = [[g_value[i,0].cpu().numpy()] for i in range(num_scenes)]
        g_value_all2 = [[g_value[i,1].cpu().numpy()] for i in range(num_scenes)]
        
        current_option_all = [[1] for i in range(num_scenes)]

        for step in range(args.max_episode_length):
            total_num_steps += 1

            g_step = (step // args.num_local_steps) % args.num_global_steps
            eval_g_step = step + 1

            # ------------------------------------------------------------------
            # option RL dicision making
            #print("local_step_count_rot")
            #print(local_step_count_rot)
            action = np.zeros(num_scenes)
            for e in range(num_scenes):
                change_goal = False
                if current_option[e] == 1: #rotate
                    if local_step_count[e] == 0: #reset rotation degree
                        cpu_actions = nn.Tanh()(g_action[e,0]).cpu().numpy() 
                        local_step_count_rot[e] = int(cpu_actions * 180 / 10)
                        local_step_count[e] = args.num_local_steps

                    if local_step_count_rot[e] == 1 and local_step_count[e] > 1:
                        local_step_count_rot[e] -= 2
                        action[e] = 1
                    elif local_step_count_rot[e] > 1 and local_step_count[e] > 1:
                        local_step_count_rot[e] -= 1
                        action[e] = 1
                    elif local_step_count_rot[e] == 1 and local_step_count[e] == 1:
                        local_step_count_rot[e] -= 1
                        action[e] = 1                            
                    
                    elif local_step_count_rot[e] == -1 and local_step_count[e] > 1:
                        local_step_count_rot[e] += 2
                        action[e] = 0
                    elif local_step_count_rot[e] < -1 and local_step_count[e] > 1:
                        local_step_count_rot[e] += 1
                        action[e] = 0
                    elif local_step_count_rot[e] == -1 and local_step_count[e] == 1:   
                        local_step_count_rot[e] += 1
                        action[e] = 0

                    local_step_count[e] -= 1

                else: #navigation
                    if local_step_count[e] == 0: #reset goal point after exploration
                        cpu_actions = nn.Sigmoid()(g_action[e,1:]).cpu().numpy()
                        global_goals[e] = [int(cpu_actions[0] * (local_w-1)), int(cpu_actions[1] * (local_h-1))]
                        #print("new goal points:")
                        #print(e)
                        #print(global_goals[e])
                        change_goal = True

                        frontier_map = local_map[e, -1, :, :].detach().cpu().numpy()
                        ind_r,ind_c = np.nonzero(frontier_map)
                        if ind_r.size == 0 or ind_c.size == 0:
                            ind_r,ind_c = np.array([int(planner_pose_inputs[e,1] * 100.0 / args.map_resolution)]),\
                                          np.array([int(planner_pose_inputs[e,0] * 100.0 / args.map_resolution)])
                        ind = np.stack((ind_r,ind_c),1)
                        dist = ind - np.array(global_goals[e])
                        dist = dist**2
                        dist = np.sum(dist,1)
                        f_ind = np.argmin(dist)
                        frontier_goals[e] = [ind_r[f_ind],ind_c[f_ind]]
                        print("new frontier points for " + str(e))
                        print((ind_r[f_ind],ind_c[f_ind]))
                        bad_frontier_map[e,0,ind_r[f_ind],ind_c[f_ind]] = 1
                        local_step_count[e] = args.num_local_steps
                    
                    local_step_count[e] -= 1

                    #print('local remaining steps')
                    #print(local_step_count[e])
                    
                    #global_goals[e] = global_goals[e][0]

                    # Get short term goal
                    planner_inputs = [{} for en in range(num_scenes)]
                    for en, p_input in enumerate(planner_inputs):
                        p_input['map_pred'] = local_map[en, 0, :, :].cpu().numpy()
                        p_input['exp_pred'] = local_map[en, 1, :, :].cpu().numpy()
                        p_input['pose_pred'] = planner_pose_inputs[en]
                        p_input['goal'] = global_goals[e]#frontier_goals[e]
                        p_input['goal_arbitrary'] = global_goals[e]
                        p_input['change_goal'] = change_goal
                        p_input['active'] = True if en==e else False

                    output = envs.get_short_term_goal(planner_inputs)

                    action_target = output[e, -1].long().to(device)
                    action[e] = action_target.cpu()
                    #if output[e,0] == True:
                    #    local_step_count_nav[e] = 0

            print("Timer for phase 2")
            timer.toc()
            timer.tic()
            # ------------------------------------------------------------------
            # Env step
            #print("step:")
            #print(current_option)
            action = torch.from_numpy(action)
            #print(action)
         
            obs, rew, done, infos = envs.step(action)

            g_masks = torch.FloatTensor([0 if x else 1
                                         for x in done]).to(device)

            envs.update_visualize(current_option)

            print("Timer for phase 3")
            timer.toc()
            timer.tic()

            # ------------------------------------------------------------------
            # Reinitialize variables when episode ends
            if step == args.max_episode_length - 1:  # Last episode step
                init_map_and_pose()
            # ------------------------------------------------------------------

            # Update the occupancy and exploration maps
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
                local_map[env_idx, 0, :, :] = env_gt_fp_projs
                local_map[env_idx, 1, :, :] = env_gt_fp_explored
                local_pose[env_idx] = env_poses


            locs = local_pose.cpu().numpy()
            planner_pose_inputs[:, :3] = locs + origins
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
                current_option_all[e].append(current_option[e,0].copy())
                g_termination_all1[e].append(g_termination[e,0].cpu().numpy().copy())
                g_termination_all2[e].append(g_termination[e,1].cpu().numpy().copy())
                g_value_all1[e].append(g_value[e,0].cpu().numpy().copy())
                g_value_all2[e].append(g_value[e,1].cpu().numpy().copy())


            print("Timer for phase 5")
            timer.toc()
            timer.tic()
            # ------------------------------------------------------------------
            

            # Add samples to global policy storage

            print("local step count")
            print(local_step_count)

            # Sample action from global policy
            local_done = True
            for e in range(num_scenes):
                if local_step_count[e] > 0:
                    local_done = False

            if local_done:# Global step


                #Update frontier map
                locs = local_pose.cpu().numpy()
                for e in range(num_scenes):
                    global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
                    local_map[e, 4] = get_frontier_map(local_map[e, 0, :, :].detach(), \
                        local_map[e, 1, :, :].detach(), local_map[e, 3, :, :].detach(), bad_frontier_map[e,0,:,:])

                global_input = local_map
                #global_input[:, 4:, :, :] = \
                #    nn.MaxPool2d(args.global_downscaling)(full_map)

                # Get exploration reward and metrics
                g_reward = torch.from_numpy(np.asarray(
                    [infos[env_idx]['exp_reward'] for env_idx
                     in range(num_scenes)])
                ).float().to(device)

                print("Rewards:")
                print(g_reward)  

                g_reward_all = np.concatenate((g_reward_all, np.expand_dims(g_reward.cpu().numpy(), axis=0)), axis=0)

                # Plot curves
                for e in range(num_scenes):

                    ax[0, e].clear()
                    ax[1, e].clear()
                    ax[2, e].clear()
                    ax[3, e].clear()

                    ax[0, e].plot(g_termination_all1[e])
                    ax[0, e].plot(g_termination_all2[e])
                    ax[1, e].plot(g_value_all1[e])
                    ax[1, e].plot(g_value_all2[e])
                    ax[2, e].plot(current_option_all[e])

                    base = 0.5
                    for i in range(1, len(current_option_all[e]), args.num_local_steps):

                        if current_option_all[e][i] == 0:
                            ax[3, e].axvspan(base, base + 1, facecolor='b', alpha=0.3)
                        else:
                            ax[3, e].axvspan(base, base + 1, facecolor='r', alpha=0.3)
                        base += 1

                    ax[3, e].plot(g_reward_all[:, e])

                ax[0, num_scenes].plot(g_termination_losses)
                ax[1, num_scenes].plot(g_value_losses)
                ax[2, num_scenes].plot(g_dist_entropies)
                ax[3, num_scenes].plot(g_action_losses)

                plt.gcf().canvas.flush_events()
                fig.canvas.start_event_loop(0.001)
                plt.pause(0.001)

                g_process_rewards += g_reward.cpu().numpy()
                g_total_rewards = g_process_rewards * \
                                  (1 - g_masks.cpu().numpy())
                g_process_rewards *= g_masks.cpu().numpy()
                per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

                if np.sum(g_total_rewards) != 0:
                    for tr in g_total_rewards:
                        g_episode_rewards.append(tr) if tr != 0 else None

                g_rollouts.insert(
                    global_input, obs, g_rec_states,
                    g_action, g_action_log_prob, g_value, current_option, g_termination,
                    g_reward.detach(), g_masks, global_orientation
                )

                g_value, g_termination, g_option, g_rec_states = \
                    g_policy.predict_option_termination(
                        g_rollouts.obs[g_step + 1],
                        current_option,
                        g_rollouts.rec_states[g_step + 1],
                        g_rollouts.rgb[g_step + 1],
                        g_rollouts.masks[g_step + 1],
                        extras=g_rollouts.extras[g_step + 1],
                        deterministic=False
                    )

                
                for e in range(num_scenes):
                    print("Beta: " + str(e))
                    print(g_termination[e,current_option[e]])  
                    if bool(Bernoulli(g_termination[e,current_option[e]]).sample().item()):
                        current_option[e] = np.random.choice(2) if np.random.rand() < g_policy.epsilon() else g_option[e]
                

                g_action, g_action_log_prob, g_rec_states = \
                    g_policy.act(
                        g_rollouts.obs[g_step + 1],
                        current_option,
                        g_rollouts.rec_states[g_step + 1],
                        g_rollouts.rgb[g_step + 1],
                        g_rollouts.masks[g_step + 1],
                        extras=g_rollouts.extras[g_step + 1],
                        deterministic=False
                    )



                print("g_value")
                print(g_value)

                print("Timer for phase 5")
                timer.toc()
                timer.tic()
            #g_reward = torch.from_numpy(np.zeros((num_scenes))).float().to(device)
            #g_masks = torch.ones(num_scenes).float().to(device)
            # ------------------------------------------------------------------

                ### TRAINING
                torch.set_grad_enabled(True)

                
                # Train Global Policy
                if g_step % args.num_global_steps == args.num_global_steps - 1:
                    print("#######Training Global Policy#######")
                    
                    g_next_value, g_terminations = g_policy.get_value(
                        g_rollouts.obs[-1],
                        g_rollouts.rec_states[-1],
                        g_rollouts.rgb[-1],
                        g_rollouts.masks[-1],
                        extras=g_rollouts.extras[-1]
                    )

                    g_rollouts.compute_returns(args.gamma, g_next_value.detach(), g_terminations.detach())
                    g_value_loss, g_termination_loss, g_action_loss, g_dist_entropy = \
                        g_agent.update(g_rollouts)
                    g_value_losses.append(g_value_loss * args.value_loss_coef)
                    g_termination_losses.append(g_termination_loss * args.termination_loss_coef)
                    g_action_losses.append(g_action_loss)
                    g_dist_entropies.append(g_dist_entropy * args.entropy_coef)
                    print("#######Finish Training Global Policy#######")
                    g_rollouts.after_update()
                # ------------------------------------------------------------------

                # Finish Training
                torch.set_grad_enabled(False)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Logging
            if total_num_steps % args.log_interval == 0:
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
                            np.mean(per_step_g_rewards),
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
            # Save best models
            if (total_num_steps * num_scenes) % args.save_interval < \
                    num_scenes:

                # Save Global Policy Model
                if len(g_episode_rewards) >= 100 and \
                        (np.mean(g_episode_rewards) >= best_g_reward) \
                        and not args.eval:
                    torch.save(g_policy.state_dict(),
                               os.path.join(log_dir, "model_best.global"))
                    best_g_reward = np.mean(g_episode_rewards)

            # Save periodic models
            if (total_num_steps * num_scenes) % args.save_periodic < \
                    num_scenes:
                step = total_num_steps * num_scenes

                if args.train_global:
                    torch.save(g_policy.state_dict(),
                               os.path.join(dump_dir,
                                            "periodic_{}.global".format(step)))
            # ------------------------------------------------------------------

    # Print and save model performance numbers during evaluation
    if args.eval:
        logfile = open("{}/explored_area.txt".format(dump_dir), "w+")
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


if __name__ == "__main__":
    main()
