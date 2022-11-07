import time
from collections import deque

import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from torch.distributions import Bernoulli
import torch
import torch.nn as nn
from torch.nn import functional as F
from pytictoc import TicToc

import gym
import logging
from arguments import get_args
from env import make_vec_envs
from utils.storage import GlobalRolloutStorage #, FIFOMemory
#from utils.optimization import get_optimizer
from model import RL_Policy 

import algo

import sys
import matplotlib
import time
from scipy.io import savemat

from scipy.ndimage.morphology import binary_dilation
from skimage import morphology

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
import matplotlib.pyplot as plt

matplotlib.use('tkagg')

plt.ion()
#plt.pause(0.001)


args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

counter = 0

def get_frontier_map(map_gt, explored_gt, visited_gt, bad_frontier_map):

    map_gt = map_gt.detach().cpu().numpy()
    explored_gt = explored_gt.detach().cpu().numpy()
    visited_gt = visited_gt.detach().cpu().numpy()

    selem = morphology.disk(5)

    '''
    contour = binary_dilation(explored_gt==0, selem) & (explored_gt==1)
    contour = contour & (binary_dilation(map_gt, selem)==0)
    contour = contour & (binary_dilation(visited_gt, morphology.disk(10))==0)
    contour = contour & (binary_dilation(bad_frontier_map, morphology.disk(10))==0)
    contour = morphology.remove_small_objects(contour, 2)
    '''
    contour = binary_dilation(explored_gt==0, selem) & (explored_gt==1)
    contour = contour & (binary_dilation(map_gt, selem)==0)
    contour = contour & (binary_dilation(visited_gt, morphology.disk(1))==0)
    contour = contour & (binary_dilation(bad_frontier_map, morphology.disk(1))==0)
    contour = morphology.remove_small_objects(contour, 2)    

    

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

    #plots
    plt.ion()
    fig, ax = plt.subplots(3, num_scenes, figsize=(10, 2.5), facecolor="whitesmoke")
    plt.pause(0.001)

    if args.eval:
        traj_lengths = args.max_episode_length // args.num_local_steps
        explored_area_log = np.zeros((num_scenes, num_episodes, traj_lengths))
        explored_ratio_log = np.zeros((num_scenes, num_episodes, traj_lengths))

    g_episode_rewards = deque(maxlen=1000)
    g_value_losses = deque(maxlen=1000)
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
    ### Full map consists of 4 channels containing the following:
    ### 1. Obstacle Map
    ### 2. Exploread Area
    ### 3. Current Agent Location
    ### 4. Past Agent Locations
    ### 5. Frontier Map

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
                         base_kwargs={'recurrent': args.use_recurrent_global,
                                      'hidden_size': g_hidden_size,
                                      'downscaling': args.global_downscaling
                                      }).to(device)


    # Loading model

    if args.load_global != "0":
        print("Loading global {}".format(args.load_global))
        state_dict = torch.load(args.load_global,
                                map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict)

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
    global_input = torch.zeros(num_scenes, 5, local_w, local_h).to(device)
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)

    global_input[:, 0:5, :, :] = local_map.detach()
    #global_input[:, 4:, :, :] = nn.MaxPool2d(args.global_downscaling)(full_map)


    if args.disabled == 1:
        current_option = np.zeros((num_scenes,1)) #initial options
    else:
        current_option = np.ones((num_scenes,1)) #initial options

    g_value = torch.zeros(num_scenes,2).to(device)
    g_termination = torch.zeros(num_scenes,2).to(device)
    g_option = [0 for i in range(num_scenes)]
    g_action = torch.zeros(num_scenes,3).to(device)
    g_action_log_prob = torch.zeros(num_scenes,1).to(device)
    g_rec_states = torch.zeros(num_scenes, g_hidden_size).to(device)

    # Run Policy 
    for e in range(num_scenes):

        g_value[e], g_termination[e], g_option[e], g_rec_states[e] = \
            g_policy.predict_option_termination(
                global_input[e].unsqueeze(0),
                current_option[e],
                g_rec_states[e].unsqueeze(0),
                obs[e].unsqueeze(0),
                g_masks[e].unsqueeze(0),
                extras=global_orientation[e].to(device),
                deterministic=False
            )

        g_action[e], g_action_log_prob[e], g_rec_states[e] = \
            g_policy.act(
                global_input[e].unsqueeze(0),
                current_option[e],
                g_rec_states[e].unsqueeze(0),
                obs[e].unsqueeze(0),
                g_masks[e].unsqueeze(0),
                extras=global_orientation[e].to(device),
                deterministic=False
            )

    start = time.time()
    
    total_num_steps = -1
    
    
    torch.set_grad_enabled(False)

    print("Timer for phase 1")
    timer.toc()
    timer.tic()

    for ep_num in range(56):

        #change_goal_flag = [True for i in range(num_scenes)]
        local_step_count = [0 for i in range(num_scenes)]
        option_counter = [0 for i in range(num_scenes)]
        
        
        torch.set_grad_enabled(False)
        global_goals = [[0,0] for i in range(num_scenes)]
        frontier_goals = [[0,0] for i in range(num_scenes)]


        g_termination_all1 = [[g_termination[i,0].cpu().numpy()] for i in range(num_scenes)]
        g_termination_all2 = [[g_termination[i,1].cpu().numpy()] for i in range(num_scenes)]
        g_value_all1 = [[g_value[i,0].cpu().numpy()] for i in range(num_scenes)]
        g_value_all2 = [[g_value[i,1].cpu().numpy()] for i in range(num_scenes)]
        
        current_option_all = [[] for i in range(num_scenes)]

        for step in range(args.max_episode_length):
            total_num_steps += 1

            g_step = step % args.num_global_steps
            eval_g_step = step + 1

            # ------------------------------------------------------------------
            # option RL dicision making

            action = np.zeros(num_scenes)
            for e in range(num_scenes):
                change_goal = False
                if current_option[e] == 1: #rotation
                    if local_step_count[e] == 0: #reset rotation degree
                        local_step_count[e] = 36

                    if local_step_count[e] > 0:
                        local_step_count[e] -= 1
                        action[e] = 1           
                    

                else: #navigation
                    if local_step_count[e] == 0: #reset goal point after exploration
                        cpu_actions = nn.Sigmoid()(g_action[e,1:]).cpu().numpy() 
                        global_goals[e] = [int(cpu_actions[0] * local_w), int(cpu_actions[1] * local_h)]
                        #print("new goal points:")
                        #print(e)
                        #print(global_goals[e])
                        change_goal = True
                        
                        frontier_map = local_map[e, -1, :, :].detach().cpu().numpy()
                        ind_r,ind_c = np.nonzero(frontier_map)
                        if ind_r.size == 0 or ind_c.size == 0:
                            print("No frontier found for env" + str(e))
                            ind_r,ind_c = np.array([int(planner_pose_inputs[e,1] * 100.0 / args.map_resolution)]),\
                                          np.array([int(planner_pose_inputs[e,0] * 100.0 / args.map_resolution)])
                        ind = np.stack((ind_r,ind_c),1)
                        dist = ind - np.array(global_goals[e])
                        dist = dist**2
                        dist = np.sum(dist,1)
                        f_ind = np.argmin(dist)
                        frontier_goals[e] = [ind_r[f_ind],ind_c[f_ind]]
                        #print("new frontier points:")
                        #print((ind_r[f_ind],ind_c[f_ind]))
                        bad_frontier_map[e,0,min(ind_r[f_ind],full_h-1),min(ind_c[f_ind],full_w-1)] = 1
                        
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
                        p_input['goal'] = frontier_goals[e]#global_goals[e]
                        p_input['goal_arbitrary'] = global_goals[e]
                        p_input['change_goal'] = change_goal
                        p_input['active'] = True if en==e else False

                    output = envs.get_short_term_goal(planner_inputs)
                    frontier_goals[e] = [int(output[e, 1].cpu().numpy()),int(output[e, 2].cpu().numpy())]
                    action_target = output[e, -1].long().to(device)
                    
                    action[e] = action_target.cpu()
                    if output[e,0] == True:
                        local_step_count[e] = 0

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
            print("Timer for phase 3")
            timer.toc()
            timer.tic()
            # ------------------------------------------------------------------


            #for e in range(num_scenes):
            #plt.plot(g_reward_all[:,1])
            #plt.gcf().canvas.flush_events()
            #plt.pause(0.1)

            #for e in range(num_scenes):
            #    current_option_all[e].append(current_option[e,0].copy())


            print("Timer for phase 4")
            timer.toc()
            timer.tic()

            print("local_step_count")
            print(local_step_count)

            # Sample action from global policy
            for e in range(num_scenes):
                if local_step_count[e] == 0:


                    local_map[e, 4,:] = get_frontier_map(local_map[e, 0, :, :].detach(), \
                                            local_map[e, 1, :, :].detach(), local_map[e, 3, :, :].detach(), bad_frontier_map[e,0,:,:])

                    locs[e] = local_pose[e].cpu().numpy()

                    global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
                    global_input[e] = local_map[e]

                    g_value[e], g_termination[e], g_option[e], g_rec_states[e] = \
                        g_policy.predict_option_termination(
                            global_input[e].unsqueeze(0),
                            current_option[e],
                            g_rec_states[e].unsqueeze(0),
                            obs[e].unsqueeze(0),
                            g_masks[e].unsqueeze(0),
                            extras=global_orientation[e].to(device),
                            deterministic=False
                        )

                            
                    # Rotate randomly by a given probability

                    current_option[e] = 1 if np.random.binomial(1, args.rot_prob) else 0

                        
                    if True:
                        current_option_all[e].append(current_option[e,0].copy())
                        #g_termination_all1[e].append(g_termination[e,0].cpu().numpy().copy())
                        #g_termination_all2[e].append(g_termination[e,1].cpu().numpy().copy())
                        #g_value_all1[e].append(g_value[e,0].cpu().numpy().copy())
                        #g_value_all2[e].append(g_value[e,1].cpu().numpy().copy())
                        ax[0,e].clear()
                        ax[1,e].clear()
                        ax[2,e].clear()
                        #ax[0,e].plot(g_termination_all1[e])
                        #ax[0,e].plot(g_termination_all2[e])
                        #ax[1,e].plot(g_value_all1[e])
                        #ax[1,e].plot(g_value_all2[e])
                        ax[2,e].plot(current_option_all[e])
                        #ax[2,e].imshow(local_map[e, 4,:].cpu().numpy())
                        plt.gcf().canvas.flush_events()
                        fig.canvas.start_event_loop(0.001)
                        plt.pause(0.001)


                    g_action[e], g_action_log_prob[e], g_rec_states[e] = \
                        g_policy.act(
                            global_input[e].unsqueeze(0),
                            current_option[e],
                            g_rec_states[e].unsqueeze(0),
                            obs[e].unsqueeze(0),
                            g_masks[e].unsqueeze(0),
                            extras=global_orientation[e].to(device),
                            deterministic=False
                        )

                    #print("g_value")
                    #print(g_value[e])
                    #print("g_termination")
                    #print(g_termination[e])
            print("Timer for phase 5")
            timer.toc()
            timer.tic()
            #g_reward = torch.from_numpy(np.zeros((num_scenes))).float().to(device)
            #g_masks = torch.ones(num_scenes).float().to(device)
 
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


                #print(log)
                logging.info(log)
            # ------------------------------------------------------------------
        savemat("exp_data/current_option_all" +str(ep_num)+ ".mat", {"current_option_all": current_option_all})
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
