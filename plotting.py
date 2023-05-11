def plotting(num_scenes, length):
    # plots
    plt.pause(0.001)

    g_termination_all1 = np.ones((num_scenes, length))
    g_termination_all2 = np.zeros((num_scenes, length))
    g_value_all1 = np.ones((num_scenes, length))
    g_value_all2 = np.zeros((num_scenes, length))
    current_option_all = np.ones((num_scenes, length))

    g_reward_all = np.ones((length, num_scenes))
    g_value_losses = np.ones((num_scenes, length))
    g_termination_losses = np.ones((num_scenes, length))
    g_action_losses = np.ones((num_scenes, length))
    g_dist_entropies = np.ones((num_scenes, length))
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
        for i in range(1, len(current_option_all[e]), length):

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    plt.ion()
    num_scenes = 1
    fig, ax = plt.subplots(4, num_scenes + 1, figsize=(10, 2.5), facecolor="whitesmoke")
    for i in range(100):
        plotting(num_scenes=num_scenes, length=40)