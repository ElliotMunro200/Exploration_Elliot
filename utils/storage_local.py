# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py

from collections import namedtuple

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):  # e.g. _flatten_helper(T, N, obs), where obs.shape = (T,2,map.size)
    return _tensor.view(T * N, *_tensor.size()[
                                2:])  # reshapes the tensor to size (80=40*2, map.size), flattening the first 2 dims.


class RolloutStorage(object):

    def __init__(self, num_steps, num_processes, obs_shape, action_space_box,
                 rec_state_size, rgb_size):

        self.n_actions1 = 1
        action_type1 = torch.float32

        self.n_actions2 = action_space_box.shape[0]
        action_type2 = torch.float32

        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rgb = torch.zeros(num_steps + 1, num_processes, *rgb_size)
        self.rec_states = torch.zeros(num_steps + 1, num_processes,
                                      rec_state_size)
        self.rewards = torch.zeros(num_steps, num_processes)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 2)
        self.returns = torch.zeros(num_steps + 1, num_processes)
        self.option = torch.zeros(num_steps + 1, num_processes)
        self.is_intermediate = torch.zeros(num_steps + 1, num_processes)
        self.terminations = torch.zeros(num_steps + 1, num_processes, 2)
        self.action_log_probs = torch.zeros(num_steps, num_processes)
        self.actions_discrete = torch.zeros((num_steps, num_processes, self.n_actions1),
                                            dtype=action_type1)
        self.actions_box = torch.zeros((num_steps, num_processes, self.n_actions2),
                                       dtype=action_type2)
        self.masks = torch.ones(num_steps + 1, num_processes)

        self.num_steps = num_steps
        self.num_processes = num_processes
        self.step = 0
        self.has_extras = False
        self.extras_size = None

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rec_states = self.rec_states.to(device)
        self.rgb = self.rgb.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.option = self.option.to(device)
        self.terminations = self.terminations.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions_discrete = self.actions_discrete.to(device)
        self.actions_box = self.actions_box.to(device)
        self.masks = self.masks.to(device)
        if self.has_extras:
            self.extras = self.extras.to(device)
        return self

    def insert(self, obs, rgb, rec_states, actions, action_log_probs, value_preds, option, terminations,
               rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.rgb[self.step + 1].copy_(rgb)
        self.rec_states[self.step + 1].copy_(rec_states)

        for e in range(len(option)):
            if option[e] == 1:  # rotate
                self.actions_discrete[self.step, e].copy_(actions[e, 0].view(-1))
            else:
                self.actions_box[self.step, e].copy_(actions[e, 1:].view(-1))

        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.option[self.step].copy_(torch.from_numpy(option).view(-1))
        self.terminations[self.step].copy_(terminations)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(
            self):  # obs, rec_states, rgb, masks, extras [0] all adopt last of previous episode. should get overwritten if change episode? It does.
        self.obs[0].copy_(self.obs[-1])
        self.rec_states[0].copy_(self.rec_states[-1])
        self.rgb[0].copy_(self.rgb[-1])
        self.masks[0].copy_(self.masks[-1])
        if self.has_extras:
            self.extras[0].copy_(self.extras[-1])

    def compute_returns(self, gamma, next_value, next_terminations):
        # collect rewards of path-planning steps
        for e in range(self.num_processes):
            self.returns[-1, e] = ((1 - next_terminations[e, int(self.option[-1, e])].detach()) * \
                                   next_value[e, int(self.option[-1, e])].detach() + \
                                   next_terminations[e, int(self.option[-1, e])].detach() * \
                                   next_value[e].max(dim=-1)[0].detach())

        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * gamma \
                                 * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch):

        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps,
                      num_mini_batch))

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=False)

        for indices in sampler:
            yield {
                'obs': self.obs[:-1].view(-1, *self.obs.size()[2:])[indices],
                'option': self.option[:-1].view(-1)[indices],
                'rec_states': self.rec_states[:-1].view(-1, self.rec_states.size(-1))[indices],
                'rgb': self.rgb[:-1].view(-1, *self.rgb.size()[2:])[indices],
                'actions_discrete': self.actions_discrete.view(-1, self.n_actions1)[indices],
                'actions_box': self.actions_box.view(-1, self.n_actions2)[indices],
                'value_preds': self.value_preds[:-1].view(-1)[indices],
                'terminations': self.terminations[:-1].view(-1)[indices],
                'returns': self.returns[:-1].view(-1)[indices],
                'masks': self.masks[:-1].view(-1)[indices],
                'old_action_log_probs': self.action_log_probs.view(-1)[indices],
                'adv_targ': advantages.view(-1)[indices],
                'extras': self.extras[:-1].view(-1, self.extras_size)[indices] \
                    if self.has_extras else None,
            }

    def recurrent_generator(self, advantages, num_mini_batch):

        num_processes = self.rewards.size(1)  # 4
        assert num_processes >= num_mini_batch, (  # 4 >= 2 is True
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch  # 2 = 4/2
        perm = torch.randperm(num_processes)  # random permutation of integers: 0-n-1 e.g. 4 => tensor([2,1,0,3])
        T, N = self.num_steps, num_envs_per_batch  # num_global_steps = num macros = 40 for Gibson, 2.

        for start_ind in range(0, num_processes, num_envs_per_batch):  # range(0,4,2) => [0,2]

            obs = []
            option = []
            is_intermediate = []
            rec_states = []
            rgb = []
            actions_discrete = []
            actions_box = []
            value_preds = []
            terminations = []
            returns = []
            masks = []
            old_action_log_probs = []
            adv_targ = []
            if self.has_extras:
                extras = []

            for offset in range(num_envs_per_batch):  # range(2) => [0,1]

                ind = perm[start_ind + offset]
                obs.append(self.obs[:-1, ind])  # all except for last of obs from scene/process = 2.
                option.append(self.option[:-1, ind])
                is_intermediate.append(self.is_intermediate[:-1, ind])
                rec_states.append(self.rec_states[0:1, ind])
                rgb.append(self.rgb[:-1, ind])
                actions_discrete.append(self.actions_discrete[:, ind])
                actions_box.append(self.actions_box[:, ind])
                value_preds.append(self.value_preds[:-1, ind])
                terminations.append(self.terminations[:-1, ind])
                returns.append(self.returns[:-1, ind])
                masks.append(self.masks[:-1, ind])
                old_action_log_probs.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                if self.has_extras:
                    extras.append(self.extras[:-1, ind])

            # These are all tensors of size (T, N, ...)
            # obs was a list of two tensors of size (T, map.size=5/6, 512, 512). torch.stack() concatenates a sequence of tensors along a new dimension.
            # Effectively, this will stack/concat the 2 tensors for each scene back into one tensor of shape = (T,2,map.size).
            obs = torch.stack(obs, 1)
            option = torch.stack(option, 1)
            is_intermediate = torch.stack(is_intermediate, 1)
            rgb = torch.stack(rgb, 1)
            actions_discrete = torch.stack(actions_discrete, 1)
            actions_box = torch.stack(actions_box, 1)
            value_preds = torch.stack(value_preds, 1)
            terminations = torch.stack(terminations, 1)
            returns = torch.stack(returns, 1)
            masks = torch.stack(masks, 1)
            old_action_log_probs = torch.stack(old_action_log_probs, 1)
            adv_targ = torch.stack(adv_targ, 1)
            if self.has_extras:
                extras = torch.stack(extras, 1)

            yield {
                'obs': _flatten_helper(T, N, obs),  # obs is of shape (T,2,map.size).
                'option': _flatten_helper(T, N, option),
                'is_intermediate': _flatten_helper(T, N, is_intermediate),
                'rgb': _flatten_helper(T, N, rgb),
                'actions_discrete': _flatten_helper(T, N, actions_discrete),
                'actions_box': _flatten_helper(T, N, actions_box),
                'value_preds': _flatten_helper(T, N, value_preds),
                'terminations': _flatten_helper(T, N, terminations),
                'returns': _flatten_helper(T, N, returns),
                'masks': _flatten_helper(T, N, masks),
                'old_action_log_probs': _flatten_helper(T, N, old_action_log_probs),
                'adv_targ': _flatten_helper(T, N, adv_targ),
                'extras': _flatten_helper(T, N, extras) if self.has_extras else None,
                'rec_states': torch.stack(rec_states, 1).view(N, -1),
            }


class GlobalRolloutStorage(RolloutStorage):

    def __init__(self, num_steps, num_processes, obs_shape, action_space_box,
                 rec_state_size, rgb_size, extras_size):
        super(GlobalRolloutStorage, self).__init__(num_steps, num_processes,
                                                   obs_shape, action_space_box, rec_state_size, rgb_size)
        self.extras = torch.zeros((num_steps + 1, num_processes, extras_size),
                                  dtype=torch.long)
        self.has_extras = True
        self.extras_size = extras_size

    def insert(self, obs, rgb, rec_states, actions, action_log_probs, value_preds, option, terminations, rewards, masks,
               extras):
        self.extras[self.step + 1].copy_(extras)
        super(GlobalRolloutStorage, self).insert(obs, rgb, rec_states, actions,
                                                 action_log_probs, value_preds, option, terminations, rewards, masks)


Datapoint = namedtuple('Datapoint',
                       ('input', 'target'))


class FIFOMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a datapoint."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Datapoint(*args)
        if self.position == 0:
            x = self.memory[0][0]
            y = self.memory[0][1]
            self.batch_in_sizes = {}
            self.n_inputs = len(x)
            for dim in range(len(x)):
                self.batch_in_sizes[dim] = x[dim].size()

            self.batch_out_sizes = {}
            self.n_outputs = len(y)
            for dim in range(len(y)):
                self.batch_out_sizes[dim] = y[dim].size()

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Samples a batch"""

        batch = {}
        inputs = []
        outputs = []

        for dim in range(self.n_inputs):
            inputs.append(torch.cat(batch_size *
                                    [torch.zeros(
                                        self.batch_in_sizes[dim]
                                    ).unsqueeze(0)]))

        for dim in range(self.n_outputs):
            outputs.append(torch.cat(batch_size *
                                     [torch.zeros(
                                         self.batch_out_sizes[dim]
                                     ).unsqueeze(0)]))

        indices = np.random.choice(len(self.memory), batch_size, replace=False)

        count = 0
        for i in indices:
            x = self.memory[i][0]
            y = self.memory[i][1]

            for dim in range(len(x)):
                inputs[dim][count] = x[dim]

            for dim in range(len(y)):
                outputs[dim][count] = y[dim]

            count += 1

        return (inputs, outputs)

    def __len__(self):
        return len(self.memory)
