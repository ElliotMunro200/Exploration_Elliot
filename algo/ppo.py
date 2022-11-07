# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO():

    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        termination_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=False):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.termination_loss_coef = termination_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
            actor_critic.parameters()), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1].gather(2,rollouts.option[:-1].unsqueeze(-1).long()).detach().squeeze(-1)
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)

        value_loss_epoch = 0
        termination_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):

            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                        advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                        advantages, self.num_mini_batch)

            for sample in data_generator:

                value_preds = sample['value_preds']
                returns = sample['returns']
                adv_targ = sample['adv_targ']

                # Reshape to do in a single forward pass for all steps
                values, terminations, action_log_probs, dist_entropy, _ = \
                        self.actor_critic.evaluate_actions(
                            sample['obs'], sample['option'], sample['rec_states'], sample['rgb'],
                            sample['masks'], sample['actions_discrete'], sample['actions_box'],
                            extras=sample['extras']
                        )
                #termination_loss = terminations.gather(1,sample['option'].long().view(-1,1)) * \
                #                  (values.gather(1,sample['option'].long().view(-1,1)).detach() - values.max(dim=-1)[0].detach().unsqueeze(-1) + 0.01) * \
                #                  sample['masks'].unsqueeze(-1)

                termination_loss = (terminations.gather(1,sample['option'].long().view(-1,1)) * \
                                  (values.gather(1,sample['option'].long().view(-1,1)).detach() - values.mean(dim=-1).detach().unsqueeze(-1)) * \
                                  sample['masks'].unsqueeze(-1)).mean()

                '''             
                print("termination_loss")
                print(terminations)
                print(terminations.gather(1,sample['option'].long().view(-1,1)))
                print(values.gather(1,sample['option'].long().view(-1,1)).detach() - values.max(dim=-1)[0].detach().unsqueeze(-1) + 0.01)
                print(sample['masks'])
                print(termination_loss.mean())
                '''

                ratio = torch.exp(action_log_probs -
                                    sample['old_action_log_probs'])

                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                        1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                print("action loss:")
                print(action_loss)        
                print("term loss:")
                print(termination_loss)   
        
                #action_loss += termination_loss.mul(1).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds.gather(1,sample['option'].long().view(-1,1)) + \
                                        (values.gather(1,sample['option'].long().view(-1,1)) - value_preds.gather(1,sample['option'].long().view(-1,1))).clamp(
                                            -self.clip_param, self.clip_param)
                    value_losses = (values.gather(1,sample['option'].long().view(-1,1)) - returns).pow(2)
                    value_losses_clipped = (value_pred_clipped
                                            - returns).pow(2)
                    value_loss = .5 * torch.max(value_losses,
                                                value_losses_clipped).mean()
                else:
                    value_loss = (returns - values.gather(1,sample['option'].long().view(-1,1))).pow(2).mean()
                    #value_loss = 0.5 * (1 - sample['option'].long().view(-1,1) - values.gather(1,sample['option'].long().view(-1,1))).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + self.termination_loss_coef * termination_loss + action_loss -
                        dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                termination_loss_epoch += termination_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        termination_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, termination_loss_epoch, action_loss_epoch, dist_entropy_epoch
