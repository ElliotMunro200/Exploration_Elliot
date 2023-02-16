import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np
from math import exp

from utils.distributions import Categorical, DiagGaussian
from utils.model import get_grid, ChannelPool, Flatten, NNBase

from RocNet.data import QuadTree
from RocNet.util import get_args, get_quad_feas
from RocNet.ROctNetmodel import ROctEncoder, BoxEncoder, TreeClassifier, encode_structure_fold
from RocNet.torchfoldext import FoldExt


# Policy model code
class Global_Policy(NNBase):

    def __init__(self, input_shape, recurrent=True, hidden_size=512,
                 downscaling=1):
        super(Global_Policy, self).__init__(recurrent, hidden_size,
                                            hidden_size)

        out_size = int(input_shape[1] / 32. * input_shape[2] / 32.) #=256, input = (5/6, 512, 512) 


        config = get_args()
        config.box_code_size = 32
        config.feature_size = 512
        config.hidden_size = 1024
        config.show_log_every = 1
        config.n_class = 10
        config.save_log = False
        config.save_log_every = 3
        config.save_snapshot = True
        config.save_snapshot_every = 1
        config.save_snapshot = 'snapshot'
        config.no_plot = False
        config.no_cuda = False
        config.cuda = not config.no_cuda
        config.gpu = 0
        config.data_path = 'data'
        config.save_path = 'models'

        config.batch_size = 100
        incre = 1000
        config.epochs = 50
        config.num_maps = input_shape[0]
        

        self.encoder = ROctEncoder(config)
        self.box_encoder = BoxEncoder(num_maps=config.num_maps, input_size = config.box_code_size, feature_size = config.feature_size)
        self.tree_classifier = TreeClassifier(feature_size = config.feature_size, hidden_size = config.hidden_size)

        self.linear1 = nn.Linear(config.feature_size + 8 + 512, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.orientation_emb = nn.Embedding(72, 8)
        self.train()

    def forward(self, inputs, rnn_hxs, visual_feature, masks, extras):

        enc_fold = FoldExt(cuda=True)
        enc_fold_nodes = []     # list of fold nodes for encoding

        '''
      
        # Collect computation nodes recursively from encoding process
        for example in inputs:
            fea,op = get_quad_feas(example,32)
            #print(fea.size())
            tree = QuadTree(fea,op.unsqueeze(0))
            enc_fold_nodes.append(encode_structure_fold(enc_fold, tree))
        '''


        # Apply the computations on the encoder model
        #set_trace()
        #x = enc_fold.apply(self.encoder, [enc_fold_nodes])
        #x = x[0]

        x = self.tree_classifier(self.box_encoder(inputs))

        orientation_emb = self.orientation_emb(extras).squeeze(1)
        x = torch.cat((x, orientation_emb, visual_feature), 1)
        
        
        x = nn.ReLU()(self.linear1(x))
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = nn.ReLU()(self.linear2(x))

        return  x, rnn_hxs


# Visual Encoder (for RGB) model code
class Visual_Encoder(NNBase):

    def __init__(self, input_shape, recurrent=False,
                 hidden_size=512, deterministic=False):

        super(Visual_Encoder, self).__init__(recurrent, hidden_size,
                                              hidden_size)

        self.deterministic = deterministic
        self.dropout = 0.5

        resnet = models.resnet18(pretrained=True)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])

        # Extra convolution layer
        self.conv = nn.Sequential(*filter(bool, [
            nn.Conv2d(512, 64, (1, 1), stride=(1, 1)),
            nn.ReLU()
        ]))

        # convolution output size
        input_test = torch.randn(1, 3, input_shape[1], input_shape[2])
        conv_output = self.conv(self.resnet_l5(input_test))
        self.conv_output_size = conv_output.view(-1).size(0)

        # projection layers
        self.proj1 = nn.Linear(self.conv_output_size, hidden_size)
        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)
        self.linear = nn.Linear(hidden_size, hidden_size)

        self.train()

    def forward(self, rgb, masks):

        resnet_output = self.resnet_l5(rgb[:, :3, :, :])
        conv_output = self.conv(resnet_output)

        proj1 = nn.ReLU()(self.proj1(conv_output.view(
            -1, self.conv_output_size)))
        if self.dropout > 0:
            x = self.dropout1(proj1)

        return x


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Policy(nn.Module):

    def __init__(self, obs_shape, action_space_discrete, action_space_box,
                  observation_space_shape, hidden_size, use_deterministic_local, device, model_type=0, base_kwargs=None):

        super(RL_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        
        if model_type == 0:
            self.visual_encoder = Visual_Encoder(observation_space_shape,
                               hidden_size=hidden_size,
                               deterministic=use_deterministic_local)

            self.network = Global_Policy(obs_shape, **base_kwargs)
        else:
            raise NotImplementedError

        self.dist_discrete = Categorical(self.network.output_size, 3)

        self.critic_linear = nn.Linear(256, 1) #value
        self.model_type = model_type
        self.num_steps = 0
        self.device = device

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, inputs, rnn_hxs, rgb, masks, extras):
        if extras is None:
            visual_feature = self.visual_encoder(rgb, masks)
            return self.network(inputs, rnn_hxs, visual_feature , masks)
        else:
            visual_feature = self.visual_encoder(rgb, masks)
            return self.network(inputs, rnn_hxs, visual_feature, masks, extras)

    def act(self, inputs, num_scenes, rnn_hxs, rgb, masks, extras=None, deterministic=False):
        actor_features, rnn_hxs = self(inputs, rnn_hxs, rgb, masks, extras)
        value = self.critic_linear(actor_features)#.squeeze(-1)
        
        action_log_probs = torch.zeros(num_scenes).to(self.device)
        action = torch.zeros(num_scenes,1).to(self.device)

        for e in range(num_scenes):

            dist = self.dist_discrete(actor_features[e]) #Categorical distribution
            if deterministic: 
                action[e,0] = dist.mode()
            else: #default is deterministic=False.
                action[e,0] = dist.sample()
            action_log_probs[e] = dist.log_probs(action[e,0])

        return  action, value, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, rgb, masks, extras=None):
        actor_features, rnn_hxs = self(inputs, rnn_hxs, rgb, masks, extras)
        value = self.critic_linear(actor_features)#.squeeze(-1)

        return value

    def evaluate_actions(self, inputs, num_scenes, rnn_hxs, rgb, masks, action_discrete, action_box, extras=None):

        actor_features, rnn_hxs = self(inputs, rnn_hxs, rgb, masks, extras)
        value = self.critic_linear(actor_features)#.squeeze(-1)
        
        action_log_probs = torch.zeros(num_scenes).to(self.device)
        dist_entropy = 0
        for e in range(num_scenes):

            dist = self.dist_discrete(actor_features[e]) #Categorical distribution
            if deterministic: 
                action[e,0] = dist.mode()
            else: #default is deterministic=False.
                action[e,0] = dist.sample()
            action_log_probs[e] = dist.log_probs(action[e,0])

            dist_entropy += dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    def epsilon(self):
        eps = 0.1 + (1.0 - 0.1) * exp(-self.num_steps / 20000)
        self.num_steps += 1
        return eps
