import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time

#########################################################################################
## Encoder
#########################################################################################

        
class TreeClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(TreeClassifier, self).__init__()
        #self.conv1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, bias=True, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=True, padding=1)
        self.conv3 = nn.Conv2d(64, feature_size, kernel_size=4, stride=1, bias=True)
        
        
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.tanh = nn.Tanh()
        self.elu = nn.ReLU()
        self.mlp2 = nn.Linear(hidden_size, feature_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.dropout = nn.Dropout(0.5) #apply dropout in a neural network
        #self.softmax = nn.Softmax()

    def forward(self, input_feature):


        output = self.elu(self.conv3(input_feature))
        output = self.mlp1(output.view(-1, output.size()[1]))
        #output = self.dropout(self.elu(output))
        output = self.elu(self.mlp2(output))


        return output

class BoxEncoder(nn.Module):

    def __init__(self, num_maps, input_size, feature_size):
        super(BoxEncoder, self).__init__()
        self.encoder = nn.Linear(feature_size, feature_size)
        self.num_maps = num_maps
        
        
        #2d conv layer1
        self.conv1 = nn.Conv2d(self.num_maps, 16, kernel_size=3, stride=2, bias=True, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        
        #2d conv layer2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=True, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        #2d conv layer3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, bias=True, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
       
        # if non-resursive
        #2d conv layer4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, bias=True, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        
        #2d conv layer5
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, bias=True, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        #2d conv layer6
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, bias=True, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        #2d conv layer7
        self.conv7 = nn.Conv2d(256, 64, kernel_size=3, stride=2, bias=True, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        # endif non-resursive
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu')) 
        
 
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        self.elu = nn.ReLU()

    def forward(self, box_input):

        box_vector = self.conv1(box_input.add(-0.5).mul(2))
        box_vector = self.bn1(box_vector)
        box_vector = self.elu(box_vector)
        #box_vector = self.maxpool(box_vector)

        #print('input box')
        #print(box_input.size())
        
        box_vector = self.conv2(box_vector)
        box_vector = self.bn2(box_vector)
        box_vector = self.elu(box_vector)
        #box_vector = self.maxpool(box_vector)
        
        #print('conv1 box')
        #print(box_vector.size())
        
        box_vector = self.conv3(box_vector)
        box_vector = self.bn3(box_vector)
        box_vector = self.elu(box_vector)
        #box_vector = self.maxpool(box_vector)

        box_vector = self.conv4(box_vector)
        box_vector = self.bn4(box_vector)
        box_vector = self.elu(box_vector)

        box_vector = self.conv5(box_vector)
        box_vector = self.bn5(box_vector)
        box_vector = self.elu(box_vector)

        box_vector = self.conv6(box_vector)
        box_vector = self.bn6(box_vector)
        box_vector = self.elu(box_vector)

        box_vector = self.conv7(box_vector)
        box_vector = self.bn7(box_vector)
        box_vector = self.elu(box_vector)
        
        return box_vector
    
class BoxEncoder2(nn.Module):

    def __init__(self, input_size, feature_size):
        super(BoxEncoder2, self).__init__()
        
        self.feature_size = feature_size


    def forward(self, box_input):

#         ##print(box_input)
#         ##print(torch.zeros(box_input.size()[0],self.feature_size))
        return Variable(torch.zeros(box_input.size()[0],64,4,4).cuda())

class AdjEncoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(AdjEncoder, self).__init__()
        self.child1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=True)
        self.child2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False)
        self.child3 = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False)
        self.child4 = nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False)

        self.second = nn.Conv2d(128, 64, kernel_size=3, stride=1, bias=True, padding=1)

        self.tanh = nn.Tanh()
        self.elu = nn.ReLU()
        
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.bn11 = nn.BatchNorm2d(128)
        self.bn12 = nn.BatchNorm2d(64)
        
        #self.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

    def forward(self, c1,c2,c3,c4):
        output = self.bn1(self.child1(c1))
        output += self.bn2(self.child2(c2))
        output += self.bn3(self.child3(c3))
        output += self.bn4(self.child4(c4))
        
        
        output = self.bn11(output)
        
        output = self.elu(output)
        output = self.second(output)
        #print(output.size())
        if len(output.size())==1:
            output = output.unsqueeze(0)
        output = self.bn12(output)
        output = self.elu(output)
#         output = self.third(output)
#         output = self.tanh(output)
        
        return output


class ROctEncoder(nn.Module):

    def __init__(self, config):
        super(ROctEncoder, self).__init__()
        self.box_encoder = BoxEncoder(num_maps = config.num_maps, input_size = config.box_code_size, feature_size = config.feature_size)
        self.box_encoder2 = BoxEncoder2(input_size = config.box_code_size, feature_size = config.feature_size)
        
        self.adj_encoder1 = AdjEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.adj_encoder2 = AdjEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.adj_encoder3 = AdjEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.adj_encoder4 = AdjEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.adj_encoder5 = AdjEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        
        self.tree_classifier = TreeClassifier(feature_size = config.feature_size, hidden_size = config.hidden_size)

    def boxEncoder(self, fea):
        return self.box_encoder(fea)

    def boxEncoder2(self, fea):
        return self.box_encoder2(fea)
    
    def adjEncoder1(self, c1,c2,c3,c4):
        return self.adj_encoder1(c1,c2,c3,c4)
    
    def adjEncoder2(self, c1,c2,c3,c4):
        return self.adj_encoder2(c1,c2,c3,c4)
    
    def adjEncoder3(self, c1,c2,c3,c4):
        return self.adj_encoder3(c1,c2,c3,c4)
    
    def adjEncoder4(self, c1,c2,c3,c4):
        return self.adj_encoder4(c1,c2,c3,c4)    
    
    def adjEncoder5(self, c1,c2,c3,c4):
        return self.adj_encoder5(c1,c2,c3,c4)   

    def treeClassifier(self, feature):
        return self.tree_classifier(feature)


def encode_structure_fold(fold, tree):

    def encode_node(node,l):
        if node.is_leaf():
            return fold.add('boxEncoder', node.fea)
        elif node.is_expand():
            child = []
            for i in range(4):
                child.append(encode_node(node.child[i],l+1))
            return fold.add('adjEncoder'+str(l), child[0], child[1],child[2],child[3])

    encoding = encode_node(tree.root,1)
    return fold.add('treeClassifier', encoding)


