import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time


#########################################################################################
## Encoder
#########################################################################################

class Sampler(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(Sampler, self).__init__()
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

        #output = self.elu(self.conv1(input_feature))
        #output = self.elu(self.conv2(input_feature))
        output = self.elu(self.conv3(input_feature))
        
        output = self.mlp1(output.view(-1, output.size()[1]))
        output = self.dropout(self.elu(output))
        output = self.mlp2(output)
        #output = self.softmax(output)
        return output

class BoxEncoder(nn.Module):

    def __init__(self, input_size, feature_size):
        super(BoxEncoder, self).__init__()
        self.encoder = nn.Linear(feature_size, feature_size)
        
        
        #2d conv layer1
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=2, bias=True, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        
        #2d conv layer2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=True, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        #2d conv layer3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, bias=True, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
       
        
        
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
        
        return box_vector



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


#########################################################################################
## Decoder (segmenter)
#########################################################################################

class SampleDecoder(nn.Module):
    """ Decode a randomly sampled noise into a feature vector """
    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(feature_size, 64, kernel_size=4, stride=1, bias=True)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, bias=True, padding=1)
        
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        #self.bn1 = nn.BatchNorm1d(hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)
#         self.bn2 = nn.BatchNorm1d(feature_size)
#         self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu')) 
        
    def forward(self, input_feature):
#         output = self.mlp1(input_feature)
#         output = self.mlp2(output)
        output = input_feature.view(-1, input_feature.size()[1] , 1, 1, 1)#
        
        output = self.elu(self.deconv1(output))
#         #output = self.tanh(self.bn4(self.deconv2(output)))
        
        return output


class AdjDecoder(nn.Module):
    """ Decode an input (parent) feature into a left-child and a right-child feature """
    def __init__(self, feature_size, hidden_size):
        super(AdjDecoder, self).__init__()
        self.mlp = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, bias=True, padding=1)
        self.mlp_child1 = nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=True)
        self.mlp_child2 = nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=True)
        self.mlp_child3 = nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=True)
        self.mlp_child4 = nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=True)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()
        self.bn = nn.BatchNorm2d(128)
        

    def forward(self, parent_feature, skip_feature):
        
        #print(parent_feature.size())
        #print(skip_feature.size())
        vector = self.mlp(torch.cat((parent_feature,skip_feature),1))
        
        vector = self.bn(vector)
        vector = self.elu(vector)
        
        child_feature1 = self.bn1(self.mlp_child1(vector))
        child_feature1 = self.elu(child_feature1)
        child_feature2 = self.bn2(self.mlp_child2(vector))
        child_feature2 = self.elu(child_feature2)
        child_feature3 = self.bn3(self.mlp_child3(vector))
        child_feature3 = self.elu(child_feature3)
        child_feature4 = self.bn4(self.mlp_child4(vector))
        child_feature4 = self.elu(child_feature4)
        
        return  child_feature1,child_feature2,child_feature3,child_feature4

class BoxDecoder(nn.Module):
    
    def __init__(self, feature_size, box_size, n_class):
        super(BoxDecoder, self).__init__()
        self.encoder = nn.Linear(feature_size, feature_size)
        
        
#         #2d deconv layer1
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, bias=True, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        
        #2d deconv layer2
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, bias=True, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        #2d deconv layer3
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, bias=True, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        
        #2d deconv layer4
        self.deconv4 = nn.ConvTranspose2d(16, n_class, kernel_size=3, stride=2, bias=True, padding=1)
        self.bn4 = nn.BatchNorm2d(1)
        
        
        #self.maxpool = nn.MaxPool2d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()

    def forward(self, box_input, skip_feature):
        
        box_vector = self.deconv1(torch.cat((box_input,skip_feature),1))
        box_vector = self.bn1(box_vector)
        #box_vector = self.elu(box_vector)
        box_vector = self.elu(box_vector)
        
        #print('input2 box')
        #print(box_input.size())
        
        box_vector = self.deconv2(box_vector)
        box_vector = self.bn2(box_vector)
        box_vector = self.elu(box_vector)
        #box_vector = self.elu(box_vector)
        
        #print('deconv1 box')
        #print(box_vector.size())
        
        box_vector = self.deconv3(box_vector)
        #box_vector = self.elu(box_vector)
        box_vector = self.bn3(box_vector)
        #box_vector = self.elu(box_vector)
        #box_vector = self.elu(box_vector)
        box_vector = self.elu(box_vector)
        #box_vector = torch.clamp(box_vector, min=1e-7, max=1-1e-7)
        
        #print('deconv2 box')
        #print(box_vector.size())
        
        box_vector = self.deconv4(box_vector)
        box_vector = self.sigmoid(box_vector)
        #box_vector = self.bn4(box_vector)
#         box_vector = self.sigmoid(box_vector)
#         box_vector = torch.clamp(box_vector, min=1e-7, max=1-1e-7)
        
        return box_vector

class ROctSegmenter(nn.Module):
    def __init__(self, config):
        super(ROctSegmenter, self).__init__()
        
        ###########Decoder###########
        self.box_decoder = BoxDecoder(feature_size = config.feature_size, box_size = config.box_code_size, n_class = 1)
        
        self.adj_decoder1 = AdjDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.adj_decoder2 = AdjDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.adj_decoder3 = AdjDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.adj_decoder4 = AdjDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.adj_decoder5 = AdjDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)   
        
        self.sample_decoder = SampleDecoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        
        
        ###########Encoder###########
        self.box_encoder = BoxEncoder(input_size = config.box_code_size, feature_size = config.feature_size)
        
        self.adj_encoder1 = AdjEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.adj_encoder2 = AdjEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.adj_encoder3 = AdjEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.adj_encoder4 = AdjEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        self.adj_encoder5 = AdjEncoder(feature_size = config.feature_size, hidden_size = config.hidden_size)
        
        self.sample_encoder = Sampler(feature_size = config.feature_size, hidden_size = config.hidden_size)

    def boxEncoder(self, fea):
        return self.box_encoder(fea)
    
    def adjEncoder1(self, c1,c2,c3,c4,c5,c6,c7,c8):
        return self.adj_encoder1(c1,c2,c3,c4,c5,c6,c7,c8)
    
    def adjEncoder2(self, c1,c2,c3,c4,c5,c6,c7,c8):
        return self.adj_encoder2(c1,c2,c3,c4,c5,c6,c7,c8)
    
    def adjEncoder3(self, c1,c2,c3,c4,c5,c6,c7,c8):
        return self.adj_encoder3(c1,c2,c3,c4,c5,c6,c7,c8)
    
    def adjEncoder4(self, c1,c2,c3,c4,c5,c6,c7,c8):
        return self.adj_encoder4(c1,c2,c3,c4,c5,c6,c7,c8)    
    
    def adjEncoder5(self, c1,c2,c3,c4,c5,c6,c7,c8):
        return self.adj_encoder5(c1,c2,c3,c4,c5,c6,c7,c8)   

    def sampleEncoder(self, feature):
        return self.sample_encoder(feature)

    def boxDecoder(self, feature, skip_feature):
        return self.box_decoder(feature, skip_feature)

    def adjDecoder1(self, feature, skip_feature):
        return self.adj_decoder1(feature, skip_feature)

    def adjDecoder2(self, feature, skip_feature):
        return self.adj_decoder2(feature, skip_feature)
    
    def adjDecoder3(self, feature, skip_feature):
        return self.adj_decoder3(feature, skip_feature)

    def adjDecoder4(self, feature, skip_feature):
        return self.adj_decoder4(feature, skip_feature)
    
    def adjDecoder5(self, feature, skip_feature):
        return self.adj_decoder5(feature, skip_feature)

    def sampleDecoder(self, feature):
        return self.sample_decoder(feature)


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




