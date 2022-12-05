import torch
import os
from torch.utils import data
from scipy.io import loadmat
from enum import Enum



class QuadTree(object):
    class NodeType(Enum):
        LEAF_MIX = 0 # mixed leaf node
        NON_LEAF = 1 # non-leaf node

    class Node(object):
        def __init__(self, fea=None, child=None, node_type=None, fea2 = None):
            self.fea = fea          # feature vector for a leaf node
            self.fea2 = fea2
            self.child = child        # child nodes
            self.node_type = node_type

        def is_leaf(self):
            return self.node_type != QuadTree.NodeType.NON_LEAF and self.fea is not None

        def is_expand(self):
            return self.node_type == QuadTree.NodeType.NON_LEAF

    def __init__(self, feas, ops):
        
        fea_list = [b for b in torch.split(feas, 1, 0)]
        #print(fea_list[0].size())
        fea_list.reverse()
        stack = []
        for id in range(ops.size()[1]):
            if ops[0, id] == QuadTree.NodeType.LEAF_MIX.value:
                stack.append(QuadTree.Node(fea=fea_list.pop(), node_type=QuadTree.NodeType.LEAF_MIX))                
            elif ops[0, id] == QuadTree.NodeType.NON_LEAF.value:
                child_node = []
                for i in range(4):
                    child_node.append(stack.pop())
                stack.append(QuadTree.Node(child=child_node, node_type=QuadTree.NodeType.NON_LEAF))

        assert len(stack) == 1
        self.root = stack[0]




####################NOT USED#####################
class QuadTree_backup(object):
    class NodeType(Enum):
        LEAF_FULL = 0  # full leaf node
        LEAF_EMPTY = 1 # empty leaf node
        LEAF_MIX = 2 # mixed leaf node
        NON_LEAF = 3 # non-leaf node

    class Node(object):
        def __init__(self, fea=None, child=None, node_type=None):
            self.fea = fea          # feature vector for a leaf node
            self.child = child        # child nodes
            self.node_type = node_type
            self.label = torch.LongTensor([self.node_type.value])

        def is_leaf(self):
            return self.node_type != QuadTree.NodeType.NON_LEAF and self.fea is not None
        
        def is_empty_leaf(self):
            return self.node_type == QuadTree.NodeType.LEAF_EMPTY

        def is_expand(self):
            return self.node_type == QuadTree.NodeType.NON_LEAF

    def __init__(self, feas, ops):
        
        #self.label = label.type(torch.LongTensor)
        fea_list = [b for b in torch.split(feas, 1, 0)]
        #print(fea_list[0].size())
        fea_list.reverse()
        stack = []
        for id in range(ops.size()[1]):
            if ops[0, id] == QuadTree.NodeType.LEAF_FULL.value:
                stack.append(QuadTree.Node(fea=fea_list.pop(), node_type=QuadTree.NodeType.LEAF_FULL))
            elif ops[0, id] == QuadTree.NodeType.LEAF_EMPTY.value:
                stack.append(QuadTree.Node(fea=fea_list.pop(), node_type=QuadTree.NodeType.LEAF_EMPTY))
            elif ops[0, id] == QuadTree.NodeType.LEAF_MIX.value:
                stack.append(QuadTree.Node(fea=fea_list.pop(), node_type=QuadTree.NodeType.LEAF_MIX))                
            elif ops[0, id] == QuadTree.NodeType.NON_LEAF.value:
                child_node = []
                for i in range(4):
                    child_node.append(stack.pop())
                stack.append(QuadTree.Node(child=child_node, node_type=QuadTree.NodeType.NON_LEAF))

        assert len(stack) == 1
        self.root = stack[0]
        
