import time
import os
import math
from time import gmtime, strftime
from datetime import datetime
import torch
import util
from torch import nn
from torch.autograd import Variable
import torch.utils.data
from torchfoldext import FoldExt
from dynamicplot import DynamicPlot
from IPython.core.debugger import set_trace

from data import ROctDataset
from ROctNetmodel_32 import ROctEncoder
import ROctNetmodel_32


config = util.get_args()

config.box_code_size = 32
config.feature_size = 1024
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


# In[4]:


config.cuda = not config.no_cuda
if config.gpu<0 and config.cuda:
    config.gpu = 0
torch.cuda.set_device(config.gpu)

if config.cuda and torch.cuda.is_available():
    print("Using CUDA on GPU ", config.gpu)
else:
    print("Not using CUDA.")

encoder = ROctEncoder(config)


# In[5]:


if config.cuda:
    encoder.cuda(config.gpu)


# In[6]:


def my_collate(batch):
    return batch


# In[7]:


valid_data = ROctDataset('/data/juncheng/ModelNet10/64_32_vox/test_1',1,10896)#29616
valid_data.trees = valid_data.trees[0:-1:1200]
valid_iter = torch.utils.data.DataLoader(valid_data, batch_size=25, shuffle=True, collate_fn=my_collate)


# In[9]:


encoder_opt = torch.optim.Adam(encoder.parameters(), lr=1e-4)
encoder = encoder.train()

print("Start training ...... ")

if config.save_log:
    fd_log = open('training_log.log', mode='a')
    fd_log.write('\n\nTraining log at '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    fd_log.write('\n#epoch: {}'.format(config.epochs))
    fd_log.write('\nbatch_size: {}'.format(config.batch_size))
    fd_log.write('\ncuda: {}'.format(config.cuda))
    fd_log.flush()

header = '     Time    Epoch   Chunk   Iteration    Progress(%) Loss Acc'
log_template = ' '.join('{:>9s},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>5.0f}/{:<5.0f},{:>9.1f}%,{:>10.2f},{:>10.2f}'.split(','))


total_iter = config.epochs * math.ceil(47892/config.batch_size) 

if not config.no_plot:
    plot_x = [x for x in range(total_iter)]
    plot_total_loss = [None for x in range(total_iter)]
    plot_valid_loss = [None for x in range(total_iter)]
    dyn_plot = DynamicPlot(title='Training loss over epochs (ROctNet)', xdata=plot_x, ydata={'Total_loss':plot_total_loss, 'Accuracy': plot_valid_loss})#
    iter_id = 0
    max_loss = 0

start = time.time()


for epoch in range(config.epochs):
    
    print(header)
    
    for base in range(1,47892,incre):
    

        #print("Loading data ...... "+str(base), end='', flush=True)
        roct_data = ROctDataset('/data/juncheng/ModelNet10/64_32_vox/shuffled_train_1', base, incre)
        train_iter = torch.utils.data.DataLoader(roct_data, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate)
        #valid_iter = torch.utils.data.DataLoader(valid_data, batch_size=200, shuffle=True, collate_fn=my_collate)
        #print("DONE")
        
        for batch_idx, batch in enumerate(train_iter):
            encoder.train()
            # Initialize torchfold for *encoding*
            enc_fold = FoldExt(cuda=config.cuda)
            enc_fold_nodes = []     # list of fold nodes for encoding
            # Collect computation nodes recursively from encoding process
            for example in batch:
                enc_fold_nodes.append(ROctNetmodel_32.encode_structure_fold(enc_fold, example))

            # Apply the computations on the encoder model
            #set_trace()
            total_loss = enc_fold.apply(encoder, [enc_fold_nodes])
            #set_trace()
            total_loss = total_loss[0].sum() / len(batch)  

            # Do parameter optimization
            encoder_opt.zero_grad()
            total_loss.backward()
            encoder_opt.step()
            
            
            encoder.eval()
            #validation loss
            enc_fold = FoldExt(cuda=config.cuda)
            enc_fold_nodes = []     # list of fold nodes for encoding
            # Collect computation nodes recursively from encoding process
            for valid_idx, valid_batch in enumerate(valid_iter):
                for example in valid_batch:
                    enc_fold_nodes.append(ROctNetmodel_32.encode_structure_fold2(enc_fold, example))

            # Apply the computations on the encoder model
            #set_trace()
            code = enc_fold.apply(encoder, [enc_fold_nodes])
            code = code[0]
            #set_trace()
            correct = 0
            for i in range(len(valid_batch)):
                #set_trace()
                correct += torch.max(code[i,:],0)[1].data.cpu() == valid_batch[i].label 
            valid_loss = 100*correct.sum()/len(valid_batch)
            
            
            # Report statistics
            if batch_idx % config.show_log_every == 0:
                print(log_template.format(strftime("%H:%M:%S",time.gmtime(time.time()-start)),
                    epoch, config.epochs, math.ceil(base/incre) , math.ceil(47892/incre) , 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx+len(train_iter)*math.floor(base/incre)+len(train_iter)*epoch*math.ceil(47892/incre)) / total_iter,
                    total_loss.data[0], valid_loss))#
            # Plot losses
            if not config.no_plot:
                plot_total_loss[iter_id] = total_loss.data[0]
                plot_valid_loss[iter_id] = valid_loss
                max_loss = max(max_loss, total_loss.data[0], valid_loss)#
                dyn_plot.setxlim(0., (iter_id+1)*1.05)
                dyn_plot.setylim(0., max_loss*1.05)
                dyn_plot.update_plots(ydata={'Total_loss':plot_total_loss,'Accuracy':plot_valid_loss})#
                iter_id += 1
                


    # Save snapshots of the models being trained
    if config.save_snapshot and (epoch+1) % config.save_snapshot_every == 0 :
        print("Saving snapshots of the models ...... ", end='', flush=True)
        torch.save(encoder, 'snapshot/model_epoch_{}_loss_{:.2f}.pkl'.format(epoch+1, total_loss.data[0]))
        print("DONE")
    # Save training log
    if config.save_log and (epoch+1) % config.save_log_every == 0 :
        fd_log = open('training_log.log', mode='a')
        fd_log.write('\nepoch:{}total_loss:{:.2f}'.format(epoch+1, total_loss.data[0]))
        fd_log.close()


# In[ ]:


# Save the final models
print("Saving final models ...... ", end='', flush=True)
torch.save(encoder, 'models/modelnet10_32_16.pkl')
print("DONE")


# In[ ]:


encoder = torch.load('models/modelnet10_16.pkl')
#encoder = torch.load('snapshot/model_epoch_11_loss_0.96.pkl')


# In[10]:


from torch.autograd import Variable
def encode_structure(model, tree):
    """
    Encode a tree into a code
    """
    def encode_node(node,l):
        if node.is_leaf():
            if not node.is_empty_leaf():
                return model.boxEncoder(Variable(node.fea.cuda()))
            else:
                return model.boxEncoder2(Variable(node.fea.cuda()))
        elif node.is_expand():
            child = []
            for i in range(8):
                child.append(encode_node(node.child[i],l+1))
            mycode = 'model.adjEncoder'+str(l)+'(child[0], child[1],child[2],child[3],child[4],child[5],child[6],child[7])'
            return eval(mycode)

    encoding = encode_node(tree.root,1)
    label = model.treeClassifier(encoding)
    return label


# In[11]:


#acc on train set
import numpy as np
correct = np.zeros(40)
total = np.zeros(40)
pred = []

encoder = encoder.eval()

#import scipy.io as sio
for base in range(1,47892,120):#29616
    #test_data = ROctDataset('/data/juncheng/modelnet40/256_16/test_1',base,200)#29616
    test_data = ROctDataset('/data/juncheng/ModelNet10/64_32_vox/shuffled_train_1',base,1)#29616
    for i in range(0,len(test_data.trees),1):
        code = encode_structure(encoder, test_data.trees[i])
        _, predicted = torch.max(code, 1)
        total[test_data.trees[i].label.numpy()[0][0]] += 1
        correct[test_data.trees[i].label.numpy()[0][0]] += (predicted.data.cpu().numpy()[0] == test_data.trees[i].label.numpy()[0][0])
        pred.append(predicted.data.cpu().numpy()[0])
        print(str(base+i))
        #print(str(test_data.trees[i].label.numpy()[0][0])+'-'+str(predicted.data.cpu().numpy()[0]))
    
    
print(sum(correct) / sum(total))


# In[ ]:


print(str(sum(correct))+'/'+str(sum(total)))


# In[ ]:


import scipy.io as sio
acc = np.zeros(40)
for i in range(40):
    acc[i] = correct[i] / total[i]
    print(i+1)
    print('Accuracy of the network on the train voxs: %d %%'  % (
        100 * correct[i] / total[i]))


sio.savemat('data/test/train_acc.mat', {'train_acc':acc})


# In[12]:


#acc on test set without voting
import numpy as np
correct = np.zeros(40)
total = np.zeros(40)
pred = []

encoder = encoder.eval()

#import scipy.io as sio
for base in range(1,10896,12):#29616
    #test_data = ROctDataset('/data/juncheng/modelnet40/256_16/test_1',base,200)#29616
    test_data = ROctDataset('/data/juncheng/ModelNet10/64_32_vox/test_1',base,1)#29616
    for i in range(0,len(test_data.trees),1):
        code = encode_structure(encoder, test_data.trees[i])
        _, predicted = torch.max(code, 1)
        total[test_data.trees[i].label.numpy()[0][0]] += 1
        correct[test_data.trees[i].label.numpy()[0][0]] += (predicted.data.cpu().numpy()[0] == test_data.trees[i].label.numpy()[0][0])
        pred.append(predicted.data.cpu().numpy()[0])
        print(str(base+i))
        #print(str(test_data.trees[i].label.numpy()[0][0])+'-'+str(predicted.data.cpu().numpy()[0]))
    

print(sum(correct) / sum(total))


# In[ ]:


print(sum(correct))
print(sum(total))


# In[ ]:


#acc on test set voting
from torch.autograd import Variable
correct = np.zeros(40)
total = np.zeros(40)
pred = []

encoder = encoder.eval()

#import scipy.io as sio
for base in range(1,10896,12):
    test_data = ROctDataset('/data/juncheng/ModelNet10/64_32_vox/test_1',base,12)#29616
    #test_data.trees = test_data.trees[0:-1:12]
    for i in range(0,len(test_data.trees),12):
        total[test_data.trees[i].label.numpy()[0][0]] += 1
        predicted = Variable(torch.zeros(12))
        for j in range(12):
            code = encode_structure(encoder, test_data.trees[i+j])
            _, t = torch.max(code, 1)
            predicted[j] = t
            #print(predicted[j])
        predicted,_ = torch.mode(predicted)
        correct[test_data.trees[i].label.numpy()[0][0]] += (predicted.data.cpu().numpy()[0] == test_data.trees[i].label.numpy()[0][0])
        pred.append(predicted.data.cpu().numpy()[0])
        print(str(base+i))

print(100 * sum(correct) / sum(total))


# In[ ]:


import numpy
numpy.mean(correct/total)*100


# In[ ]:


#acc on test set classwise
import scipy.io as sio
acc = np.zeros(40)
for i in range(40):
    acc[i] = correct[i] / total[i]
    print(i+1)
    print('Accuracy of the network on the test voxs: %d %%'  % (
        100 * correct[i] / total[i]))

#print(math.sum(correct) / math.sum(total))
sio.savemat('data/test/test_acc.mat', {'test_acc':acc})


# In[ ]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(str(count_parameters(encoder)/1024/1024)+'M')


# In[ ]:





