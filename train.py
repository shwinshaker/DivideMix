from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
# from PreResNet import *
from sklearn.mixture import GaussianMixture
# import dataloader_cifar as dataloader
from dataloader import WeakSupDataloader

from transformers import BertForSequenceClassification, AdamW

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data_dir', default='./data', type=str, help='path to dataset')
parser.add_argument('--dataset', default='nyt-fine', type=str)
parser.add_argument('--num_class', default=26, type=int)
parser.add_argument('--encoding_max_length', default=64, type=int)
parser.add_argument('--model', default='bert-base-uncased', type=str)
parser.add_argument('--randremove_num', default=1, type=int)
parser.add_argument('--to_random_noise', default=True, type=bool)

parser.add_argument('--lr', '--learning_rate', default=2e-5, type=float, help='initial learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--wd', '--weight_decay', default=0, type=float, help='weight decay')
parser.add_argument('--num_epochs', default=300, type=int)
# parser.add_argument('--noise_mode',  default='sym')

parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
# parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
# parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=5, type=int)
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
# torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

gradient_clipping = True

# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()                 
        # batch_size = inputs_x.size(0)
        batch_size = labels_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        # inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        # inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()
        labels_x, w_x = labels_x.cuda(), w_x.cuda()

        # with torch.no_grad():
        # label co-guessing of unlabeled samples
        outputs_u11 = net(inputs_u)
        outputs_u12 = net(inputs_u2)
        with torch.no_grad():
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
        
        pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
        ptu = pu**(1/args.T) # temparature sharpening
        
        targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
        targets_u = targets_u.detach()       
        
        # label refinement of labeled samples
        outputs_x = net(inputs_x)
        outputs_x2 = net(inputs_x2)            
        
        px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
        px = w_x*labels_x + (1-w_x)*px # if the probability of a correct label is high, then trust the label, otherwise trust the ensemble prediction
        ptx = px**(1/args.T) # temparature sharpening 
                    
        targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
        targets_x = targets_x.detach()       
        
        # - Now: Without mixmatch, equivalent to mimizing the distance between net1 and net2, similar to fixmatch
        # # mixmatch - disabled: 
        # #TODO: how to mixup two sequence is unclear, directly mixup embeddings?
        # #                      But num tokens would be different, although there is padding but it doens't make sense
        # l = np.random.beta(args.alpha, args.alpha)        
        # l = max(l, 1-l)
                
        # all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        # all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        # idx = torch.randperm(all_inputs.size(0))

        # input_a, input_b = all_inputs, all_inputs[idx]
        # target_a, target_b = all_targets, all_targets[idx]
        
        # mixed_input = l * input_a + (1 - l) * input_b        
        # mixed_target = l * target_a + (1 - l) * target_b
                
        # logits = net(mixed_input)
        # logits_x = logits[:batch_size*2] # all_inputs has size of batchsize*4
        # logits_u = logits[batch_size*2:]        
           
        # Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)

        Lx, Lu, lamb = criterion((outputs_x + outputs_x2)*0.5, targets_x,
                                 (outputs_u11 + outputs_u12)*0.5 , targets_u,
                                 epoch + batch_idx/num_iter, warm_up)
        
        # regularization
        logits = torch.cat([(outputs_x + outputs_x2)*0.5, (outputs_u11 + outputs_u12)*0.5])
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if gradient_clipping:
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        
        sys.stdout.write('\r')
        # sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
        #         %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()

def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, index) in enumerate(dataloader):      
        # inputs, labels = inputs.cuda(), labels.cuda() 
        labels = labels.cuda() 
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        #TODO: try penalty
        # if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
        #     penalty = conf_penalty(outputs)
        #     L = loss + penalty      
        # elif args.noise_mode=='sym':   
        #     L = loss
        optimizer.zero_grad()
        loss.backward()  
        if gradient_clipping:
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test(epoch, net1, net2, test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # inputs, targets = inputs.cuda(), targets.cuda()
            targets = targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch, acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch, acc))
    test_log.flush()  

def eval_train(model, eval_loader, all_loss):    
    model.eval()
    losses = torch.zeros(8229) # train_size
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            # inputs, targets = inputs.cuda(), targets.cuda() 
            targets = targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            # for b in range(inputs.size(0)):
            for b in range(targets.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    # if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
    #     history = torch.stack(all_loss)
    #     input_loss = history[-5:].mean(0)
    #     input_loss = input_loss.reshape(-1,1)
    # else:
    #     input_loss = losses.reshape(-1,1)
    history = torch.stack(all_loss)
    input_loss = history[-5:].mean(0)
    input_loss = input_loss.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob, all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    #TODO: why rampup_length = 16?
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

class EncodingModel(nn.Module):
    def __init__(self, model):
        super(EncodingModel, self).__init__()
        self.model = model

    def forward(self, inputs):
        inputs = {key: val.cuda() for key, val in inputs.items()}
        return self.model(**inputs).logits

def create_model():
    # model = ResNet18(num_classes=args.num_class)
    # model = model.cuda()

    print('\n=====> Initializing model..')
    # from pretrained: replace the pretraining head with a randomly initialized classification head
    model = BertForSequenceClassification.from_pretrained(
        args.model,
        num_labels=args.num_class,  # The number of output labels -- 2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    print("     Total params: %.2fM" % (sum(p.numel() for p in model.parameters())/1000000.0))

    # wrap inference
    model = EncodingModel(model)

    return model

path_name = args.dataset
if args.to_random_noise:
    path_name += '_to_rand_noise'
stats_log = open('./checkpoints/%s' % path_name + '_stats.txt','w+') 
test_log = open('./checkpoints/%s' % path_name + '_acc.txt','w+')     

warm_up = 5
# if args.dataset=='cifar10':
    # warm_up = 10
# elif args.dataset=='cifar100':
    # warm_up = 30

# loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    # root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))
loader = WeakSupDataloader(args.dataset,
                           data_dir=args.data_dir,
                           batch_size=args.batch_size,
                           num_workers=16,
                           log=stats_log,
                           config=args)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
# optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer1 = AdamW(net1.parameters(), lr=args.lr, weight_decay=args.wd)
optimizer2 = AdamW(net2.parameters(), lr=args.lr, weight_decay=args.wd)

# original lr scheduler: batch wise
# scheduler = get_linear_schedule_with_warmup(optimizer,
                                            # num_warmup_steps=0,  # Default value in run_glue.py
                                            # num_training_steps=len(loaders.trainloader) * config.epochs)
# from torch.optim.lr_scheduler import LinearLR
# scheduler1 = LinearLR(optimizer1, start_factor=0.5, total_iters=args.num_epochs)
# scheduler2 = LinearLR(optimizer2, start_factor=0.5, total_iters=args.num_epochs)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
# if args.noise_mode=='asym':
    # conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks

for epoch in range(args.num_epochs + 1):   
    # - linear schedule
    lr = args.lr * (1 - epoch / args.num_epochs)
    # lr=args.lr
    # if epoch >= 150:
        # lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch < warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch, net2, optimizer2, warmup_trainloader) 
    else:         
        print('\nEval Net1')
        prob1, all_loss[0] = eval_train(net1, eval_loader, all_loss[0])   
        print('\nEval Net2')
        prob2, all_loss[1] = eval_train(net2, eval_loader, all_loss[1])          
               
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)      
        
        # use net2 to divide the data (labeled&unlabeled), use net1 to perform semi-supervised learning
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2) # co-divide
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader) # train net1  
        
        # use net1 to divide the data (labeled&unlabeled), use net2 to perform semi-supervised learning
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1) # co-divide
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader) # train net2         

    test(epoch, net1, net2, test_loader)  

