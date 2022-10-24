from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter

from .tools import EncodingDataset
            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class EncodingArr:
    @classmethod
    def cat(cls, arr1, arr2):
        return #TODO

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, index):
        return {key: val[index] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings[list(self.encodings.keys())[0]])

class WeakSupDataset(Dataset): 
    # def __init__(self, dataset, root_dir, mode
                #  pred=[], probability=[], log=''): 

    def set_transform(self, perturbator):
        def _transform(inputs):
            return perturbator.remove_random_words_batch(inputs, self.config.randremove_num)
        self.transform = _transform

    def __init__(self, mode, dataset, data_dir, pred=[], probability=[], config=None, log=None)
        
        # self.transform = transform
        self.mode = mode  
        # self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        self.config = config
     
        data_dict = get_dataset(dataset, data_dir=data_dir, config=config)
        self.decode = lambda inputs: data_dict['tokenizer'].decode(inputs['input_ids'])
        self.label_decode = lambda label: data_dict['label_names'][label]
        self.classes = data_dict['labels'].unique().tolist()
        seed_word_perturbator = SeedWordPerturbator(data_dict['tokenizer'],
                                                    list(data_dict['label_names'].keys()),
                                                    data_dict['label_names'],
                                                    dataset=dataset,
                                                    data_dir=data_dir)
        self.set_transform(seed_word_perturbator)

        if self.mode=='test':
            inputs = EncodingArr(data_dict['encodings'])
            labels = data_dict['labels']
            # if dataset=='cifar10':                
            #     test_dic = unpickle('%s/test_batch'%root_dir)
            #     self.test_data = test_dic['data']
            #     self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            #     self.test_data = self.test_data.transpose((0, 2, 3, 1))  
            #     self.test_label = test_dic['labels']
            # elif dataset=='cifar100':
            #     test_dic = unpickle('%s/test'%root_dir)
            #     self.test_data = test_dic['data']
            #     self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            #     self.test_data = self.test_data.transpose((0, 2, 3, 1))  
            #     self.test_label = test_dic['fine_labels']                            
        else:    
            inputs = EncodingArr(data_dict['encodings'])
            labels = torch.empty(len(self.inputs), dtype=torch.int64)
            # train_data=[]
            # train_label=[]
            # if dataset=='cifar10': 
            #     for n in range(1,6):
            #         dpath = '%s/data_batch_%d'%(root_dir,n)
            #         data_dic = unpickle(dpath)
            #         train_data.append(data_dic['data'])
            #         train_label = train_label+data_dic['labels']
            #     train_data = np.concatenate(train_data)
            # elif dataset=='cifar100':    
            #     train_dic = unpickle('%s/train'%root_dir)
            #     train_data = train_dic['data']
            #     train_label = train_dic['fine_labels']
            # train_data = train_data.reshape((50000, 3, 32, 32))
            # train_data = train_data.transpose((0, 2, 3, 1))

            # load weak labels
            # trainsubids = np.array([], dtype=int)
            tar = get_weak_supervision(np.arange(len(self.inputs)), data_dir, dataset)
            # save_array('id_train_weak_sup.npy',  tar['index'], config=config) # save all ids having weak labels for later selection of weak labels

            # if hasattr(config, 'pseudo_weak_sup_select') and config.pseudo_weak_sup_select:
            #     print('\n==> Select weak pseudo-labels based on confidence..')
            #     tar_path = '%s/pseudo_unlabeled=%s.pt' % (config.weak_model_path, 'id_train_weak_sup')
            #     tar = pseudo_label_selection(torch.load(tar_path), classes=trainset.classes,
            #                                 threshold=config.pseudo_weak_sup_threshold,
            #                                 threshold_type=config.pseudo_weak_sup_threshold_type,
            #                                 class_balance=config.pseudo_weak_sup_class_balance,
            #                                 save_dir=config.save_dir)

            # save_array('id_unlabeled.npy', np.setdiff1d(np.arange(len(trainset)), tar['index']), config=config)

            # add_label(trainset, ids=tar['index'], labels=tar['pseudo_label'])
            # trainsubids = add_trainsubids(trainsubids, tar['index'])

            # if os.path.exists(noise_file):
            #     noise_label = json.load(open(noise_file,"r"))
            # else:    #inject noise   
            #     noise_label = []
            #     idx = list(range(50000))
            #     random.shuffle(idx)
            #     num_noise = int(self.r*50000)            
            #     noise_idx = idx[:num_noise]
            #     for i in range(50000):
            #         if i in noise_idx:
            #             if noise_mode=='sym':
            #                 if dataset=='cifar10': 
            #                     noiselabel = random.randint(0,9)
            #                 elif dataset=='cifar100':    
            #                     noiselabel = random.randint(0,99)
            #                 noise_label.append(noiselabel)
            #             elif noise_mode=='asym':   
            #                 noiselabel = self.transition[train_label[i]]
            #                 noise_label.append(noiselabel)                    
            #         else:    
            #             noise_label.append(train_label[i])   
            #     print("save noisy labels to %s ..."%noise_file)        
            #     json.dump(noise_label,open(noise_file,"w"))       
            
            weak_ids = tar['index']
            weak_labels = tar['pseudo_label']
            if self.mode == 'weak':
                self.inputs = inputs[weak_ids]
                self.labels = weak_labels
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    # clean = (np.array(noise_label)==np.array(train_label))                                                       
                    # auc_meter = AUCMeter()
                    # auc_meter.reset()
                    # auc_meter.add(probability,clean)        
                    # auc,_,_ = auc_meter.value()               
                    # log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    # log.flush()      

                    # - pred_idx must be indexing weak data (self.mode=weak), thus indexing weak_ids
                    # TODO: Assert something here to ensure this
                    self.inputs = inputs[weak_ids[pred_idx]]
                    self.labels = weak_labels[pred_idx]
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                    _inputs = []
                    _inputs.append(inputs[weak_ids[pred_idx]])
                    # self.labels = [weak_labels[pred_idx]]
                    # - all other unlabeled data
                    _inputs.append(inputs[np.setdiff1d(np.arange(len(inputs)), weak_ids)])
                    self.inputs = {k: torch.cat([e.encodings[k] for e in _inputs]) for k in inputs.encodings} 
                    # TODO: not sure if correct
                    # TODO: make it a class method
                
                # self.train_data = train_data[pred_idx]
                # self.noise_label = [noise_label[i] for i in pred_idx]                          
                # print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            input_, label, prob = self.inputs[index], self.labels[index], self.probability[index]
            input1 = self.transform(input_) 
            input2 = self.transform(input_) 
            return input1, input2, label, prob            
        elif self.mode=='unlabeled':
            input_ = self.inputs[index]
            input1 = self.transform(input_) 
            input2 = self.transform(input_) 
            return input1, input2
        elif self.mode=='weak':
            input_, label = self.inputs[index], self.labels[index]
            return input_, label, index        
        elif self.mode=='test':
            input_, label = self.inputs[index], self.labels[index]
            return input_, label
           
    def __len__(self):
        return len(self.inputs)
        

from .perturbation import SeedWordPerturbator
class weak_sup_dataloader():  
    def __init__(self, dataset, batch_size, num_workers, data_dir, log=None, config=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.log = log
        self.config = config

    def run(self, mode, pred=[], prob=[]):
        if mode=='warmup':
            # train on weak labels to get preliminary classifiers
            all_dataset = WeakSupDataset(mode="weak", 
                                         dataset=self.dataset,
                                         data_dir=self.data_dir,
                                         config=self.config)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader

        elif mode=='eval_train':
            # only evaluate on weak labels to see which are noisy, which are clean
            eval_dataset = WeakSupDataset(mode='weak',
                                          dataset=self.dataset,
                                          data_dir=self.data_dir,
                                          config=self.config)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader    

        elif mode=='train':
            labeled_dataset = WeakSupDataset(mode="labeled", 
                                             dataset=self.dataset,
                                             data_dir=self.data_dir,
                                             pred=pred, probability=prob, 
                                             log=self.log,
                                             config=self.config)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = WeakSupDataset(mode="unlabeled", 
                                               dataset=self.dataset,
                                               data_dir=self.data_dir,
                                               pred=pred,
                                               log=self.log,
                                               config=self.config)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = WeakSupDataset(mode="test", 
                                          dataset=self.dataset,
                                          data_dir=self.data_dir,
                                          config=self.config)
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
    