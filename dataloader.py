from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# import random
import numpy as np
# from PIL import Image
import os
import torch
# import json
from torchnet.meter import AUCMeter

from utils.dataset import get_dataset
from utils.weak_supervision import get_weak_supervision
from utils.tools import EncodingDataset
from utils.perturbation import SeedWordPerturbator
            

# class EncodingInput:
#     def __init__(self, dic):
#         self.dic = dic
#         self.key0 = list(dic.keys())[0]

#     def cuda(self):
#         return {key: val.cuda() for key, val in self.dic.items()}

#     def size(self):
#         return self.dic[self.key0].size()

#     def __getitem__(self, index):
#         return EncodingInput({key: val[index] for key, val in self.encodings.items()})

#     def __len__(self):
#         return self.dic[self.key0].size(0)


# class EncodingArr:
class DictOfTensor:
    @classmethod
    def cat(cls, arr1, arr2):
        return #TODO

    def __init__(self, dic):
        self.dic = dic
        self.key0 = list(dic.keys())[0]

    def __getitem__(self, index):
        return DictOfTensor({key: val[index] for key, val in self.dic.items()})

    def __len__(self):
        return len(self.dic[self.key0])

    def cuda(self):
        return DictOfTensor({key: val.cuda() for key, val in self.dic.items()})

    def size(self):
        return self.dic[self.key0].size()


class WeakSupDataset(Dataset): 
    def __init__(self, mode, dataset, data_dir, pred=[], probability=[], config=None, log=None):

        self.count = 0   
        self.mode = mode  
        self.config = config
     
        data_dict = get_dataset(dataset, data_dir=data_dir, config=config)
        self.decode = lambda inputs: data_dict['tokenizer'].decode(inputs['input_ids'])
        self.label_decode = lambda label: data_dict['label_names'][label]
        self.classes = data_dict['labels'].unique().tolist()
        print(f'Num of classes: {len(self.classes)}')
        seed_word_perturbator = SeedWordPerturbator(data_dict['tokenizer'],
                                                    list(data_dict['label_names'].keys()),
                                                    data_dict['label_names'],
                                                    dataset=dataset,
                                                    data_dir=data_dir)
        self.set_transform(seed_word_perturbator, to_random_noise=config.to_random_noise)

        if self.mode=='test':
            self.inputs = DictOfTensor(data_dict['encodings'])
            self.labels = data_dict['labels']
        else:    
            inputs = DictOfTensor(data_dict['encodings'])
            labels = torch.empty(len(inputs), dtype=torch.int64)

            # load weak labels
            # trainsubids = np.array([], dtype=int)
            tar = get_weak_supervision(np.arange(len(inputs)), data_dir, dataset)
            # save_array('id_train_weak_sup.npy',  tar['index'], config=config) # save all ids having weak labels for later selection of weak labels
            # save_array('id_unlabeled.npy', np.setdiff1d(np.arange(len(trainset)), tar['index']), config=config)

            if self.mode == 'labeled':
                # log the prediction of noisy or not
                assert(np.all(tar['true_label'] == np.array([data_dict['labels'][i] for i in tar['index']])))
                clean = (tar['pseudo_label'] == tar['true_label'])                                                       
                auc_meter = AUCMeter()
                auc_meter.reset()
                auc_meter.add(probability, clean)        
                auc, _, _ = auc_meter.value()               
                log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                log.flush()     

            weak_ids = tar['index']
            weak_labels = tar['pseudo_label']
            if self.mode == 'weak':
                self.inputs = inputs[weak_ids]
                self.labels = weak_labels
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
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
                    self.inputs = DictOfTensor({k: torch.cat([e.dic[k] for e in _inputs]) for k in inputs.dic})
                    # TODO: not sure if correct
                    # TODO: make it a class method
                
                # self.train_data = train_data[pred_idx]
                # self.noise_label = [noise_label[i] for i in pred_idx]                          
                # print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            

    def set_transform(self, perturbator, to_random_noise=False):

        if self.mode == 'labeled':
            if to_random_noise:
                def _transform(input_, label):
                    input_ad, offset = perturbator.remove_seed_words(input_, label)
                    assert(offset > 0), offset
                    return perturbator.remove_random_words(input_ad, self.config.randremove_num)
            else:
                _transform = lambda input_, label: perturbator.remove_random_words(input_, self.config.randremove_num)
        elif self.mode == 'unlabeled':
            # no label available for unlabeled, cannot remove seed word
            _transform = lambda input_: perturbator.remove_random_words(input_, self.config.randremove_num)
        elif self.mode == 'weak':
            if to_random_noise:
                def _transform(input_, label):
                    input_ad, offset = perturbator.remove_seed_words(input_, label)
                    # assert(offset > 0), offset
                    if not offset > 0:
                        self.count += 1
                        print('[%i] %i' % (self.count, label), end='\r')
                        # print(offset, input_, label)
                    return input_ad
            else:
                _transform = lambda input_, label: input_
        else:
            _transform = lambda input_, label: input_
            # raise KeyError(self.mode)

        if self.mode == 'unlabeled':
            def preprocessed_transform(input_):
                input_ = input_.dic
                input_ = _transform(input_)
                return input_
        else:
            def preprocessed_transform(input_, label):
                input_ = input_.dic
                input_ = _transform(input_, label)
                return input_ # {key: val.cuda() for key, val in input_.items()}

        self.transform = preprocessed_transform

    def __getitem__(self, index):
        if self.mode=='labeled':
            input_, label, prob = self.inputs[index], self.labels[index], self.probability[index]
            input1 = self.transform(input_, label) 
            input2 = self.transform(input_, label) 
            return input1, input2, label, prob            
        elif self.mode=='unlabeled':
            input_ = self.inputs[index]
            input1 = self.transform(input_) 
            input2 = self.transform(input_) 
            return input1, input2
        elif self.mode=='weak':
            input_, label = self.inputs[index], self.labels[index]
            input_ = self.transform(input_, label)
            return input_, label, index        
        elif self.mode=='test':
            input_, label = self.inputs[index], self.labels[index]
            input_ = self.transform(input_, label)
            return input_, label
           
    def __len__(self):
        return len(self.inputs)
        

class WeakSupDataloader:  
    def __init__(self, dataset, data_dir, batch_size, num_workers, log=None, config=None):
        self.dataset = dataset
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
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
        
    