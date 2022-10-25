import torch
import numpy as np
import json
import os

def copy(tensor):
    return tensor.detach().clone()

def zeros(length, device):
    return torch.zeros(length, device=device, dtype=int)
    
class SeedWordPerturbator:
    def __init__(self, tokenizer, classes, label_names, dataset, data_dir):
        self.tokenizer = tokenizer
        self.classes = classes
        self.label_decode = lambda label: label_names[label]
        
        # load seed words dict
        with open(os.path.join(data_dir, dataset, "seedwords.json")) as fp:
            self.seed_words = json.load(fp)

        # encoded seed word look up dict
        self.seed_words_encoded = {}
        for c in self.classes:
            self.seed_words_encoded[c] = []
            for seed_word in self.seed_words[self.label_decode(c)]:
                self.seed_words_encoded[c].append(self.tokenizer(seed_word)['input_ids'][1])

    def random_seed_word(self, label):
        return np.random.choice(self.seed_words_encoded[label])

    def insert_element_random(self, tensor, element):
        nonzero_idx = tensor.nonzero().flatten() # [-1]
        insert_id = torch.randint(nonzero_idx[-1], (1,)).squeeze()
        new_tensor = torch.cat([tensor[:insert_id], torch.tensor([element], device=tensor.device), tensor[insert_id:]])
        return new_tensor[:len(tensor)]
        
    def insert_seed_word_batch(self, inputs, labels):
        input_ids = []
        for i, input_id in enumerate(inputs['input_ids']):
            seed_word = self.random_seed_word(labels[i].item())
            input_ids.append(self.insert_element_random(copy(input_id), seed_word))
        input_ids = torch.stack(input_ids)
        
        new_inputs = {}
        new_inputs['input_ids'] = input_ids
        new_inputs['token_type_ids'] = copy(inputs['token_type_ids'])
        # because insert at places where encoding is non-zero, attention mask should be the same
        new_inputs['attention_mask'] = copy(inputs['attention_mask'])
        return new_inputs

    # def remove_elements_from_tensor(self, tensor, elements):
    #     return torch.tensor([e.item() for e in tensor if e not in elements], device=tensor.device)

    def remove_elements_from_tensor(self, tensor, elements):
        is_in = torch.prod(torch.stack([(tensor != e) for e in elements]), dim=0)
        return tensor[is_in.nonzero().flatten()]

    def remove_seed_words(self, input_, label):
        device = input_['input_ids'].device
        input_id = input_['input_ids']
        seed_words = self.seed_words_encoded[label.item()]
        new_input_id = self.remove_elements_from_tensor(copy(input_id), seed_words)
        offset = len(input_id) - len(new_input_id)
        new_input_id = torch.cat([new_input_id, zeros(offset, device)]) # append zeros
        new_attention_mask = copy(input_['attention_mask'])[offset:] # remove ones from the beginning
        new_attention_mask = torch.cat([new_attention_mask, zeros(offset, device)]) # append zeros

        new_input = {}
        new_input['input_ids'] = new_input_id
        new_input['attention_mask'] = new_attention_mask
        new_input['token_type_ids'] = copy(input_['token_type_ids'])
        return new_input, offset

    def remove_seed_words_batch(self, inputs, labels):
        device = inputs['input_ids'][0].device
        input_ids = []
        attention_masks = []
        offsets = []
        for i, input_id in enumerate(inputs['input_ids']):
            seed_words = self.seed_words_encoded[labels[i].item()]
            new_input_id = self.remove_elements_from_tensor(copy(input_id), seed_words)
            offset = len(input_id) - len(new_input_id)
            new_input_id = torch.cat([new_input_id, zeros(offset, device)]) # append zeros
            new_attention_mask = copy(inputs['attention_mask'][i])[offset:] # remove ones from the beginning
            new_attention_mask = torch.cat([new_attention_mask, zeros(offset, device)]) # append zeros
            offsets.append(offset)
            input_ids.append(new_input_id)
            attention_masks.append(new_attention_mask)
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        offsets = torch.tensor(offsets, device=device)

        new_inputs = {}
        new_inputs['input_ids'] = input_ids
        new_inputs['attention_mask'] = attention_masks
        new_inputs['token_type_ids'] = copy(inputs['token_type_ids'])
        return new_inputs, offsets

    def remove_random_elements_from_tensor(self, tensor, num_remove):
        nonzero_idx = tensor.nonzero().flatten() # [-1]
        remove_idx = torch.randint(nonzero_idx[-1], (num_remove,)).unique() # note it is possible only 1 is removed even num_remove=2
        rest_idx = self.remove_elements_from_tensor(nonzero_idx, remove_idx)
        return tensor[rest_idx]

    def remove_random_words(self, input_, num_remove):
        device = input_['input_ids'].device
        new_input_id = self.remove_random_elements_from_tensor(copy(input_['input_ids']), num_remove=num_remove)
        offset = len(input_['input_ids']) - len(new_input_id)
        new_input_id = torch.cat([new_input_id, zeros(offset, device)]) # append zeros
        new_attention_mask = copy(input_['attention_mask'])[offset:] # remove ones from the beginning
        new_attention_mask = torch.cat([new_attention_mask, zeros(offset, device)]) # append zeros

        new_input = {}
        new_input['input_ids'] = new_input_id
        new_input['attention_mask'] = new_attention_mask
        new_input['token_type_ids'] = copy(input_['token_type_ids'])
        return new_input

    def remove_random_words_batch(self, inputs, num_remove):
        device = inputs['input_ids'][0].device
        input_ids = []
        attention_masks = []
        for i, input_id in enumerate(inputs['input_ids']):
            new_input_id = self.remove_random_elements_from_tensor(copy(input_id), num_remove=num_remove)
            offset = len(input_id) - len(new_input_id)
            new_input_id = torch.cat([new_input_id, zeros(offset, device)]) # append zeros
            new_attention_mask = copy(inputs['attention_mask'][i])[offset:] # remove ones from the beginning
            new_attention_mask = torch.cat([new_attention_mask, zeros(offset, device)]) # append zeros
            input_ids.append(new_input_id)
            attention_masks.append(new_attention_mask)
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)

        new_inputs = {}
        new_inputs['input_ids'] = input_ids
        new_inputs['attention_mask'] = attention_masks
        new_inputs['token_type_ids'] = copy(inputs['token_type_ids'])
        return new_inputs