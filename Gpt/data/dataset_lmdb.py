"""
The main Dataset class for training/testing on checkpoint data (parameters, metrics, etc.).
"""
import json
import os

from dataclasses import dataclass
from glob import glob
from typing import Any, Dict
import random
import torch
from torch.utils.data import Dataset
import copy
from .database import Database
from .augment import random_permute_flat
from .normalization import get_normalizer
from Gpt.tasks import TASK_METADATA
from Gpt.download import find_checkpoints
from Gpt.vis import moduleify_single


def get_flat_params(state_dict):
    parameters = []
    for parameter in state_dict.values():
        parameters.append(parameter.flatten())
    return torch.cat(parameters).cpu()


@dataclass
class ParameterDataset(Dataset):
    dataset_dir: str = "/home/text-to-model/Text-to-Model-main/Gpt/checkpoint_datasets/cifar_data"  # Path to the checkpoint dataset
    dataset_name: str = "cifar100_personalize10"                # Name of the dataset in tasks.py
    split: str = "train"                            # Split of the checkpoint dataset to use ("train" or "test")
    max_train_runs: int = 1000000                   # Maximum number of runs to train on (default: all)
    num_test_runs: int = 500                        # Number of runs in the test split
    target_epoch_size: int = 806400                 # Amount of data to train on per-"epoch" (806400 is arbitrary)
    train_metric: str = "avg_test_loss"             # Conditional metric for G.pt
    min_step_spacing: int = 1                       # Minimum spacing between starting and future checkpoints
    max_step_spacing: int = None                    # Maximum spacing between starting and future checkpoints
    normalizer_name: str = "none"                 # Parameter normalization algorithm
    openai_coeff: float = 4.185                      # Scaling coefficient for the "openai" DALL-E 2 normalizer
    single_run_debug: bool = False                  # Option to debug with a single run
    min_val: float = None                            # Minimum value of parameters (usually passed to test dataset)
    max_val: float = None                            # Maximum value of parameters (usually passed to test dataset)
    permute_augment: bool = False                   # if True, applies permutation augmentation to parameters
    verify_done: bool = False                       # if True, filters runs that don't have a DONE.txt file
    download: bool = False                           # If True, auto-downloads and caches the checkpoint dataset

    def __post_init__(self):

        # Load the json metadata
        json_path = os.path.join(self.dataset_dir, self.split , 'meta_data.json')
        with open(json_path, 'r', encoding='utf-8') as file:
            self.run_jsons = json.load(file)
        self.length = self.run_jsons["num_data"]
        self.runs = [idx for idx in range(self.length)]

        # Build metadata details:
        self.num_data = self.run_jsons["num_data"]
        self.architecture = TASK_METADATA[self.dataset_name]['constructor']()
        self.parameter_sizes = self.run_jsons["param_sizes"]
        self.parameter_names = list(self.architecture.state_dict().keys())
        assert len(self.parameter_names) == len(self.parameter_sizes)

        # Setup permutation aug:
        self.make_permut_aug_fn()


    def get_database(self, run_index):
        data_path = os.path.join(self.dataset_dir, self.split , f'{run_index}.pth')
        data = torch.load(data_path)
        return data

    def make_permut_aug_fn(self):
        task_dict = TASK_METADATA[self.dataset_name]
        self.use_augment = self.permute_augment and 'aug_fn' in task_dict
        if self.use_augment:
            self.task_aug_fn = task_dict['aug_fn']
            print(f'(split={self.split}) Using augmentation')
        else:
            print(f'(split={self.split}) NOT using augmentation')

    def aug_fn(self, p, seed=None):
        if self.use_augment:
            return random_permute_flat(p, self.architecture, seed, self.task_aug_fn)
        else:
            return p

    # Classifier aug. by swaping the classes' positions
    def classifier_aug_fn(self, p, classes, seed=None):
        net_constructor = TASK_METADATA[self.dataset_name]['constructor']
        model = moduleify_single(p, net_constructor)

        model_dict = model.state_dict()

        new_classes = copy.deepcopy(classes)
        random.shuffle(new_classes)


        original_weights = model_dict['fc.weight'].clone()
        original_bias = model_dict['fc.bias'].clone()

        new_weights = torch.zeros_like(original_weights)
        new_bias = torch.zeros_like(original_bias)

        for idx, new_class in enumerate(new_classes):
            original_idx = classes.index(new_class)
            new_weights[idx] = original_weights[original_idx]
            new_bias[idx] = original_bias[original_idx]

        model_dict['fc.weight'] = new_weights
        model_dict['fc.bias'] = new_bias

        new_p = get_flat_params(model_dict)

        return new_p, new_classes

    #TODO medmnist

    # def classifier_aug_fn(self, p, classes, describe_class, seed=None):
    #     net_constructor = TASK_METADATA[self.dataset_name]['constructor']
    #     model = moduleify_single(p, net_constructor)
    
    #     model_dict = model.state_dict()
    
    #     new_classes = copy.deepcopy(classes)
    #     describe_class = copy.deepcopy(describe_class)
    
    #     indices = list(range(len(new_classes)))
    #     random.shuffle(indices)
    #     new_classes = [new_classes[i] for i in indices]
    #     describe_class = [describe_class[i] for i in indices]
    
    #     original_weights = model_dict['fc.weight'].clone()
    #     original_bias = model_dict['fc.bias'].clone()
    
    #     new_weights = torch.zeros_like(original_weights)
    #     new_bias = torch.zeros_like(original_bias)
    
    #     for idx, new_class in enumerate(new_classes):
    #         original_idx = classes.index(new_class)
    #         new_weights[idx] = original_weights[original_idx]
    #         new_bias[idx] = original_bias[original_idx]
    
    #     model_dict['fc.weight'] = new_weights
    #     model_dict['fc.bias'] = new_bias
    
    #     new_p = get_flat_params(model_dict)
    
    #     return new_p, new_classes, describe_class

    def get_run_p_acc(self, run_index: int):
        if self.single_run_debug:
            run_index = 0
        p_acc = self.run_jsons[str(run_index)]["personalized_acc"]
        return p_acc

    def get_run_network(self, run_index, iter=0, normalize=False, augment=False):
        self.get_database(run_index)

    def __getitem__(self, run_index: int) -> Dict[str, Any]:

        parameters = self.get_database(run_index)
        classes = self.run_jsons[str(run_index)]["classes"]
        p_acc = self.get_run_p_acc(run_index)

        

        if self.use_augment:
            # Permutation invariance aug (maybe is not effective in our setting)
            # parameters = self.aug_fn(parameters, seed=None)

            # Classifier aug
            parameters, classes = self.classifier_aug_fn(parameters, classes)

        #TODO medmnist
        # domain = self.run_jsons[str(run_index)]["domain"]
        # describe_class = self.run_jsons[str(run_index)]["describe_class"]
        # if self.use_augment:
        #     # Classifier aug
        #     parameters, classes, describe_class = self.classifier_aug_fn(parameters, classes, describe_class)

        # outputs = {
        #     "parameters": parameters,
        #     "classes": classes,
        #     "p_acc": p_acc,
        #     "domain": domain,
        #     "describe_class": describe_class,
        # }

        outputs = {
            "parameters": parameters,
            "classes": classes,
            "p_acc": p_acc
        }

        return outputs

    def __len__(self) -> int:
        return self.length
