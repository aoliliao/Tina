"""
Contains classes and function for generating G.pt visuals and evaluation metrics.
"""
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import os
import clip
from Gpt.data.all_domainnet import DomainNetImageList
from Gpt.tasks import CIFAR100_INDEX2CLASSTEXT, CIFAR100_CLASSTEXT2INDEX,TINY_CLASSTEXT2INDEX, MED10_CLASSTEXT2INDEX
from Gpt.tasks import TASK_METADATA
from Gpt.diffusion.gaussian_diffusion import GaussianDiffusion
from Gpt.distributed import all_gather, synchronize, is_main_proc, rank0_to_all
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets
from typing import Callable, Any, Dict
import time
from data_gen.ResNet_domain import resnet18_bef
from data_gen.resnet_cifar import ResNet20_before
# from data_gen.resnet_nobn_cifar import ResNet20_before
from data_gen.vit import vit_base_32_before, vit_base_224_before
import medmnist
from medmnist import INFO, Evaluator

def synth(
    diffusion,
    G,
    batch_dict,             # Test data
    clip_denoised=False,
    param_range=None,
    thresholding="none",
    **p_sample_loop_kwargs
):
    """
    Samples from G.pt via the reverse diffusion process.
    Specifically, this function draws a sample from p(theta^*|prompt_loss,starting_loss,starting_theta).
    """

    if param_range is not None:
        assert param_range[0] < param_range[1]
        denoised_fn = create_thresholding_fn(thresholding, param_range)
        clip_denoised = False
        # print(f"Using thresholding={thresholding} with min_val={param_range[0]}, max_val={param_range[1]}")
    else:
        denoised_fn = None

    w_test = batch_dict["parameters"].cuda()

    classes = batch_dict["classes"]

    model_kwargs = {
        'classes': classes
    }

    # TODO medmnist !!
    # model_kwargs = {
    #     'classes': batch_dict["describe_class"]
    # }

    shape = w_test.shape
    time_start = time.time()
    sample = diffusion.p_sample_loop(
        G,
        shape,
        clip_denoised=clip_denoised,
        model_kwargs=model_kwargs,
        device='cuda',
        denoised_fn=denoised_fn,
        **p_sample_loop_kwargs
    )
    time_end = time.time()  
    time_sum = time_end - time_start
    print(time_sum)
    return sample

def test_diffusion_model(diffusion, G, batch_dict):
    """
    Given a trained diffusion model, generate the models by the diffusion.
    Test the generated models by its personalized datasets.
    Return the upper bound of the test personalized acc and the test accuracy of diffused models.
    """
    
    # time_start = time.time()
    flat_ws = synth(
        diffusion, G, batch_dict
    )
    # time_end = time.time() 
    # time_sum = time_end - time_start 
    # print(time_sum)
    classes = batch_dict["classes"]
    # print('classes:',classes)
    new_classes = [[classes[i][j] for i in range(len(classes))] for j in range(len(classes[0]))]
    assert len(new_classes[0]) == 5
    
    # assert len(new_classes[0]) == 20
    result_acc = {}
    key_counts = {}
    p_acc_list = []
    target_acc_list = []

    result_auc = {}
    p_auc_list = []
    for i in range(len(flat_ws)):
        generated_single_dict = {}
        generated_single_dict["parameters"] = flat_ws[i]
        generated_single_dict["classes"] = new_classes[i]
        generated_single_dict["p_acc"] =  batch_dict["p_acc"][i]

        #TODO medmnist

        # generated_single_dict["domain"] = batch_dict["domain"][i]


        # (target_p_acc, p_acc) = test_single_model_personalization(generated_single_dict,i)
        (target_p_acc, p_acc, p_auc) = test_single_model_personalization(generated_single_dict,i)

        key = generated_single_dict["domain"]
        value = p_acc
        if key in result_acc:
            result_acc[key] += value
            key_counts[key] += 1
        else:
            result_acc[key] = value
            key_counts[key] = 1

        if key in result_auc:
            result_auc[key] += p_auc
        else:
            result_auc[key] = p_auc

        print("(target_p_acc, p_acc, p_auc)", (target_p_acc, p_acc, p_auc))
        target_acc_list.append(target_p_acc)
        p_acc_list.append(p_acc)
        p_auc_list.append(p_auc)
    
    final_target_p_acc = sum(target_acc_list) / len(target_acc_list)
    final_p_acc = sum(p_acc_list) / len(p_acc_list)
    final_p_auc = sum(p_auc_list) / len(p_auc_list)
    for key in result_acc:
        result_acc[key] /= key_counts[key]
    for key in result_auc:
        result_auc[key] /= key_counts[key]
    print(result_auc)
    print(result_acc, key_counts)
    return final_target_p_acc, final_p_acc, final_p_auc


def moduleify_single(Gpt_output, net_constructor):
    """
    Make one flat_w (Gpt_output) into a model instance.

    Gpt_output: (N, D) tensor (N = batch_size, D = number of parameters)
    net_constructor: Function (should take no args/kwargs) that returns a randomly-initialized neural network
                     with the appropriate architecture
    unnormalize_fn: Function that takes a (N, D) tensor and "unnormalizes" it back to the original parameter space

    Returns: A length-N list of nn.Module instances, where the i-th nn.Module has the parameters from Gpt_output[i].
             If N = 1, then a single nn.Module is returned instead of a list.
    """
    Gpt_output = Gpt_output.unsqueeze(0)
    num_nets = Gpt_output.size(0)
    net_instance = net_constructor()
    target_state_dict = net_instance.state_dict()
    parameter_names = target_state_dict.keys()
    parameter_sizes = [v.size() for v in target_state_dict.values()]
    parameter_chunks = [v.numel() for v in target_state_dict.values()]

    parameters = torch.split(Gpt_output, parameter_chunks, dim=1)
    modules = []
    for i in range(num_nets):
        net = net_constructor()
        # Build a state dict from the generated parameters:
        state_dict = {
            pname: param[i].reshape(size) for pname, param, size in \
                zip(parameter_names, parameters, parameter_sizes)
        }
        net.load_state_dict(state_dict, strict=True)
        modules.append(net.cuda())

    # return a single network
    return modules[0]



def moduleify_multiple(Gpt_output, net_constructor):
    """
    Make multiple flat_ws (Gpt_output) into model instances.

    Gpt_output: (N, D) tensor (N = batch_size, D = number of parameters)
    net_constructor: Function (should take no args/kwargs) that returns a randomly-initialized neural network
                     with the appropriate architecture
    unnormalize_fn: Function that takes a (N, D) tensor and "unnormalizes" it back to the original parameter space

    Returns: A length-N list of nn.Module instances, where the i-th nn.Module has the parameters from Gpt_output[i].
             If N = 1, then a single nn.Module is returned instead of a list.
    """
    num_nets = Gpt_output.size(0)
    net_instance = net_constructor()
    target_state_dict = net_instance.state_dict()
    parameter_names = target_state_dict.keys()
    parameter_sizes = [v.size() for v in target_state_dict.values()]
    parameter_chunks = [v.numel() for v in target_state_dict.values()]

    parameters = torch.split(Gpt_output, parameter_chunks, dim=1)
    modules = []
    for i in range(num_nets):
        net = net_constructor()
        # Build a state dict from the generated parameters:
        state_dict = {
            pname: param[i].reshape(size) for pname, param, size in \
                zip(parameter_names, parameters, parameter_sizes)
        }
        net.load_state_dict(state_dict, strict=True)
        modules.append(net.cuda())

    # return a list of networks
    return modules


# Test one personalized model
def test_single_model_personalization(batch_dict,index):
    """
    Given a data point, construct the personalized dataset, and verify the personalization.
    """
    flat_w = batch_dict["parameters"].cuda()
    target_p_acc = batch_dict["p_acc"]
    prompt_classes = batch_dict["classes"]

    # #TODO cifar100

    # # # Generate model objective
    # net_constructor = TASK_METADATA["cifar100_personalize10"]['constructor']
    # model = moduleify_single(flat_w, net_constructor)
    # # Craft personalized data loader
    # CIFAR_PATH = "./Dataset/cifar"
    # # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform = transforms.Compose([
    #     # transforms.Resize([224, 224]),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # testset = datasets.CIFAR100(root=CIFAR_PATH, train=False, download=False, transform=transform)
    # num_workers = 4
    # class_list = testset.classes
    # # print(class_list)
    # # {19: 'cattle', 29: 'dinosaur'} etc...
    # task_labels = {CIFAR100_CLASSTEXT2INDEX[prompt_class]:prompt_class for prompt_class in prompt_classes}
    # selected_indices = [idx for idx, label in enumerate(testset.targets) if label in task_labels.keys()]
    # test_data = torch.utils.data.Subset(testset, selected_indices)

    # TODO med10_person5
    # # # Generate model objective
    # net_constructor = TASK_METADATA["med11_personalize5"]['constructor']
    # model = moduleify_single(flat_w, net_constructor)
    # # Craft personalized data loader
    # transform = transforms.Compose([
    #     transforms.Resize([32, 32]),
    #     transforms.ToTensor(),
    # ])
   
    # data_flag = 'organamnist'
    # download = False
    # info = INFO[data_flag]
    # task = info['task']
    # DataClass = getattr(medmnist, info['python_class'])
    # testset = DataClass(split='test', transform=transform, download=download)
    # num_workers = 4
    # # class_list = testset.classes
    # # print(class_list)
    # # {19: 'cattle', 29: 'dinosaur'} etc...
    # task_labels = {MED10_CLASSTEXT2INDEX[prompt_class]:prompt_class for prompt_class in prompt_classes}
    # # print(task_labels)
    # # print(type(testset.labels),testset.labels.astype(int))
    # selected_indices = [idx for idx, label in enumerate(testset.labels) if label in list(task_labels.keys())]
    # test_data = torch.utils.data.Subset(testset, selected_indices)

    # TODO all_med
    net_constructor = TASK_METADATA["med11_personalize5"]['constructor']
    model = moduleify_single(flat_w, net_constructor)
    # Craft personalized data loader
    transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    data_flag_a = 'organamnist'
    data_flag_c = 'organcmnist'
    data_flag_s = 'organsmnist'
    download = False
    info_a = INFO[data_flag_a]
    DataClass_a = getattr(medmnist, info_a['python_class'])
    info_c = INFO[data_flag_c]
    DataClass_c = getattr(medmnist, info_c['python_class'])
    info_s = INFO[data_flag_s]
    DataClass_s = getattr(medmnist, info_s['python_class'])
    domain = batch_dict["domain"]
    # print(domain)
    if domain == 'Axial':
        DataClass = DataClass_a
    elif domain == 'Coronal':
        DataClass = DataClass_c
    elif domain == 'Sagittal':
        DataClass = DataClass_s
    testset = DataClass(split='test', transform=transform, download=download)
    num_workers = 4
   
    task_labels = {MED10_CLASSTEXT2INDEX[prompt_class]:prompt_class for prompt_class in prompt_classes}
    # print(task_labels)
    # print(type(testset.labels),testset.labels.astype(int))
    selected_indices = [idx for idx, label in enumerate(testset.labels) if label in list(task_labels.keys())]
    test_data = torch.utils.data.Subset(testset, selected_indices)

    # TODO tiny200
    # # Generate model objective
    # net_constructor = TASK_METADATA["tiny200_personalize20"]['constructor']
    # model = moduleify_single(flat_w, net_constructor)
    # # Craft personalized data loader
    # TINY_PATH = "./Dataset/tiny-imagenet-200"
    # transform_train = transforms.Compose([
    #     transforms.Resize([32, 32]),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    # ])
    # testset = datasets.ImageFolder(f'{TINY_PATH}/val', transform_train)
    # class_list = testset.classes
    # # print(class_list)
    # # {19: 'cattle', 29: 'dinosaur'} etc...
    # task_labels = {TINY_CLASSTEXT2INDEX[prompt_class]:prompt_class for prompt_class in prompt_classes}
    # selected_indices = [idx for idx, label in enumerate(testset.targets) if label in task_labels.keys()]
    # test_data = torch.utils.data.Subset(testset, selected_indices)


    # Map the data index to from 100 to 10
    new_label_mapping = {}
    for i, key in enumerate(task_labels.keys()):
        new_label_mapping[key] = i
    # testset = [(img, new_label_mapping[original_label]) for img, original_label in test_data]
    testset = [(img, new_label_mapping[int(original_label.squeeze())]) for img, original_label in test_data]
    personalized_data_loader = test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    # TODO resnet(cifar100)
    # before_model_path = './Gpt/checkpoint_datasets/cifar_res_class/0511'
    # before_model = ResNet20_before().cuda()
    # model_path = os.path.join(before_model_path, "res_before_stageone" + ".pth")
    # assert (os.path.exists(model_path))
    # state_dict = torch.load(model_path, map_location='cuda')
    # before_model.load_state_dict(state_dict['model'])

    # TODO resnet(tiny)
    # before_model_path = './Gpt/checkpoint_datasets/tinyImage/res'
    # before_model = ResNet20_before().cuda()
    # model_path = os.path.join(before_model_path, "res_before_stageone" + ".pth")
    # assert (os.path.exists(model_path))
    # state_dict = torch.load(model_path, map_location='cuda')
    # before_model.load_state_dict(state_dict['model'])

    # TODO resnet(domainnet)
    # before_model_path = './Gpt/checkpoint_datasets/real_domain/res18'
    # before_model = resnet18_bef().cuda()
    # # before_model_path = './Gpt/checkpoint_datasets/vit_domain'
    # # before_model = vit_base_32_before().cuda()
    # # before_model_path = './Gpt/checkpoint_datasets/res20_domain'
    # # before_model = ResNet20_before().cuda()
    # model_path = os.path.join(before_model_path, "res_real_before_stageone" + ".pth")
    # assert (os.path.exists(model_path))
    # state_dict = torch.load(model_path, map_location='cuda')
    # before_model.load_state_dict(state_dict['model'])

    # before_model_path = './Gpt/checkpoint_datasets/0803_cifar_vit'
    # before_model = vit_base_224_before().cuda()
    # model_path = os.path.join(before_model_path, "vit_before_stageone" + ".pth")
    # assert (os.path.exists(model_path))
    # state_dict = torch.load(model_path, map_location='cuda')
    # before_model.load_state_dict(state_dict['model'])

    # # Test the acc
    # correct = 0.0
    # model.eval()
    # before_model.eval()
    # with torch.no_grad():
    #     for idx, (data, target) in enumerate(personalized_data_loader):
    #         data, target = data.cuda(), target.cuda()
    #         before_out = before_model(data)
    #         # before_out = before_model.encode_image(data).float()
    #         output = model(before_out)
    #         # output = model(data)
    #         pred = output.argmax(dim=1)
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    #     p_acc = correct / len(test_loader.dataset) * 100

    # Test the acc
    from sklearn import metrics
    from sklearn.preprocessing import label_binarize
    y_prob = []
    y_true = []
    correct = 0.0
    model.eval()
    # before_model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(personalized_data_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

            y_prob.append(output.softmax(dim=-1).detach().cpu().numpy())
            nc = 5
            if nc == 2:
                nc += 1
            lb = label_binarize(target.detach().cpu().numpy(), classes=np.arange(nc))
            if nc == 2:
                lb = lb[:, :2]
            y_true.append(lb)
        
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        # p_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        p_auc = metrics.roc_auc_score(y_true, y_prob)

        p_acc = correct / len(test_loader.dataset) * 100
    
    # model_path = os.path.join('./gen_netparam', 'nc15R3')
    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)
    # model_path = os.path.join(model_path, f"{index}" + ".pth")
    # torch.save({'flat_w': flat_w,
    #             'model' : model.state_dict(),
    #             'target_p_acc': target_p_acc,
    #             'prompt_classes': prompt_classes,
    #             'p_acc': p_acc}, model_path)

    # return (target_p_acc, p_acc)
    return (target_p_acc, p_acc, p_auc)



def create_thresholding_fn(thresholding, param_range):
    """
    Creates a function that thresholds after each diffusion sampling step.

    thresholding = "none": No thresholding.
    thresholding = "static": Clamp the sample to param_range.
    """

    if thresholding == "none":
        def denoised_fn(x):
            return x
    elif thresholding == "static":
        def denoised_fn(x):
            return torch.clamp(x, param_range[0], param_range[1])
    else:
        raise NotImplementedError

    return denoised_fn

def r_squared(preds, targets):
    """
    Computes R^2 for a batch of predictions and targets and then averages over the batch dimension.
    """
    assert preds.size() == targets.size()
    assert preds.dim() > 1
    avg = targets.mean(dim=1, keepdims=True)
    var = (targets - avg).pow(2).sum(dim=1)
    err = (targets - preds).pow(2).sum(dim=1)
    r2 = 1.0 - err / (var + 1e-8)
    r2 = r2.mean().item()  # take average of R2s over batch dimension
    return r2