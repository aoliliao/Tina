import clip
import timm
import torch
from timm.models import register_model
from timm.models.vision_transformer import _create_vision_transformer
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets



class ModelVit(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None):
        super(ModelVit, self).__init__()
        self.model = model
        self.fc = torch.nn.Linear(feature_dim, num_classes)
        self.normalize = normalize
        # if not self.normalize:
        #     print('normalize skipped.')

        if initial_weights is not None and type(initial_weights) == tuple:
            print('tuple.')
            w, b = initial_weights
            self.fc.weight = torch.nn.Parameter(w.clone())
            self.fc.bias = torch.nn.Parameter(b.clone())
        else:
            # if initial_weights is None:
            #     initial_weights = torch.zeros_like(self.classification_head.weight)
            #     torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
            # self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
            # # Note: modified. Initial bug in forgetting to zero bias.
            # self.classification_head.bias = torch.nn.Parameter(torch.zeros_like(self.classification_head.bias))
            pass

        # # Note: modified. Get rid of the language part.
        # delattr(self.model, 'transformer')

    def forward(self, images):
        # print(images.shape)
        features = self.model(images)
        # if self.normalize:
        #     features = features / features.norm(dim=-1, keepdim=True)
        logits = self.fc(features)
        return logits


def vit_base_224(num_classes=100):
    model, _ = clip.load('ViT-B/32')
    # model.head = nn.Sequential()
    model = ModelVit(model, 512, num_classes, False)
    return model


def vit_base_224_before(num_classes=10):
    model, _ = clip.load('ViT-B/32')
    return model

def vit_base_224_after(num_classes=10):
    model = nn.Sequential()
    model = ModelVit(model, 512, num_classes, False)
    return model

@register_model  # 注册模型
def vit_base_patch16_32(pretrained: bool = False, **kwargs):
    model_args = dict(img_size=32)
    #想要加载的预训练权重对应的模型
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def vit_base_32(num_classes=10):
    model = timm.create_model('vit_base_patch16_32', pretrained=False)
    model.head = nn.Sequential()
    model = ModelVit(model, 768, num_classes, False)
    return model

def vit_base_32_before(num_classes=10):
    model = timm.create_model('vit_base_patch16_32', pretrained=False)
    model.head = nn.Sequential()
    return model

def vit_base_32_after(num_classes=10):
    model = nn.Sequential()
    model = ModelVit(model, 768, num_classes, False)
    return model
