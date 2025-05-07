"""
This file contains metadata for different supervised learning and reinforcement learning tasks.

To define a new task, you should add a new sub-dictionary to TASK_METADATA below with the following keys:

(1) task_test_fn, a Python function of the form f(x, y, z, ..., nn.Module) that outputs loss/reward/etc.

(2) constructor, a Python function that returns a (random-init) nn.Module with the architecture G.pt is learning

(3) data_fn, a Python function that returns (x, y, z, ...); i.e., anything you want to cache for task_test_fn
             Note that data_fn must return a tuple or list, even if it returns only one thing

(4) minimize, a boolean indicating if the goal is to minimize the output of task_test_fn (False = maximize)

(5) best_prompt, the "best" loss/error/return/etc. you want to prompt G.pt with for one-step training

(6) recursive_prompt, the loss/error/return/etc. you want to prompt G.pt with for recursive optimization

(Optional) You can also include an 'aug_fn' key that maps to a function that performs a loss-preserving augmentation
           on the neural network parameters directly.

Whatever key you choose for your new task should be passed in with dataset.name.

See below for examples.
"""

import data_gen.train_cifar10
import data_gen.resnet_cifar
import  data_gen.resnet_nobn_cifar
import  data_gen.ResNet_domain
import data_gen.cnn_femnist
import data_gen.vit





CIFAR100_INDEX2CLASSTEXT = {19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium fish', 86: 'telephone', 90: 'train',
               28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow tree', 82: 'sunflower', 17: 'castle',
               71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine tree', 70: 'rose',
               87: 'television', 84: 'table', 64: 'possum', 52: 'oak tree', 42: 'leopard', 47: 'maple tree',
               65: 'rabbit', 21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake',
               45: 'lobster', 49: 'mountain', 56: 'palm tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark',
               14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal',
               43: 'lion', 51: 'mushroom', 35: 'girl', 83: 'sweet pepper', 33: 'forest', 27: 'crocodile', 53: 'orange',
               92: 'tulip', 50: 'mouse', 15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo',
               66: 'raccoon', 77: 'snail', 69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver',
               61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray',
               30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle', 79: 'spider',
               85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn mower',
               37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 60: 'plain', 91: 'trout',
               3: 'bear', 58: 'pickup truck', 16: 'can'}

CIFAR100_CLASSTEXT2INDEX = {"cattle": 19, "dinosaur": 29, "apple": 0, "boy": 11, "aquarium fish": 1, "telephone": 86, "train": 90,
 "cup": 28, "cloud": 23, "elephant": 31, "keyboard": 39, "willow tree": 96, "sunflower": 82, "castle": 17, "sea": 71, "bicycle": 8,
  "wolf": 97, "squirrel": 80, "shrew": 74, "pine tree": 59, "rose": 70, "television": 87, "table": 84, "possum": 64, "oak tree": 52,
   "leopard": 42, "maple tree": 47, "rabbit": 65, "chimpanzee": 21, "clock": 22, "streetcar": 81, "cockroach": 24, "snake": 78, 
   "lobster": 45, "mountain": 49, "palm tree": 56, "skyscraper": 76, "tractor": 89, "shark": 73, "butterfly": 14, "bottle": 9,
    "bee": 6, "chair": 20, "woman": 98, "hamster": 36, "otter": 55, "seal": 72, "lion": 43, "mushroom": 51, "girl": 35, "sweet pepper": 83,
     "forest": 33, "crocodile": 27, "orange": 53, "tulip": 92, "mouse": 50, "camel": 15, "caterpillar": 18, "man": 46, "skunk": 75,
      "kangaroo": 38, "raccoon": 66, "snail": 77, "rocket": 69, "whale": 95, "worm": 99, "turtle": 93, "beaver": 4, "plate": 61, 
      "wardrobe": 94, "road": 68, "fox": 34, "flatfish": 32, "tiger": 88, "ray": 67, "dolphin": 30, "poppy": 62, "porcupine": 63, 
      "lamp": 40, "crab": 26, "motorcycle": 48, "spider": 79, "tank": 85, "orchid": 54, "lizard": 44, "beetle": 7, "bridge": 12, 
      "baby": 2, "lawn mower": 41, "house": 37, "bus": 13, "couch": 25, "bowl": 10, "pear": 57, "bed": 5, "plain": 60, "trout": 91, 
      "bear": 3, "pickup truck": 58, "can": 16}


TINY_INDEX2CLASSTEXT = {0: 'goldfish', 1: 'European fire salamander', 2: 'bullfrog', 3: 'tailed frog', 4: 'American alligator', 
           5: 'boa constrictor', 6: 'trilobite', 7: 'scorpion', 8: 'black widow', 9: 'tarantula', 10: 'centipede', 11: 'goose', 
           12: 'koala', 13: 'jellyfish', 14: 'brain coral', 15: 'snail', 16: 'slug', 17: 'sea slug', 18: 'American lobster', 19: 'spiny lobster', 
           20: 'black stork', 21: 'king penguin', 22: 'albatross', 23: 'dugong', 24: 'Chihuahua', 25: 'Yorkshire terrier', 26: 'golden retriever', 
           27: 'Labrador retriever', 28: 'German shepherd', 29: 'standard poodle', 30: 'tabby', 31: 'Persian cat', 32: 'Egyptian cat', 
           33: 'cougar', 34: 'lion', 35: 'brown bear', 36: 'ladybug', 37: 'fly', 38: 'bee', 39: 'grasshopper', 40: 'walking stick', 
           41: 'cockroach', 42: 'mantis', 43: 'dragonfly', 44: 'monarch', 45: 'sulphur butterfly', 46: 'sea cucumber', 
           47: 'guinea pig', 48: 'hog', 49: 'ox', 50: 'bison', 51: 'bighorn', 52: 'gazelle', 53: 'Arabian camel', 54: 'orangutan', 55: 'chimpanzee', 
           56: 'baboon', 57: 'African elephant', 58: 'lesser panda', 59: 'abacus', 60: 'academic gown', 61: 'altar', 62: 'apron', 63: 'backpack', 
           64: 'bannister', 65: 'barbershop', 66: 'barn', 67: 'barrel', 68: 'basketball', 69: 'bathtub', 70: 'beach wagon', 71: 'beacon', 
           72: 'beaker', 73: 'beer bottle', 74: 'bikini', 75: 'binoculars', 76: 'birdhouse', 77: 'bow tie', 78: 'brass', 79: 'broom', 80: 'bucket', 
           81: 'bullet train', 82: 'butcher shop', 83: 'candle', 84: 'cannon', 85: 'cardigan', 86: 'cash machine', 87: 'CD player', 88: 'chain',
             89: 'chest', 90: 'Christmas stocking', 91: 'cliff dwelling', 92: 'computer keyboard', 93: 'confectionery', 94: 'convertible', 
             95: 'crane', 96: 'dam', 97: 'desk', 98: 'dining table', 99: 'drumstick', 100: 'dumbbell', 101: 'flagpole', 102: 'fountain', 
             103: 'freight car', 104: 'frying pan', 105: 'fur coat', 106: 'gasmask', 107: 'go-kart', 108: 'gondola', 109: 'hourglass', 
             110: 'iPod', 111: 'jinrikisha', 112: 'kimono', 113: 'lampshade', 114: 'lawn mower', 115: 'lifeboat', 116: 'limousine', 
             117: 'magnetic compass', 118: 'maypole', 119: 'military uniform', 120: 'miniskirt', 121: 'moving van', 122: 'nail', 123: 'neck brace', 
             124: 'obelisk', 125: 'oboe', 126: 'organ', 127: 'parking meter', 128: 'pay-phone', 129: 'picket fence', 130: 'pill bottle', 
             131: 'plunger', 132: 'pole', 133: 'police van', 134: 'poncho', 135: 'pop bottle', 136: "potter's wheel", 137: 'projectile', 
             138: 'punching bag', 139: 'reel', 140: 'refrigerator', 141: 'remote control', 142: 'rocking chair', 143: 'rugby ball', 144: 'sandal', 
             145: 'school bus', 146: 'scoreboard', 147: 'sewing machine', 148: 'snorkel', 149: 'sock', 150: 'sombrero', 151: 'space heater', 
             152: 'spider web', 153: 'sports car', 154: 'steel arch bridge', 155: 'stopwatch', 156: 'sunglasses', 157: 'suspension bridge', 
             158: 'swimming trunks', 159: 'syringe', 160: 'teapot', 161: 'teddy', 162: 'thatch', 163: 'torch', 164: 'tractor', 165: 'triumphal arch', 
             166: 'trolleybus', 167: 'turnstile', 168: 'umbrella', 169: 'vestment', 170: 'viaduct', 171: 'volleyball', 172: 'water jug', 
             173: 'water tower', 174: 'wok', 175: 'wooden spoon', 176: 'comic book', 177: 'plate', 178: 'guacamole', 179: 'ice cream', 
             180: 'ice lolly', 181: 'pretzel', 182: 'mashed potato', 183: 'cauliflower', 184: 'bell pepper', 185: 'mushroom', 186: 'orange', 
             187: 'lemon', 188: 'banana', 189: 'pomegranate', 190: 'meat loaf', 191: 'pizza', 192: 'potpie', 193: 'espresso', 194: 'alp', 
             195: 'cliff', 196: 'coral reef', 197: 'lakeside', 198: 'seashore', 199: 'acorn'}

TINY_CLASSTEXT2INDEX = {'goldfish': 0, 'European fire salamander': 1, 'bullfrog': 2, 'tailed frog': 3, 'American alligator': 4, 'boa constrictor': 5, 
                        'trilobite': 6, 'scorpion': 7, 'black widow': 8, 'tarantula': 9, 'centipede': 10, 'goose': 11, 'koala': 12, 'jellyfish': 13, 
                        'brain coral': 14, 'snail': 15, 'slug': 16, 'sea slug': 17, 'American lobster': 18, 'spiny lobster': 19, 'black stork': 20, 
                        'king penguin': 21, 'albatross': 22, 'dugong': 23, 'Chihuahua': 24, 'Yorkshire terrier': 25, 'golden retriever': 26, 
                        'Labrador retriever': 27, 'German shepherd': 28, 'standard poodle': 29, 'tabby': 30, 'Persian cat': 31, 'Egyptian cat': 32, 
                        'cougar': 33, 'lion': 34, 'brown bear': 35, 'ladybug': 36, 'fly': 37, 'bee': 38, 'grasshopper': 39, 'walking stick': 40, 
                        'cockroach': 41, 'mantis': 42, 'dragonfly': 43, 'monarch': 44, 'sulphur butterfly': 45, 'sea cucumber': 46, 'guinea pig': 47, 
                        'hog': 48, 'ox': 49, 'bison': 50, 'bighorn': 51, 'gazelle': 52, 'Arabian camel': 53, 'orangutan': 54, 'chimpanzee': 55, 
                        'baboon': 56, 'African elephant': 57, 'lesser panda': 58, 'abacus': 59, 'academic gown': 60, 'altar': 61, 'apron': 62, 
                        'backpack': 63, 'bannister': 64, 'barbershop': 65, 'barn': 66, 'barrel': 67, 'basketball': 68, 'bathtub': 69, 
                        'beach wagon': 70, 'beacon': 71, 'beaker': 72, 'beer bottle': 73, 'bikini': 74, 'binoculars': 75, 'birdhouse': 76, 
                        'bow tie': 77, 'brass': 78, 'broom': 79, 'bucket': 80, 'bullet train': 81, 'butcher shop': 82, 'candle': 83, 'cannon': 84, 
                        'cardigan': 85, 'cash machine': 86, 'CD player': 87, 'chain': 88, 'chest': 89, 'Christmas stocking': 90, 'cliff dwelling': 91, 
                        'computer keyboard': 92, 'confectionery': 93, 'convertible': 94, 'crane': 95, 'dam': 96, 'desk': 97, 'dining table': 98, 
                        'drumstick': 99, 'dumbbell': 100, 'flagpole': 101, 'fountain': 102, 'freight car': 103, 'frying pan': 104, 'fur coat': 105, 
                        'gasmask': 106, 'go-kart': 107, 'gondola': 108, 'hourglass': 109, 'iPod': 110, 'jinrikisha': 111, 'kimono': 112, 
                        'lampshade': 113, 'lawn mower': 114, 'lifeboat': 115, 'limousine': 116, 'magnetic compass': 117, 'maypole': 118, 
                        'military uniform': 119, 'miniskirt': 120, 'moving van': 121, 'nail': 122, 'neck brace': 123, 'obelisk': 124, 'oboe': 125, 
                        'organ': 126, 'parking meter': 127, 'pay-phone': 128, 'picket fence': 129, 'pill bottle': 130, 'plunger': 131, 'pole': 132, 
                        'police van': 133, 'poncho': 134, 'pop bottle': 135, "potter's wheel": 136, 'projectile': 137, 'punching bag': 138, 
                        'reel': 139, 'refrigerator': 140, 'remote control': 141, 'rocking chair': 142, 'rugby ball': 143, 'sandal': 144, 
                        'school bus': 145, 'scoreboard': 146, 'sewing machine': 147, 'snorkel': 148, 'sock': 149, 'sombrero': 150, 
                        'space heater': 151, 'spider web': 152, 'sports car': 153, 'steel arch bridge': 154, 'stopwatch': 155, 'sunglasses': 156, 
                        'suspension bridge': 157, 'swimming trunks': 158, 'syringe': 159, 'teapot': 160, 'teddy': 161, 'thatch': 162, 'torch': 163, 
                        'tractor': 164, 'triumphal arch': 165, 'trolleybus': 166, 'turnstile': 167, 'umbrella': 168, 'vestment': 169, 'viaduct': 170, 
                        'volleyball': 171, 'water jug': 172, 'water tower': 173, 'wok': 174, 'wooden spoon': 175, 'comic book': 176, 'plate': 177, 
                        'guacamole': 178, 'ice cream': 179, 'ice lolly': 180, 'pretzel': 181, 'mashed potato': 182, 'cauliflower': 183, 
                        'bell pepper': 184, 'mushroom': 185, 'orange': 186, 'lemon': 187, 'banana': 188, 'pomegranate': 189, 'meat loaf': 190, 
                        'pizza': 191, 'potpie': 192, 'espresso': 193, 'alp': 194, 'cliff': 195, 'coral reef': 196, 'lakeside': 197, 'seashore': 198, 
                        'acorn': 199}

MED10_INDEX2CLASSTEXT = {0: 'bladder', 1: 'femur-left', 2: 'femur-right', 3: 'heart', 4: 'kidney-left', 5: 'kidney-right', 
               6: 'liver', 7: 'lung-left', 8: 'lung-right', 9: 'pancreas', 10: 'spleen'}
MED10_CLASSTEXT2INDEX = {'bladder': 0, 'femur-left': 1, 'femur-right': 2, 'heart': 3, 'kidney-left': 4, 'kidney-right': 5,
                           'liver': 6, 'lung-left': 7, 'lung-right': 8, 'pancreas': 9, 'spleen': 10}

TASK_METADATA = {
    "med11_personalize5": {
        "task_test_fn": lambda *args, **kwargs: data_gen.train_cifar10.test_epoch(*args, **kwargs)[1],
        "constructor": data_gen.train_cifar10.ConvNet_5,
        "data_fn": data_gen.train_cifar10.unload_test_set,
        "aug_fn": data_gen.train_cifar10.random_permute_cnn,
        "minimize": True,
        "best_prompt": 35.0,
        "recursive_prompt": 45.0
    },
    "tiny200_personalize20": {
        "task_test_fn": lambda *args, **kwargs: data_gen.train_cifar10.test_epoch(*args, **kwargs)[1],
        # "constructor": data_gen.train_cifar10.ConvNet_20,
        "constructor": data_gen.resnet_cifar.ResNet20_after,
        "data_fn": data_gen.train_cifar10.unload_test_set,
        "aug_fn": data_gen.train_cifar10.random_permute_cnn,
        "minimize": True,
        "best_prompt": 35.0,
        "recursive_prompt": 45.0
    },
    "cifar100_personalize20": {
        "task_test_fn": lambda *args, **kwargs: data_gen.train_cifar10.test_epoch(*args, **kwargs)[1],
        "constructor": data_gen.train_cifar10.ConvNet_20,
        "data_fn": data_gen.train_cifar10.unload_test_set,
        "aug_fn": data_gen.train_cifar10.random_permute_cnn,
        "minimize": True,
        "best_prompt": 35.0,
        "recursive_prompt": 45.0
    },
    "cifar100_personalize10": {
        "task_test_fn": lambda *args, **kwargs: data_gen.train_cifar10.test_epoch(*args, **kwargs)[1],
        # "constructor": data_gen.resnet_nobn_cifar.ResNet20_after,
        # "constructor": data_gen.resnet_cifar.ResNet20_after,
        # "constructor": data_gen.train_cifar10.ConvNet,
        "constructor": data_gen.train_cifar10.ConvNet10k,
        # "constructor": data_gen.train_cifar10.ConvNet33k,
        # "constructor": data_gen.train_cifar10.ConvNet76k,
        # "constructor": data_gen.vit.vit_base_224_after,
        "data_fn": data_gen.train_cifar10.unload_test_set,
        "aug_fn": data_gen.train_cifar10.random_permute_cnn,
        "minimize": True,
        "best_prompt": 35.0,
        "recursive_prompt": 45.0
    },

    "cifar10_loss": {
        "task_test_fn": data_gen.train_cifar10.test_epoch,
        "constructor": data_gen.train_cifar10.ConvNet,
        "data_fn": data_gen.train_cifar10.unload_test_set,
        "aug_fn": data_gen.train_cifar10.random_permute_cnn,
        "minimize": True,
        "best_prompt": 1.2,
        "recursive_prompt": 1.4
    },
    "cifar10_error": {
        "task_test_fn": lambda *args, **kwargs: data_gen.train_cifar10.test_epoch(*args, **kwargs)[1],
        "constructor": data_gen.train_cifar10.ConvNet,
        "data_fn": data_gen.train_cifar10.unload_test_set,
        "aug_fn": data_gen.train_cifar10.random_permute_cnn,
        "minimize": True,
        "best_prompt": 35.0,
        "recursive_prompt": 45.0
    }
}


def get(dataset_name, key):
    return TASK_METADATA[dataset_name][key]


