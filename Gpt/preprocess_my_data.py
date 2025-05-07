import os
import json
import torch

# Read the existing Pth in sequence, then flatten the model parameters inside and store them again according to the index
#Generate a new JSON file to store metadata under different indices, such as class and p-acc


def get_param_sizes(state_dict):
    return torch.tensor([p.numel() for p in state_dict.values()], dtype=torch.long)

# Pull the model parameters into a 1-dimensional tensor
def get_flat_params(state_dict):
    parameters = []
    for parameter in state_dict.values():
        parameters.append(parameter.flatten())
    return torch.cat(parameters).cpu()



def parse_filename_correct(filename):
    # Decompose file name
    parts = filename.split('_')
    
    # 1. get model
    model = parts[0]
    
    # 2. get classes
    classes = parts[1:6]
    # 3. get personalized_acc
    acc_str = parts[6].replace("acc", "")  
    # last_number = parts[12].split('.')[0]   
    domain = parts[7].split('.')[0]
    personalized_acc = float(f"{acc_str}")
    
    
    change_dict = {'aquariumfish': 'aquarium fish', 'willowtree': 'willow tree', 'pinetree': 'pine tree',
                   'oaktree': 'oak tree', 'mapletree': 'maple tree', 'palmtree': 'palm tree', 'sweetpepper': 'sweet pepper',
                   'lawnmower': 'lawn mower', 'pickuptruck': 'pickup truck'}
    # change_dict = {'aircraftcarrier': 'aircraft carrier', 'alarmclock': 'alarm clock',
    #                'animalmigration': 'animal migration', 'baseballbat': 'baseball bat',
    #                'birthdaycake': 'birthday cake', 'ceilingfan': 'ceiling fan', 'cellphone': 'cell phone',
    #                'coffeecup': 'coffee cup','cruiseship': 'cruise ship', 'divingboard': 'diving board', 'firehydrant': 'fire hydrant',
    #                'flipflops': 'flip flops','floorlamp': 'floor lamp', 'flyingsaucer': 'flying saucer', 'fryingpan': 'frying pan',
    #                'gardenhose': 'garden hose','golfclub': 'golf club', 'hockeypuck': 'hockey puck', 'hockeystick': 'hockey stick',
    #                'hotairballoon': 'hot air balloon','hotdog': 'hot dog', 'hottub': 'hot tub', 'houseplant': 'house plant', 'icecream': 'ice cream',
    #                'lightbulb': 'light bulb', 'paintcan': 'paint can', 'palmtree': 'palm tree', 'paperclip': 'paper clip',
    #                'pickuptruck': 'pickup truck','pictureframe': 'picture frame', 'policecar': 'police car', 'poweroutlet': 'power outlet',
    #                'remotecontrol': 'remote control','rollercoaster': 'roller coaster', 'schoolbus': 'school bus', 'seaturtle': 'sea turtle',
    #                'seesaw': 'see saw','sleepingbag': 'sleeping bag', 'smileyface': 'smiley face', 'soccerball': 'soccer ball',
    #                'stopsign': 'stop sign','stringbean': 'string bean', 'swingset': 'swing set', 'tennisracquet': 'tennis racquet',
    #                'TheEiffelTower': 'The Eiffel Tower','TheGreatWallofChina': 'The Great Wall of China', 'TheMonaLisa': 'The Mona Lisa',
    #                'trafficlight': 'traffic light','washingmachine': 'washing machine', 'winebottle': 'wine bottle', 'wineglass': 'wine glass'}
    describe_class = []
    for index in range(len(classes)):
        key = classes[index]
        if key in change_dict.keys():
            classes[index] = change_dict[key]
        describe_class.append(str(domain) + ' ' +classes[index])

    # Combine into a JSON
    # result = {
    #     "model": model,
    #     "domain": domain,
    #     "describe_class": describe_class,
    #     "classes": classes,
    #     "personalized_acc": personalized_acc
    # }

    result = {
        "model": model,
        "classes": classes,
        "personalized_acc": personalized_acc
    }

    return result
    # return json.dumps(result, indent=4)





def parse_filename_correct_cifar20(filename):
    parts = filename.split('_')
    model = parts[0]
    classes = parts[1:21]
    acc_str = parts[21].replace("acc", "")
    last_number = parts[22].split('.')[0]
    domain = parts[22].split('.')[0]
    personalized_acc = float(f"{acc_str}")

    # change_dict = {'aquariumfish': 'aquarium fish', 'willowtree': 'willow tree', 'pinetree': 'pine tree',
    #                'oaktree': 'oak tree', 'mapletree': 'maple tree', 'palmtree': 'palm tree', 'sweetpepper': 'sweet pepper',
    #                'lawnmower': 'lawn mower', 'pickuptruck': 'pickup truck'}
    change_dict = {'Europeanfiresalamander': 'European fire salamander', 'tailedfrog': 'tailed frog', 
                   'Americanalligator': 'American alligator', 'boaconstrictor': 'boa constrictor', 'blackwidow': 'black widow', 
                   'braincoral': 'brain coral', 'seaslug': 'sea slug', 'Americanlobster': 'American lobster', 'spinylobster': 'spiny lobster', 
                   'blackstork': 'black stork', 'kingpenguin': 'king penguin', 'Yorkshireterrier': 'Yorkshire terrier', 
                   'goldenretriever': 'golden retriever', 'Labradorretriever': 'Labrador retriever', 'Germanshepherd': 'German shepherd', 
                   'standardpoodle': 'standard poodle', 'Persiancat': 'Persian cat', 'Egyptiancat': 'Egyptian cat', 'brownbear': 'brown bear', 
                   'walkingstick': 'walking stick', 'sulphurbutterfly': 'sulphur butterfly', 'seacucumber': 'sea cucumber', 'guineapig': 'guinea pig', 
                   'Arabiancamel': 'Arabian camel', 'Africanelephant': 'African elephant', 'lesserpanda': 'lesser panda', 
                   'academicgown': 'academic gown', 'beachwagon': 'beach wagon', 'beerbottle': 'beer bottle', 'bowtie': 'bow tie', 
                   'bullettrain': 'bullet train', 'butchershop': 'butcher shop', 'cashmachine': 'cash machine', 'CDplayer': 'CD player', 
                   'Christmasstocking': 'Christmas stocking', 'cliffdwelling': 'cliff dwelling', 'computerkeyboard': 'computer keyboard', 
                   'diningtable': 'dining table', 'freightcar': 'freight car', 'fryingpan': 'frying pan', 'furcoat': 'fur coat', 
                   'lawnmower': 'lawn mower', 'magneticcompass': 'magnetic compass', 'militaryuniform': 'military uniform', 
                   'movingvan': 'moving van', 'neckbrace': 'neck brace', 'parkingmeter': 'parking meter', 'picketfence': 'picket fence', 
                   'pillbottle': 'pill bottle', 'policevan': 'police van', 'popbottle': 'pop bottle', "potter'swheel": "potter's wheel", 
                   'punchingbag': 'punching bag', 'remotecontrol': 'remote control', 'rockingchair': 'rocking chair', 'rugbyball': 'rugby ball', 
                   'schoolbus': 'school bus', 'sewingmachine': 'sewing machine', 'spaceheater': 'space heater', 'spiderweb': 'spider web', 
                   'sportscar': 'sports car', 'steelarchbridge': 'steel arch bridge', 'suspensionbridge': 'suspension bridge', 
                   'swimmingtrunks': 'swimming trunks', 'triumphalarch': 'triumphal arch', 'waterjug': 'water jug', 'watertower': 'water tower', 
                   'woodenspoon': 'wooden spoon', 'comicbook': 'comic book', 'icecream': 'ice cream', 'icelolly': 'ice lolly', 
                   'mashedpotato': 'mashed potato', 'bellpepper': 'bell pepper', 'meatloaf': 'meat loaf', 'coralreef': 'coral reef'}
    describe_class = []
    for index in range(len(classes)):
        key = classes[index]
        if key in change_dict.keys():
            classes[index] = change_dict[key]
        # describe_class.append('domain: '+str(domain) + ' class:' +classes[index])

    result = {
        "model": model,
        "classes": classes,
        "personalized_acc": personalized_acc
    }

    # 转换成 JSON 字符串
    return result, last_number

def list_files(directory, save_dir):
    meta_data = {}
    entries = os.listdir(directory)
    # print(entries)
    # return 
    for idx, entry in enumerate(entries):
        result = parse_filename_correct(entry)
        full_path = os.path.join(directory, entry)
        # print(full_path)
        params = torch.load(full_path)
        p_acc = params['acc'] * 100.0
        params = params['model']

        result['personalized_acc'] = p_acc
        meta_data[idx] = result
        flat_params = get_flat_params(params)

        torch.save(flat_params, os.path.join(save_dir, f"{idx}.pth"))
    
    # print(get_param_sizes(params))

    meta_data['param_sizes'] = get_param_sizes(params).tolist()
    meta_data['num_data'] = len(entries)


    with open(os.path.join(save_dir, "meta_data.json"), 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)


# 使用函数
directory_path = './checkpoint_datasets/med/0328/conv5k_mom09_train_all/data/test'
save_dir = './checkpoint_datasets/med/0328/conv5k_mom09_train_all/test'
list_files(directory_path, save_dir)



