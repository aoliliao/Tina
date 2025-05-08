## Text-to-Model: Conditional Neural Network Diffusion for Generating Personalized Models from Texts

### How to run the code


#### 1. Prepare the Data

Place the personalized models (P-models) used for training into the `./Gpt/checkpoint_datasets/` directory.  
It's recommended to follow this filename format:

```
conv_apple_crocodile_plate_spider_couch_possum_rabbit_sea_pinetree_raccoon_acc71_2.pth
```

Each part of the filename is separated by an underscore `_`, which includes:
- **Network architecture** (e.g., `conv`)
- **Class names** used in classification (e.g., `apple`, `crocodile`, etc.)
- **Model accuracy** (e.g., `acc71`)
- **An optional index or identifier** (e.g., `2`)

#### ➤ Convert P-models to Training Data

Run the following script to convert personalized models into the experimental dataset format:

```bash
python Gpt/preprocess_my_data.py
```

This will generate a new dataset under:

```
./Gpt/checkpoint_datasets/cifar_data  # You can customize this name
```

---

#### 2.Run the Experiment

####  Configure Parameters

Modify the corresponding `.yaml` configuration file located in:

```
./config/
```

Each YAML file contains:
- Dataset and output paths
- Model save locations
- Tina’s hyperparameters
- Other relevant experimental settings

You can adjust most parameters directly within the YAML file.

####  Start Training

Use the following command to run the main script:

```bash
python main.py
```

Make sure to set the YAML configuration correctly by modifying the `@hydra.main` decorator in `main.py` if needed.

---

#### 3. Dataset and Metadata Settings

- Place the **image dataset** used for evaluation in:

```
./Dataset/
```

- The file `./Gpt/task.py` defines:
  - Metadata settings for personalized tasks
  - Mapping between class names and their corresponding dataset labels
  - Data loading logic for each personalized task

---

#### 4. Model Generation and Testing

The script `./Gpt/vis.py` is responsible for:
- Implementing Tina's model generation process
- Testing the quality and performance of generated models



### reference
This code is modified by the codebase [Learning to Learn with Generative Models of Neural Network Checkpoints](https://github.com/wpeebles/G.pt)
