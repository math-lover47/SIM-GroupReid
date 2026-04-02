# Social Interaction Modeling for Group Re-identification
Group Re-identification (G-ReID) focus on associating group images that contain the same members across different camera views.
The key challenge is that identity differentiation and position differentiation in group topology structure changes are difficult
to capture. According to the social psychology principles, we found that the core members are more likely to remain in the group 
with smaller position changes, and peripheral members are more likely to have significant position changes or even fade out of the group.
To this end, we propose a novel social interaction modeling (SIM), which treats group as a social interaction field, explore more 
authentic and robustness group features through the member differentiation. The member differentiation contains identity and position 
differentiation. Our method constructs the social interaction calculation module (SICM) to capture the member differentiation in fields, 
and implements identity differentiation and position differentiation by the social prior attention mechanism (SPAM) and social layout 
variation module (SLVM), respectively. A large number of experiments have been conducted on three available datasets show that the proposed 
method SIM is effective, and outperforms all previous state-of-the-art methods, surpassing the baseline on Rank1/mAP by up to 8.6\%/9.6\% 
on DukeGroup, 3.7\%/2.7\% on RoadGroup and 2.5\%/2.9\% on CSG. Code will be available on github.

Pipline of SIM was shown below.
![image](./assets/pipline.png)

Figure \ref{fig:fig4} presents the top-3 retrieval visualizations comparing baseline UMSOT and our proposed SIM.
The advantages of SIM are primarily demonstrated in two aspects:
1) Achieve core member mining and enhance group feature learning, achieve identity differentiated learning.
UMSOT tends to retrieve gallery images with higher overall similarity to the query.
When processing groups with less distinctive members (Rows 1, 2, and 4), SIM effectively focuses on core members of the group.
2) Conduct more realistic layout modeling and explore potential layout changes,
achieve position differentiated learning.
UMSOT does not emphasize layout variations.
For groups in Rows 3 and 5, SIM better captures the topological changes of the group structure.
![image](./assets/retrieval.jpg)


The implementation for SIM, Social Interaction Modeling for Group Re-identification.

Our backbone was based UMSOT(2024IJCV_Uncertainty Modeling for Group Re-Introduction)

Backbone(SPAM & SLVM in SIM)
Backbone refer to [group_vit](./fastreid/modeling/backbones/group_vit.py)fastreid/modeling/backbones/group_vit.py

## Requirements
### Installation    
Please refer to [INSTALL.md](INSTALL.md).

### Datasets
{dataset_name}: CSG, RoadGroup and DukeGroup.

Download the CSG dataset and modify the dataset path, line 26 in [csg.py](./fastreid/data/datasets/CSG.py) (./fastreid/data/datasets/CSG.py):
> self.root = XXX

Please read [README.md](./datasets/README.md) for more details.

### Prepare ViT Pre-trained Models
Download the ViT Pre-trained model and modify the path, line 11 in [bagtricks_gvit.yml](./configs/CSG/bagtricks_gvit.yml) (./configs/CSG/bagtricks_gvit.yml):
> PRETRAIN_PATH: XXX

## Data preprocess(SICM in SIM)
Please run [CSG_interaction.py](./fastreid/data/tramsforms/CSG_interaction.py) to get social interaction information of our SIM for each group image under the folder.

Then you should chage lines 99 to 114 in [common.py](./fastreid/data/common.py) to select which CSG you want to use(CSG in defalut).

## Training
Single or multiple GPU training is supported. Please refer to [scripts](./scripts/) folder.

Run training with `python tools/train_net.py --config-file ./configs/CSG/bagtricks_gvit.yml MODEL.DEVICE "cuda:0"` in Terminal.
## Test
Test was after the training every 5 epoch.

## Acknowledgement
Codebase from [fast-reid](https://github.com/JDAI-CV/fast-reid). So please refer to that repository for more usage.

## To Reproduce
We use '{dataset_name} = CSG' as example above. 

Please use the correct {dataset_name} you want to reproduce, change all {dataset_name}, such as [.yml](./configs/CSG/bagtricks_gvit.yml), 
[data preprocess](./fastreid/data/tramsforms/CSG_interaction.py), [dataset load](./fastreid/data/common.py), dataset_name you input in ternimal to run, and change other necessary file.

