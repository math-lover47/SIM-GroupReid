# Setup Dataset


the default path is `datasets/` relative to your current working directory.


## Expected dataset structure for CSG

1. Download dataset to `datasets/` from "Learning Multi-Attention Context Graph for Group-Based Re-Identification",TPAMI,2020.
2. Extract dataset. The dataset structure would like:

```bash
datasets/
    CUHK-SYSU/
        images/
        GReID_label/
	    cuhk_train.pkl/
	    cuhk_gallery.pkl/
	    cuhk_test.pkl/
	    
```

## Expected dataset structure for RoadGroup

1. Download datasets to `datasets/` from "Group Re-Identification with Multigrained Matching and Integration",TCYB,2021.
2. Extract dataset. The dataset structure would like:

```bash
datasets/
    RoadGroup/
        Road_Group/
        Road_Group_Annotations/
	    person_bounding_box.json/
	    group_id.json/
	    person_correspondance.json/
```

## Expected dataset structure for DukeGroup

1. Download datasets to `datasets/` "Group Re-Identification with Multigrained Matching and Integration",TCYB,2021.
2. Extract dataset. The dataset structure would like:

```bash
datasets/
        
        DukeMTMC_Group/
        DukeMTMC_Group_Annotations/
	    person_bounding_box.json/
	    group_id.json/
	    person_correspondance.json/
```
