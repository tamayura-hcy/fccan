# OCT_DDA
This is the source code for Paper "Cross-domain retinopathy classification with optical coherence tomography images via a novel deep domain adaptation method"

## Pre-request
1. Download the pretrained vgg16 model from https://download.pytorch.org/models/vgg16_bn-6c64b313.pth

## Split the dataset by person
1. Run the python code `CELL_data_split.py` to split CELL dataset
2. Run the python code `BOE_data_split_by_person.py` to split BOE dataset
3. Run the python code `TMI_data_split_by_person.py` to split TMI dataset


## Benchmark methods
1. Run the python code `ADDA_vgg16_train.py` for ADDA
2. Run the python code `CAT_vgg16_train.py` for CAT
3. Run the python code `DDC_vgg16__train.py` for DDC

## Proposed DDA method
1. Run the python code `ADDA_EM_vgg16_train.py` for proposed method 
