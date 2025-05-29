# SUFFICIENT

3D isotropic high-resolution fetal brain MRI reconstruction from motion corrupted thick data based on physical-informed unsupervised learning. 
More information and instructions will follow soon.



## 1.Download Pre-trained Models

sufficient/monaifbs/models/checkpoint_dynUnet_DiceXent.pt https://zenodo.org/record/4282679#.X7fyttvgqL5

## 2. Fetal brain segmentation

run python sufficient/fetal_brain_seg.py

## 3. Pre-precessing

run python sufficient/preprocessing.py

## 4. Training sufficient

run python sufficient/train.py
