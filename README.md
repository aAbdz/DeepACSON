# DeepACSON
Automated Segmentation of White Matter in 3D Electron Microscopy

DeepACSON was developed by Ali Abdollahzadeh at the University of Eastern Finland to trace the entirety of myelinated axons in low-resolution, large field-of-view 3D electron microscopy images of white matter.

DeepACSON is described in

A. Abdollahzadeh, I. Belevich, E. Jokitalo, A. Sierra, J. Tohka, DeepACSON: Automated Segmentation of White Matter in 3D Electron Microscopy, bioRxivdoi:https://doi.org/10.1101/828541.

# BM4D denoising
We used BM4D filter to denoise 3D-electron microscopy images: you can download BM4D v3.2 from https://www.cs.tut.fi/~foi/GCF-BM3D/

# Training
Install Elktronn as instructed in https://github.com/ELEKTRONN. The network can be trained with the train.py script, which expects .h5 files as training materials.

# Inference
Assign the address to .mdl file. Use the inference.py script for semantic segmentation of ultrastructures.


