# DeepACSON
Automated Segmentation of White Matter in 3D Electron Microscopy

DeepACSON was developed by Ali Abdollahzadeh at the University of Eastern Finland to trace the entirety of myelinated axons in low-resolution, large field-of-view 3D electron microscopy images of white matter.

A. Abdollahzadeh, I. Belevich, E. Jokitalo, A. Sierra, J. Tohka, DeepACSON: Automated Segmentation of White Matter in 3D Electron Microscopy, bioRxivdoi:https://doi.org/10.1101/828541.

A. Abdollahzadeh, A. Sierra, J. Tohka, Cylindrical shape decomposition for 3D segmentation of tubular objects, arXiv:1911.00571v2 [cs.CV] (2019). 
URL http://arxiv.org/abs/1911.00571.

# BM4D denoising
We used BM4D filter to denoise 3D-electron microscopy images: you can download BM4D v3.2 from https://www.cs.tut.fi/~foi/GCF-BM3D/

# Training
Install Elktronn as instructed in https://github.com/ELEKTRONN. The network can be trained with the train.py script, which expects .h5 files as training materials.

# Inference
Use the inference.py script for semantic segmentation of ultrastructures.

# Instance segmentation
The cylindrical shape decomposition algorithm is currently supported for Python 2 and requires NumPy, SciPy, Scikit-image, and scikit-fmm.

To skeletonize a 3D binary object, we implemented the method from Hassouna & Farag (CVPR 2005); the method detects ridges in the distance field of the object surface.
If you use skeleton3D in your research, please cite:

M. Hassouna and A. Farag, Robust centerline extraction framework using level sets, DOI 10.1109/CVPR.2005.306

A. Abdollahzadeh, A. Sierra, J. Tohka, Cylindrical shape decomposition for 3D segmentation of tubular objects, arXiv:1911.00571v2 [cs.CV] (2019). 
URL http://arxiv.org/abs/1911.00571.

```
from skeleton3D import skeleton
skel = skeleton(BW)
```



