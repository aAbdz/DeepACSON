# DeepACSON
Automated Segmentation of White Matter in 3D Electron Microscopy

DeepACSON was developed by Ali Abdollahzadeh at the University of Eastern Finland to trace the entirety of myelinated axons in low-resolution, large field-of-view 3D electron microscopy images of white matter.

A. Abdollahzadeh, I. Belevich, E. Jokitalo, A. Sierra, J. Tohka, DeepACSON: Automated Segmentation of White Matter in 3D Electron Microscopy, bioRxivdoi:https://doi.org/10.1101/828541.

A. Abdollahzadeh, A. Sierra, J. Tohka, Cylindrical shape decomposition for 3D segmentation of tubular objects, arXiv:1911.00571v2 [cs.CV] (2019). 
URL http://arxiv.org/abs/1911.00571.

## BM4D denoising
We used BM4D filter to denoise 3D-electron microscopy images: you can download BM4D v3.2 from https://www.cs.tut.fi/~foi/GCF-BM3D/

## Training and inference
Install Elktronn as instructed in https://github.com/ELEKTRONN. The network can be trained with the train.py script, which expects .h5 files as training materials. Use the inference.py script for semantic segmentation of ultrastructures.

## Instance segmentation
The cylindrical shape decomposition algorithm is currently supported for Python 2 and requires NumPy, SciPy, Scikit-image, and scikit-fmm.

### Skeletonization
To skeletonize a 3D voxel-based object, we implemented the method from Hassouna & Farag (CVPR 2005); the method detects ridges in the distance field of the object surface. If you use skeleton3D in your research, please cite:

M. Hassouna and A. Farag, Robust centerline extraction framework using level sets, DOI 10.1109/CVPR.2005.306

A. Abdollahzadeh, A. Sierra, J. Tohka, Cylindrical shape decomposition for 3D segmentation of tubular objects, arXiv:1911.00571v2 [cs.CV] (2019). 
URL http://arxiv.org/abs/1911.00571.

Our implementation only requires:
- numpy>=1.0.2 
- scikit-fmm 2019.1.30

```python
import numpy as np
from skeleton3D import skeleton

fn = ./example/mAxon_mError.npy
bw = np.load(fn)
skel = skeleton(bw)
```
You can modify the code to define the shotest path either as

- euler shortest path: sub-voxel precise 
- discrete shortest path: more robust but voxel precise

You can also modify the code to exclude branches shorter than *length_threshold* value.

### Cylindrical shape decomposition
To apply CSD on a 3D voxel-based object, given its skeleton as *skel*, we have:

```python
from shape_decomposition import object_analysis
decomposed_image, decomposed_skeleton = object_analysis(BW, skel)
```
You can modify the code to define the code for *H_th* value, which lies in range [0, 1]. *H_th* is the similarity threshold between cross-sectional contours. 

DeepACSON instance segmentation, i.e., the CSD algorithm, can be run for every 3D object, i.e., an axon, independently on a CPU core. Therefore, instance segmentation of *N* axons can be run in parallel, where *N* is the number of CPU cores assigned for the segmentation task.










