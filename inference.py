# -*- coding: utf-8 -*-

import numpy as np
from elektronn2.utils import h5load
from elektronn2 import neuromancer as nm

model_path = 'ADDRESS: path to .mdl file'
model = nm.model.modelload(model_path)

raw_path = 'ADDRESS: path to test set of .h5 format'
save_path = 'ADDRESS: path to save as .npy format'
  
testIm = h5load(raw_path, 'raw')
testIm4d = testIm[None, :, :, :]

pred = model.predict_dense(testIm4d)
np.save(save_path, pred)

