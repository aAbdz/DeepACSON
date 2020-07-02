# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 18:47:52 2018

@author: aliabd
"""

import numpy as np
import h5py
import pickle
from skeleton3D import skeleton
from multiprocessing import Pool, TimeoutError, current_process
import os
from sys import argv
import traceback
from time import time

read_dir = '/home/aliabd/Project/LMComp_Proj/DeepAcson/FCN/tests/'
save_dir = '/home/aliabd/Project/LMComp_Proj/DeepAcson/FCN/tests/'
read_fname = 'mat_objs5.mat'

read_file = os.path.join(read_dir,read_fname)

save_log_file = os.path.join(save_dir, 'log_file_skeleton0.txt')
    
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def exe_skeleton(ax):
    
    print_set = range(0, 40000, 20)
    if ax in print_set:
        print 'I am' + str(current_process()) + 'processing obj #' + str(ax)

    try:
    
        with h5py.File(read_file, 'r') as f:

            cell_content = f[f.keys()[1]]

            axon_inx = cell_content[0]
            axon_bb = cell_content[1]

            bb_ref = axon_bb[ax]
            bb = np.squeeze(np.array((f[bb_ref])).astype(int))

            inx_ref = axon_inx[ax]
            obj_inx = np.array((f[inx_ref])).T - 1 

        obj_sz = (bb[4], bb[3], bb[5])
        
        cropAx = np.zeros(obj_sz, dtype=np.int8)

        for j in obj_inx:
            cropAx[tuple(j)]=1

        skel = skeleton(cropAx)



    except Exception:

        with open(save_log_file, 'a') as f:
            f.write('\n error in obj %d \r' % ax)

        skel = []

    return ax, skel


            
if __name__ == '__main__':
    
    start_time = time()

    with h5py.File(read_file, 'r') as f:
        cell_content = f[f.keys()[1]] 
        l = len(cell_content[1])

    print l

    seq = range(l)

    num_cores = int(argv[1])

    timed_out_results = 0
 
    chunks = chunkIt(seq, 1)

    f = open(save_log_file, 'w')
    f.close()

    counter = 0

    ## len(chunks)

    for ii in range(len(chunks)):

	chunk = chunks[ii]

        p = Pool(num_cores)

        results = []
        for j in chunk:
               
            results.append(p.apply_async(exe_skeleton, (j,)))

        final_results = []
        
        for result in results:

            try:

                res = result.get(1600)
                final_results.append(res)

		with open(save_log_file, 'a') as f:
 		    f.write('\n ontime obj %d \r' % res[0])

                #print 'on time_' + str(res[0])
                
            except TimeoutError:

                timed_out_results += 1
                with open(save_log_file, 'a') as f:
                    f.write('\n timeoutError %d \r' % timed_out_results)

                pass
        
        print 'write2disk'

        save_file = os.path.join(save_dir, 'skeletons_')
        save_file = save_file + str(counter)
        
        with open(save_file,'wb') as file_handle:
            pickle.dump(final_results, file_handle)

        del final_results

        counter += 1

	print 'CHUNK IS DONE'

    end_time = time()
    duration = (end_time - start_time) ##/ 60

    with open(save_log_file, 'a') as f:
        f.write('\n Job completed in %d sec \r' % duration)
    
    print 'JOB IS DONE'
