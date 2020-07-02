# -*- coding: utf-8 -*-

import numpy as np
import h5py
import pickle
from shape_decomposition_v04 import object_analysis
from multiprocessing import Pool, TimeoutError, current_process
import os
from sys import argv
import traceback
from time import time


read_dir_1 = ''
read_dir_2 = ''

read_fname_axon = 'mat_obj.mat'
read_fname_skeleton = 'all_skeletons'

read_file_axon = os.path.join(read_dir_1, read_fname_axon)
read_file_skeleton = os.path.join(read_dir_2, read_fname_skeleton)

save_dir = '/home/aliabd/Project/LMComp_Proj/DeepAcson/FCN/tests/'

save_log_file = os.path.join(save_dir, 'log_file_38-end.txt')
    
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out



def exe_shape_decomposition(ax, skel):

    print_set = range(0, 50000, 50)
    if ax in print_set:
        print 'I am' + str(current_process()) + 'processing obj #' + str(ax)
        
    with h5py.File(read_file_axon, 'r') as f:

        cell_content = f[f.keys()[1]]

        axon_inx = cell_content[0]
        axon_bb = cell_content[1]

        bb_ref = axon_bb[ax]
        bb = np.squeeze(np.array((f[bb_ref])).astype(int))

        inx_ref = axon_inx[ax]
        obj_inx = np.array((f[inx_ref])).T - 1 
    
    obj_sz = (bb[4], bb[3], bb[5])
    rec_inx = np.array((bb[1], bb[0], bb[2]))-1
            
    try:

        dec_im, dec_skel, dec_quants = object_analysis(obj_inx, obj_sz, skel)
    
    except Exception:

        with open(save_log_file, 'a') as f:
            f.write('\n error in obj %d \r' % ax)

        dec_im, dec_skel, dec_quants = [obj_inx], skel, []

    return ax, dec_im, dec_skel, dec_quants, rec_inx


            
if __name__ == '__main__':

    start_time = time()

    with h5py.File(read_file_axon, 'r') as f:
        cell_content = f[f.keys()[1]] 
        LenObj = len(cell_content[1])


    with open(read_file_skeleton, 'rb') as filehandle_skel:
        List_of_skeletons = pickle.load(filehandle_skel)


    num_cores = int(argv[1])


    counter = 0

    timed_out_results = 0
 
    seq = range(LenObj) 
    chunks = chunkIt(seq, 1)

    f = open(save_log_file, 'w')
    f.close()
    #25, len(chunks)
    for ii in range(0, len(chunks)):
        
        chunk = chunks[ii]

        p = Pool(num_cores)

        results = []
        for j in chunk:
            
            skel_inf = List_of_skeletons[j]

            if len(skel_inf) != 2:
                obj_skel = []
            else:
                obj_skel = skel_inf[1]
            
            results.append(p.apply_async(exe_shape_decomposition, (j, obj_skel)))

        final_results = []
        
        for result in results:

            try:

                res = result.get(1200)
                final_results.append(res)

                with open(save_log_file, 'a') as f:
                    f.write('\n ontime obj %d \r' % res[0])


            except TimeoutError:

                timed_out_results += 1
                with open(save_log_file, 'a') as f:
                    f.write('\n timeoutError %d \r' % timed_out_results)

                pass
        
        print 'write2disk'


        save_file = os.path.join(save_dir, 'analysis_')
        save_file = save_file + str(counter) + '.npy'
        np.save(save_file, final_results)

        del final_results

        counter += 1

	print 'CHUNK IS DONE'

    end_time = time()
    duration = (end_time - start_time) / 60

    with open(save_log_file, 'a') as f:
        f.write('\n Job completed in %d min \r' % duration)

    print 'JOB IS DONE'




















