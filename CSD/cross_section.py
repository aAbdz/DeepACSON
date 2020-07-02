# -*- coding: utf-8 -*-
"""
Created on Wed Aug 01 11:49:42 2018

@author: aliabd
"""

import numpy as np
import plane_rotation as pr
from scipy.interpolate import RegularGridInterpolator as rgi
from unit_tangent_vector import unit_tangent_vector
from skimage.measure import label, regionprops

def cross_section(obj,skel,g_radius,g_res):
    
    cs_quant={'Point':[],'Area':[],'MajorAxis':[],'MinorAxis':[],'Eccentricity':[],'EquivDiameter':[]}
    
    sz=obj.shape
    
    x,y=np.mgrid[-g_radius:g_radius:g_res,-g_radius:g_radius:g_res]
    z=np.zeros_like(x)
    xyz=np.array([np.ravel(x),np.ravel(y),np.ravel(z)]).T
    
    tangent_vecs=unit_tangent_vector(skel)      
    interpolating_func=rgi((range(sz[0]),range(sz[1]),range(sz[2])), 
                               obj,bounds_error=False,fill_value=0)
    
    curve_length=skel.shape[0]
    for i in range(curve_length):
        point=skel[i]        
        utv=tangent_vecs[i]
        
        if np.array_equal(utv, np.array([0, 0, 0])):
            continue
        
        rot_axis=pr.unit_normal_vector(utv,np.array([0,0,1]))
        theta=pr.angle(utv,np.array([0,0,1]))
    
        rot_mat=pr.rotation_matrix_3D(rot_axis,theta)
        rotated_plane=np.squeeze(pr.rotate_vector(xyz,rot_mat))
            
        cross_section_plane=rotated_plane+point        
        
        cross_section=interpolating_func(cross_section_plane)
        bw_cross_section=cross_section>=0.5
        bw_cross_section=np.reshape(bw_cross_section,x.shape)

        label_cross_section,nn=label(bw_cross_section,neighbors=4,return_num=True)
        
        if nn == 0:
            continue
        if nn>1:
            main_lbl = label_cross_section[tuple(np.array(x.shape)/2)]
            bw_cross_section = label_cross_section==main_lbl
        
        props=regionprops(label_cross_section)
        cs_quant['Point'].append(tuple(point))
        cs_quant['Area'].append((props[0].area)*(g_radius**2))
        cs_quant['MajorAxis'].append((props[0].major_axis_length)*g_radius)
        cs_quant['MinorAxis'].append((props[0].minor_axis_length)*g_radius)
        cs_quant['Eccentricity'].append(props[0].eccentricity)
        cs_quant['EquivDiameter'].append((props[0].equivalent_diameter)*g_radius)
    return cs_quant


    