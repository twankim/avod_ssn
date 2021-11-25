# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2019-04-08 21:07:10
# @Last Modified by:   twankim
# @Last Modified time: 2019-05-20 21:39:16
import numpy as np
import tensorflow as tf
from wavedata.tools.core import calib_utils


import tf_slim as slim

_USE_GLOBAL_STEP = 0

class SINFields:
    VALID_SIN_TYPES = ['rand','rect','vert','lowres']
    SIN_INPUT_NAMES = ['image','lidar']
    VALID_MAX_MAGTD = {'image': 255,
                       'lidar': 0.2}
    SIN_LEVEL_MAX = 10.0
    SIN_LEVEL_MIN = 0.0
    SIN_METRIC_NAME = 'minAP'
    
    CROP_H_MAX = 0.4
    CROP_H_MIN = 0.2
    # Exclude top region (not detected by LIDAR and usually contains no object)
    CROP_H_OFFSET = 0.25

    CROP_W_MAX = 0.4
    CROP_W_MIN = 0.2

    VERT_W_MAX=0.1
    VERT_W_MIN=0.0
    
    LASERS_NUM = 64
    LOWRES_STRIDE_MIN = 1
    LOWRES_STRIDE_MAX = 8

def genSINtoInputs(image_input,point_cloud,
                   sin_type='rand',
                   sin_level=1,
                   sin_input_name='image',
                   mask_2d=None,
                   frame_calib_p2=None):
    """
        Add Single Input Noise to a given input
    """
    assert sin_type in SINFields.VALID_SIN_TYPES,\
        "sin_type must be one of [{}]".format(','.join(SINFields.VALID_SIN_TYPES))
    assert sin_input_name in SINFields.SIN_INPUT_NAMES,\
        "sin_input_name must be one of [{}]".format(','.join(SINFields.SIN_INPUT_NAMES))

    if sin_input_name == 'image':
        sin_image_input = np.copy(image_input).astype(np.float)
        sin_point_cloud = point_cloud
        sin_image_input = genSIN(sin_image_input,
                                 sin_point_cloud,
                                 sin_type, sin_level,
                                 sin_input_name = sin_input_name,
                                 mask_2d = mask_2d,
                                 frame_calib_p2=frame_calib_p2)
        # Fit to range and type
        np.clip(sin_image_input, 0.0, 255.0, out=sin_image_input)
        sin_image_input = sin_image_input.astype(np.uint8)
    elif sin_input_name == 'lidar':
        sin_image_input = image_input
        sin_point_cloud = np.copy(point_cloud).astype(np.float)
        sin_point_cloud = genSIN(sin_image_input,
                                 sin_point_cloud,
                                 sin_type,
                                 sin_level,
                                 sin_input_name = sin_input_name,
                                 mask_2d = mask_2d,
                                 frame_calib_p2=frame_calib_p2)
        # Fit to range and type
        # np.clip(sin_point_cloud, -100.0, 100.0, out=sin_point_cloud)
        sin_point_cloud = sin_point_cloud.astype(np.single)
    else:
        sin_image_input = image_input
        sin_point_cloud = point_cloud
    
    return sin_image_input, sin_point_cloud


def genSINtoAllInputs(image_input,point_cloud,
                      sin_type='rand',
                      sin_level=1,
                      mask_2d=None,
                      frame_calib_p2=None):
    """
        Add Single Input Noise to all inputs
    """
    assert sin_type in SINFields.VALID_SIN_TYPES,\
        "sin_type must be one of [{}]".format(','.join(SINFields.VALID_SIN_TYPES))

    # Image
    sin_image_input = np.copy(image_input).astype(np.float)
    sin_image_input = genSIN(sin_image_input,
                             point_cloud,
                             sin_type, sin_level,
                             sin_input_name = 'image',
                             mask_2d = mask_2d,
                             frame_calib_p2=frame_calib_p2)
    # Fit to range and type
    np.clip(sin_image_input, 0.0, 255.0, out=sin_image_input)
    sin_image_input = sin_image_input.astype(np.uint8)

    # Point Cloud
    sin_point_cloud = np.copy(point_cloud).astype(np.float)
    sin_point_cloud = genSIN(image_input,
                             sin_point_cloud,
                             sin_type, sin_level,
                             sin_input_name = 'lidar',
                             mask_2d = mask_2d,
                             frame_calib_p2=frame_calib_p2)
    # Fit to range and type
    # np.clip(sin_point_cloud, -100.0, 100.0, out=sin_point_cloud)
    sin_point_cloud = sin_point_cloud.astype(np.single)
    
    return sin_image_input, sin_point_cloud


def genSIN(image_input,point_cloud,
           sin_type='rand', sin_level=1, sin_input_name='image', mask_2d=None,frame_calib_p2=None):
    if sin_input_name == 'image':
        input_shape = np.shape(image_input)
    elif sin_input_name == 'lidar':
        input_shape = np.shape(point_cloud)
    else:
        input_shape = np.shape(image_input)

    max_magnitude = SINFields.VALID_MAX_MAGTD[sin_input_name]
    sin_factor = (np.clip(sin_level,SINFields.SIN_LEVEL_MIN,SINFields.SIN_LEVEL_MAX) \
                  / SINFields.SIN_LEVEL_MAX) *1.5*max_magnitude
    # Set to 1.5-sigma region
    if sin_type == 'rand':
        sin_input = sin_factor*np.random.randn(*input_shape)
        if sin_input_name == 'image':
            sin_input += image_input
        elif sin_input_name == 'lidar':
            sin_input += point_cloud
        else:
            sin_input += image_input
    elif sin_type == 'rect':
        sin_input = genRectOcc(image_input,point_cloud,sin_input_name,mask_2d,frame_calib_p2)
    elif sin_type == 'vert':
        sin_input = genVertOcc(image_input,point_cloud,sin_input_name,frame_calib_p2,sin_level)
    elif sin_type == 'lowres':
        stride_sub = get_stride_sub(sin_level)
        sin_input = genLowRes(image_input,point_cloud,sin_input_name,stride_sub)
    else:
        # Default is to use random noise
        sin_input += sin_factor*np.random.randn(*input_shape)
    return sin_input

def genRectOcc(image_input,point_cloud,sin_input_name,mask_2d=None,frame_calib_p2=None):
    """
        mask_2d: 2x2 array. 
                 mask_2d[0] -> r_len, r_corner (0~1 scale) y-axis (height)
                 mask_2d[1] -> r_len, r_corner (0~1 scale) x-axis (width)

    """
    image_shape = np.shape(image_input)

    if mask_2d is None:
        mask_2d = genMask2D()

    # Generate rectangular occlusion in 2D image plane
    im_y_start = np.round(mask_2d[0,1]*image_shape[0]).astype(np.int)
    im_y_end = min(image_shape[0],
                   im_y_start + np.round(mask_2d[0,0]*image_shape[0]).astype(np.int))
    im_x_start = np.round(mask_2d[1,1]*image_shape[1]).astype(np.int)
    im_x_end = min(image_shape[1],
                   im_x_start + np.round(mask_2d[1,0]*image_shape[1]).astype(np.int))

    if sin_input_name == 'image':
        if len(image_shape) == 2:
            image_input[im_y_start:im_y_end,im_x_start:im_x_end] = 0
        elif len(image_shape) == 3:
            image_input[im_y_start:im_y_end,im_x_start:im_x_end,:] = 0
        else:
            raise ValueError(
                'Image shape is wrong. Must be either 2d (b/w) or 3d.')
        return image_input
    elif sin_input_name == 'lidar':
        # Occlude lidar point cloudes to match 2d mask in image plane
        point_in_im = calib_utils.project_to_image(point_cloud, p=frame_calib_p2).T

        # Filter based on the given image size
        im_occ_filter = (point_in_im[:, 1] >= im_y_start) & \
                        (point_in_im[:, 1] < im_y_end) & \
                        (point_in_im[:, 0] > im_x_start) & \
                        (point_in_im[:, 0] < im_x_end)
        return point_cloud[:,np.invert(im_occ_filter)]
    else:
        raise ValueError(
            'Currently supports {}'.format(','.join(SIN_INPUT_NAMES)))

def genMask2D():
    mask_2d = np.zeros((2,2))

    # Y-axis
    mask_2d[0,0] = np.random.uniform(low=SINFields.CROP_H_MIN,high=SINFields.CROP_H_MAX)
    if SINFields.CROP_H_OFFSET > 1-mask_2d[0,0]:
        high_width = SINFields.CROP_H_OFFSET
    else:
        high_width = 1-mask_2d[0,0]
    mask_2d[0,1] = np.random.uniform(low=SINFields.CROP_H_OFFSET, high=high_width)

    # X-axis
    mask_2d[1,0] = np.random.uniform(low=SINFields.CROP_W_MIN,high=SINFields.CROP_W_MAX)
    mask_2d[1,1] = np.random.uniform(low=0, high=1-mask_2d[1,0])

    return mask_2d

def genVertOcc(image_input,point_cloud,sin_input_name,frame_calib_p2=None,sin_level=1):
    """
        mask_2d: 2x2 array. 
                 mask_2d[0] -> r_len, r_corner (0~1 scale) y-axis (height)
                 mask_2d[1] -> r_len, r_corner (0~1 scale) x-axis (width)

    """
    image_shape = np.shape(image_input)
    im_width = image_shape[1]
    width_vert = int(im_width*minmax_scale(sin_level,
                                           SINFields.SIN_LEVEL_MIN,
                                           SINFields.SIN_LEVEL_MAX,
                                           SINFields.VERT_W_MIN,
                                           SINFields.VERT_W_MAX))

    idx_x_starts = np.arange(0,im_width,2*width_vert)
    idx_x_ends = np.arange(width_vert,im_width,2*width_vert)
    m_verts = len(idx_x_ends)
    if len(idx_x_starts)>len(idx_x_ends):
        idx_x_ends = np.append(idx_x_ends,im_width)
        m_verts += 1

    idx_zeros = np.zeros(im_width,dtype=np.bool)
    for j in range(m_verts):
        idx_zeros[idx_x_starts[j]:idx_x_ends[j]] = True

    if sin_input_name == 'image':
        if len(image_shape) == 2:
            image_input[:,idx_zeros] = 0
        elif len(image_shape) == 3:
            image_input[:,idx_zeros,:] = 0
        else:
            raise ValueError(
                'Image shape is wrong. Must be either 2d (b/w) or 3d.')
        return image_input
    elif sin_input_name == 'lidar':
        # Occlude lidar point cloudes to match 2d mask in image plane
        point_in_im = calib_utils.project_to_image(point_cloud, p=frame_calib_p2).T

        # Filter based on the given image size
        im_occ_filter = idx_zeros[point_in_im[:,0].astype(int)]
        return point_cloud[:,np.invert(im_occ_filter)]
    else:
        raise ValueError(
            'Currently supports {}'.format(','.join(SIN_INPUT_NAMES)))

def genLowRes(image_input,point_cloud,sin_input_name,stride_sub=1):
    """
        mask_2d: 2x2 array. 
                 mask_2d[0] -> r_len, r_corner (0~1 scale) y-axis (height)
                 mask_2d[1] -> r_len, r_corner (0~1 scale) x-axis (width)

    """
    if sin_input_name == 'lidar':
        # subsampling is done separately in the kitti_dataset.py
        return point_cloud
    elif sin_input_name == 'image':
        if stride_sub == 1:
            return image_input
        image_shape = np.shape(image_input)
        im_height = image_shape[0]
        idx_y_starts = np.arange(0,im_height,stride_sub)
        idx_y_ends = np.arange(1,im_height,stride_sub)
        m_verts = len(idx_y_ends)
        if len(idx_y_starts)>len(idx_y_ends):
            idx_y_ends = np.append(idx_y_ends,im_height)
            m_verts += 1

        idx_nonzeros = np.zeros(im_height,dtype=np.bool)
        for j in range(m_verts):
            idx_nonzeros[idx_y_starts[j]:idx_y_ends[j]] = True

        if len(image_shape) == 2:
            image_input[np.invert(idx_nonzeros),:] = 0
        elif len(image_shape) == 3:
            image_input[np.invert(idx_nonzeros),:,:] = 0
        else:
            raise ValueError(
                'Image shape is wrong. Must be either 2d (b/w) or 3d.')
        return image_input
    else:
        raise ValueError(
            'Currently supports {}'.format(','.join(SIN_INPUT_NAMES)))

def minmax_scale(x,i_min,i_max,o_min,o_max):
    # MinMax scaling of x
    # i_min<= x <= i_max to o_min<= x_new <= o_max
    return (x-i_min)/float(i_max-i_min)*(o_max-o_min)+o_min

def get_stride_sub(sin_level):
    stride_sub = np.clip(np.floor(sin_level).astype(int),
                         SINFields.LOWRES_STRIDE_MIN,
                         SINFields.LOWRES_STRIDE_MAX)
    return stride_sub

def get_point_cloud_sub(img_idx,calib_dir,velo_dir,image_shape=None,stride_sub=1):
    im_size = [image_shape[1], image_shape[0]]

    point_cloud = get_lidar_point_cloud_sub(
        img_idx, calib_dir, velo_dir,im_size=im_size,stride_sub=stride_sub)

    return point_cloud

def get_lidar_point_cloud_sub(img_idx, calib_dir, velo_dir,
                              im_size=None, min_intensity=None,stride_sub=1):
    """ Calculates the lidar point cloud, and optionally returns only the
    points that are projected to the image.

    :param img_idx: image index
    :param calib_dir: directory with calibration files
    :param velo_dir: directory with velodyne files
    :param im_size: (optional) 2 x 1 list containing the size of the image
                      to filter the point cloud [w, h]
    :param min_intensity: (optional) minimum intensity required to keep a point

    :return: (3, N) point_cloud in the form [[x,...][y,...][z,...]]
    """

    # Read calibration info
    frame_calib = calib_utils.read_calibration(calib_dir, img_idx)
    x, y, z, i = calib_utils.read_lidar(velo_dir=velo_dir, img_idx=img_idx)

    starts = np.where(np.diff(np.sign(y)) > 0)[0] + 1
    true_starts = np.append(np.diff(starts) > 2, [True])
    starts = starts[true_starts]
    n_lasers = starts.shape[0] + 1
    starts = [0] + starts.tolist() + [len(x)]

    include = np.zeros(len(x),dtype=bool)
    lasers_num = range(0,SINFields.LASERS_NUM,stride_sub)
    for laser in lasers_num:
        if laser < n_lasers:
            include[starts[laser]:starts[laser+1]] = True

    i = i[include]

    # Calculate the point cloud
    pts = np.vstack((x[include], y[include], z[include])).T
    pts = calib_utils.lidar_to_cam_frame(pts, frame_calib)

    # The given image is assumed to be a 2D image
    if not im_size:
        point_cloud = pts.T
        return point_cloud

    else:
        # Only keep points in front of camera (positive z)
        pts = pts[pts[:, 2] > 0]
        point_cloud = pts.T

        # Project to image frame
        point_in_im = calib_utils.project_to_image(point_cloud, p=frame_calib.p2).T

        # Filter based on the given image size
        image_filter = (point_in_im[:, 0] > 0) & \
                       (point_in_im[:, 0] < im_size[0]) & \
                       (point_in_im[:, 1] > 0) & \
                       (point_in_im[:, 1] < im_size[1])

    if not min_intensity:
        return pts[image_filter].T

    else:
        intensity_filter = i > min_intensity
        point_filter = np.logical_and(image_filter, intensity_filter)
        return pts[point_filter].T