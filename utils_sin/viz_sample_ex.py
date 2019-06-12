# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2019-04-08 21:07:10
# @Last Modified by:   twankim
# @Last Modified time: 2019-05-20 19:29:05
import os
import sys
import numpy as np
import argparse

import cv2
from utils_sin.sin_utils import SINFields, genSINtoInputs, genMask2D, get_stride_sub
from wavedata.tools.core import calib_utils
import matplotlib.pyplot as plt

D_MAX = 50.0
D_MIN = 0.7

def plotSINImage(data_dir, out_dir, img_idx, sin_type, sin_level, on_img, sin_input_name,
                 mask_2d=None):
    fname = '{:06d}'.format(img_idx)
    path_img = os.path.join(data_dir,'image_2')
    path_velo = os.path.join(data_dir,'velodyne')
    path_calib = os.path.join(data_dir,'calib')

    # Load image
    cv_bgr_image = cv2.imread(os.path.join(path_img,fname+'.png'))
    rgb_image = cv_bgr_image[..., :: -1]
    image_shape = rgb_image.shape[0:2]
    
    # Load point cloud
    frame_calib = calib_utils.read_calibration(path_calib, img_idx)
    x,y,z,_ = calib_utils.read_lidar(velo_dir=path_velo, img_idx=img_idx)
    if sin_type == 'lowres':
        starts = np.where(np.diff(np.sign(y)) > 0)[0] + 1
        true_starts = np.append(np.diff(starts) > 2, [True])
        starts = starts[true_starts]
        n_lasers = starts.shape[0] + 1
        starts = [0] + starts.tolist() + [len(x)]

        include = np.zeros(len(x),dtype=bool)
        stride_sub = get_stride_sub(sin_level)
        lasers_num = range(0,SINFields.LASERS_NUM,stride_sub)
        for laser in lasers_num:
            if laser < n_lasers:
                include[starts[laser]:starts[laser+1]] = True

        pts_lowres = np.vstack((x[include], y[include], z[include])).T # N x 3
        pts_lowres = calib_utils.lidar_to_cam_frame(pts_lowres, frame_calib)
        pts_lowres = pts_lowres[pts_lowres[:,2]>0] # Only keep points in front of camera (positive z)
        # Project to image frame
        point_in_im_lowres = calib_utils.project_to_image(pts_lowres.T, p=frame_calib.p2).T # N x 3
        im_size = [image_shape[1], image_shape[0]]
        # Filter based on the given image size
        image_filter_lowres = (point_in_im_lowres[:, 0] > 0) & \
                              (point_in_im_lowres[:, 0] < im_size[0]) & \
                              (point_in_im_lowres[:, 1] > 0) & \
                              (point_in_im_lowres[:, 1] < im_size[1])
        point_cloud_lowres = pts_lowres[image_filter_lowres,:].T
    else:
        include = np.ones(len(x),dtype=bool)

    pts = np.vstack((x, y, z)).T # N x 3
    pts = calib_utils.lidar_to_cam_frame(pts, frame_calib)
    pts = pts[pts[:,2]>0] # Only keep points in front of camera (positive z)
    # Project to image frame
    point_in_im = calib_utils.project_to_image(pts.T, p=frame_calib.p2).T # N x 3
    im_size = [image_shape[1], image_shape[0]]
    # Filter based on the given image size
    image_filter = (point_in_im[:, 0] > 0) & \
                   (point_in_im[:, 0] < im_size[0]) & \
                   (point_in_im[:, 1] > 0) & \
                   (point_in_im[:, 1] < im_size[1])
    point_cloud = pts[image_filter,:].T
    point_in_im = point_in_im[image_filter,:]

    image_input_sin,point_cloud_sin = genSINtoInputs(rgb_image,point_cloud,
                                                     sin_type = sin_type,
                                                     sin_level = sin_level,
                                                     sin_input_name = sin_input_name,
                                                     mask_2d = mask_2d,
                                                     frame_calib_p2 = frame_calib.p2)
    if sin_type == 'lowres':
        point_cloud_sin = point_cloud_lowres
    
    fname_out = fname+'_{}_sin_{}_{}'.format(sin_input_name,sin_type,sin_level)
    if sin_input_name == 'image':
        cv2.imwrite(os.path.join(out_dir,fname+'_image_org.png'),cv_bgr_image)
        cv2.imwrite(os.path.join(out_dir,fname_out+'.png'),
                    image_input_sin[...,::-1])
    elif sin_input_name == 'lidar':
        # Clip distance with min/max values (for visualization)
        pointsDist = point_cloud[2,:]
        # for i_pt,pdist in enumerate(pointsDist):
        #     pointsDist[i_pt] = D_MAX if pdist>D_MAX \
        #                        else pdist if pdist>D_MIN \
        #                        else D_MIN
        image_w_pts = points_on_img(point_in_im,
                                    pointsDist,
                                    rgb_image,
                                    on_img=on_img)

        point_in_im2 = calib_utils.project_to_image(point_cloud_sin, p=frame_calib.p2).T
        # Clip distance with min/max values (for visualization)
        pointsDist2 = point_cloud_sin[2,:]
        # for i_pt,pdist in enumerate(pointsDist2):
        #     pointsDist2[i_pt] =  D_MAX if pdist>D_MAX \
        #                          else pdist if pdist>D_MIN \
        #                          else D_MIN

        image_filter2 = (point_in_im2[:, 0] > 0) & \
                        (point_in_im2[:, 0] < im_size[0]) & \
                        (point_in_im2[:, 1] > 0) & \
                        (point_in_im2[:, 1] < im_size[1])
        point_in_im2 = point_in_im2[image_filter2,:]
        pointsDist2 = pointsDist2[image_filter2]
        image_w_pts2 = points_on_img(point_in_im2,
                                     pointsDist2,
                                     rgb_image,
                                     on_img=on_img)

        cv2.imwrite(os.path.join(out_dir,fname+'_lidar_org.png'),image_w_pts[...,::-1])
        if on_img:
            cv2.imwrite(os.path.join(out_dir,fname_out+'.png'),
                        image_w_pts2[...,::-1])
        else:
            cv2.imwrite(os.path.join(out_dir,fname_out+'_black.png'),
                        image_w_pts2[...,::-1])
    else:
        raise ValueError("Invalid sin_input_name {}".format(sin_input_name))


def points_on_img(points2D,pointsDist,image,mode='standard',on_img=True):
    _CMAP = plt.get_cmap('jet')
    if on_img:
        image_wp = np.copy(image)
    else:
        image_wp = np.zeros_like(image)
    points2D = np.floor(points2D).astype('int')
    for i,point in enumerate(points2D):
        pre_pixel = dist_to_pixel(pointsDist[i],mode=mode)
        image_wp[point[1],point[0],:] = (255*np.array(_CMAP(pre_pixel/255.0)[:3]))\
                                        .astype(np.uint8)

    return image_wp

def dist_to_pixel(val_dist, mode,
                  d_max=D_MAX, d_min=D_MIN):
    """ Returns pixel value from distance measurment
    Args:
        val_dist: distance value (m)
        mode: 'inverse' vs 'standard'
        d_max: maximum distance to consider
        d_min: minimum distance to consider
    Returns:
        pixel value in 'uint8' format
    """
    val_dist = d_max if val_dist>d_max else val_dist if val_dist>d_min else d_min
    if mode == 'standard':
        return np.round(minmax_scale(val_dist,
                                     d_max,d_min,
                                     1,255)).astype('uint8')
    elif mode == 'inverse':
        return np.round(minmax_scale(1.0/np.sqrt(val_dist+5),
                                     1.0/np.sqrt(d_max+5),1.0/np.sqrt(d_min+5),
                                     1,255)).astype('uint8')
    else:
        # Default is standard
        return np.round(minmax_scale(val_dist,
                                     d_max,d_min,
                                     1,255)).astype('uint8')

def minmax_scale(x,i_min,i_max,o_min,o_max):
    # MinMax scaling of x
    # i_min<= x <= i_max to o_min<= x_new <= o_max
    return (x-i_min)/float(i_max-i_min)*(o_max-o_min)+o_min


def main(args):
    if args.sin_type == 'rect':
        mask_2d = genMask2D()
    else:
        mask_2d = None
    for sin_input_name in SINFields.SIN_INPUT_NAMES:
        print('..Generating an image with SIN added to {} (type: {}, level: {:.3f})'.format(
            sin_input_name, args.sin_type,args.sin_level))
        plotSINImage(args.data_dir,
                     args.out_dir,
                     args.img_idx,
                     args.sin_type,
                     args.sin_level,
                     args.on_img,
                     sin_input_name,
                     mask_2d = mask_2d)

def parse_args():
    default_data_dir = '/data/kitti_avod/object/training'

    def str2bool(v):
        return v.lower() in ('true', '1')

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        type=str,
                        dest='data_dir',
                        default=default_data_dir,
                        help='output dir to save checkpoints')

    parser.add_argument('--out_dir',
                        type=str,
                        dest='out_dir',
                        default='./etc',
                        help='output dir to save generated samples')

    parser.add_argument('--img_idx',
                        type=int,
                        dest='img_idx',
                        required=True,
                        help='index of file to load')

    parser.add_argument('--sin_type',
                        type=str,
                        dest='sin_type',
                        default='rand',
                        help='Type of the SIN (rand)')

    parser.add_argument('--sin_level',
                        type=float,
                        dest='sin_level',
                        default=5,
                        help='Level of an additive single input noise (1~10)')

    parser.add_argument('--on_img',
                        type=str2bool,
                        dest='on_img',
                        default=False,
                        help='Plot lidar data on image')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print("Called with args:")
    print(args)
    sys.exit(main(args))