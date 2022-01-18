"""Visualization code for point clouds and 3D bounding boxes with mayavi.

Modified by Charles R. Qi
Date: September 2017
"""

import argparse
import os

# import mayavi.mlab as mlab
import numpy as np
import torch
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchpack.utils.config import configs

# from model_zoo import minkunet, spvcnn, spvnas_specialized
import open3d as o3d
from core.datasets.semantic_poss import SEM_COLOR
from tool_func import *

def process_point_cloud(input_point_cloud, input_labels=None, voxel_size=0.05):
    input_point_cloud[:, 3] = input_point_cloud[:, 3]
    pc_ = np.round(input_point_cloud[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)

    label_map = create_label_map_poss()
    if input_labels is not None:
        labels_ = label_map[input_labels & 0xFFFF].astype(
            np.int64)  # semantic labels
    else:
        labels_ = np.zeros(pc_.shape[0], dtype=np.int64)

    feat_ = input_point_cloud

    # if input_labels is not None:
    #     out_pc = input_point_cloud[labels_ != labels_.max(), :3]
    #     pc_ = pc_[labels_ != labels_.max()]
    #     feat_ = feat_[labels_ != labels_.max()]
    #     labels_ = labels_[labels_ != labels_.max()]
    # else:
    #     out_pc = input_point_cloud
    #     pc_ = pc_

    out_pc = input_point_cloud
    pc_ = pc_

    _, inds, inverse_map = sparse_quantize(pc_,
                                            return_index=True,
                                            return_inverse=True)
    # inds, labels, inverse_map = sparse_quantize(pc_,
    #                                             feat_,
    #                                             labels_,
    #                                             return_index=True,
    #                                             return_inverse=True)
    pc = np.zeros((inds.shape[0], 4))
    pc[:, :3] = pc_[inds]

    feat = feat_[inds]
    labels = labels_[inds]
    lidar = SparseTensor(
        torch.from_numpy(feat).float(),
        torch.from_numpy(pc).int())
    
    # labels = SparseTensor(labels, pc)
    # labels_ = SparseTensor(labels_, pc_)
    # inverse_map = SparseTensor(inverse_map, pc_)
    return {
        'pc': out_pc,
        'lidar': lidar,
        'targets': labels,
        'targets_mapped': labels_,
        'inverse_map': inverse_map
    }


# mlab.options.offscreen = True

def draw_lidar(pc,
               color=None,
               fig=None,
               bgcolor=(1, 1, 1),
               pts_scale=0.06,
               pts_mode='2dcircle',
               pts_color=None):
    if fig is None:
        fig = mlab.figure(figure=None,
                          bgcolor=bgcolor,
                          fgcolor=None,
                          engine=None,
                          size=(800, 500))
    if color is None:
        color = pc[:, 2]
    pts = mlab.points3d(pc[:, 0],
                        pc[:, 1],
                        pc[:, 2],
                        color,
                        mode=pts_mode,
                        scale_factor=pts_scale,
                        figure=fig)
    pts.glyph.scale_mode = 'scale_by_vector'
    pts.glyph.color_mode = 'color_by_scalar'  # Color by scalar
    pts.module_manager.scalar_lut_manager.lut.table = cmap
    pts.module_manager.scalar_lut_manager.lut.number_of_colors = cmap.shape[0]

    mlab.view(azimuth=180,
              elevation=70,
              focalpoint=[12.0909996, -1.04700089, -2.03249991],
              distance=62,
              figure=fig)

    return fig

def inference(pc, model, label_file_name=None):    
    
    if label_file_name and os.path.exists(label_file_name):
        label = np.fromfile(label_file_name, dtype=np.int32)
    else:
        label = None
    feed_dict = process_point_cloud(pc, label)
    inputs = feed_dict['lidar'].to(device)
    with torch.no_grad():
        outputs = model(inputs)
    predictions = outputs.argmax(1).cpu().numpy()
    predictions = predictions[feed_dict['inverse_map']]
    return feed_dict, predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--velodyne-dir', type=str, default='/hdd/dyd/SemanticPOSS/sequences/05/velodyne')
    # parser.add_argument('--velodyne-dir', type=str, default='/hdd/dyd/lidarcap/velodyne/6')
    parser.add_argument('--model', type=str, default='SemanticKITTI_val_SPVCNN@65GMACs')
    # args = parser.parse_args()
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    
    output_dir = args.velodyne_dir.replace('velodyne', 'human_semantic')
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    from core import builder
    from torchpack import distributed as dist
    model = builder.make_model().to(device)
    # if 'MinkUNet' in args.model:
    #     model = minkunet(args.model, pretrained=True)
    # elif 'SPVCNN' in args.model:
        # model = spvcnn(args.model, pretrained=True)
    # elif 'SPVNAS' in args.model:
    #     model = spvnas_specialized(args.model, pretrained=True)
    # else:
    #     raise NotImplementedError

    model = model.to(device)
    init = torch.load(os.path.join(args.run_dir, 'checkpoints', 'max-test-iou.pt'),
                      map_location='cuda:%d' % dist.local_rank() 
                      if torch.cuda.is_available() else 'cpu')['model']
    model.load_state_dict(init)
    input_point_clouds = sorted(os.listdir(args.velodyne_dir))
    model.eval()

    files_num = len(input_point_clouds)
    for i, point_cloud_name in enumerate(input_point_clouds):
        point_cloud_file = f'{args.velodyne_dir}/{point_cloud_name}'

        if point_cloud_name.endswith('.bin'):
            # semantickitti
            pc = np.fromfile(point_cloud_file,
                     dtype=np.float32).reshape(-1, 4)   # 读取点云
            label_file_name = os.path.join(args.velodyne_dir.replace(
                'velodyne', 'labels'), point_cloud_name.replace('.bin', '.label'))
            out_file_name = os.path.join(
                output_dir, point_cloud_name.replace('.bin', '.pcd'))
            
        elif point_cloud_name.endswith('.pcd'):
            # lidarcap
            pc = read_pcd(point_cloud_file)
            
            make_horizon = read_json_file(args.velodyne_dir.split('velodyne')[
                                          0] + '/make_horizon.json')
            make_horizon = np.asarray(make_horizon[os.path.basename(args.velodyne_dir)])

            pc[:, :3] = pc[:, :3] @ make_horizon.T
            label_file_name = None
            out_file_name = os.path.join(output_dir, point_cloud_name)
        else:
            continue
        # ground, non_ground = filter_ground(pc[:, :3])
        feed_dict, predictions = inference(
            pc, model, label_file_name=label_file_name)

        output = o3d.geometry.PointCloud()
        output.points = o3d.utility.Vector3dVector(feed_dict['pc'][:,:3])
        output.colors = o3d.utility.Vector3dVector(SEM_COLOR[predictions]/255)
        o3d.io.write_point_cloud(out_file_name, output)
        print(f'\rFile save in {out_file_name}. Processed {i:d}/{files_num}', end='\r',flush=True)
