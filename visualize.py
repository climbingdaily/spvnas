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

from model_zoo import minkunet, spvcnn, spvnas_specialized
import open3d as o3d
from core.datasets.semantic_poss import LABEL_DICT, KEPT_LABELS, SEM_COLOR


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

def create_label_map_poss():
    reverse_label_name_mapping = {}
    label_map = np.zeros(260)
    cnt = 0
    unlabel_id = 0
    for label_id in LABEL_DICT:
        if label_id == 0:
            label_map[label_id] = unlabel_id
            reverse_label_name_mapping['unlabeled'] = 0
        elif label_id == 4 or label_id == 5:
            label_map[label_id] = 1
            reverse_label_name_mapping['pedestrian'] = 1
            cnt = 2
        elif LABEL_DICT[label_id] in KEPT_LABELS:
            label_map[label_id] = cnt
            reverse_label_name_mapping[LABEL_DICT[label_id]] = cnt
            cnt += 1
        else:
            label_map[label_id] = unlabel_id
            reverse_label_name_mapping['unlabeled'] = 0
    # self.num_classes = cnt
    # self.angle = 0.0
    return label_map

def create_label_map(num_classes=19):
    name_label_mapping = {
        'unlabeled': 0,
        'outlier': 1,
        'car': 10,
        'bicycle': 11,
        'bus': 13,
        'motorcycle': 15,
        'on-rails': 16,
        'truck': 18,
        'other-vehicle': 20,
        'person': 30,
        'bicyclist': 31,
        'motorcyclist': 32,
        'road': 40,
        'parking': 44,
        'sidewalk': 48,
        'other-ground': 49,
        'building': 50,
        'fence': 51,
        'other-structure': 52,
        'lane-marking': 60,
        'vegetation': 70,
        'trunk': 71,
        'terrain': 72,
        'pole': 80,
        'traffic-sign': 81,
        'other-object': 99,
        'moving-car': 252,
        'moving-bicyclist': 253,
        'moving-person': 254,
        'moving-motorcyclist': 255,
        'moving-on-rails': 256,
        'moving-bus': 257,
        'moving-truck': 258,
        'moving-other-vehicle': 259
    }

    for k in name_label_mapping:
        name_label_mapping[k] = name_label_mapping[k.replace('moving-', '')]
    train_label_name_mapping = {
        0: 'car',
        1: 'bicycle',
        2: 'motorcycle',
        3: 'truck',
        4: 'other-vehicle',
        5: 'person',
        6: 'bicyclist',
        7: 'motorcyclist',
        8: 'road',
        9: 'parking',
        10: 'sidewalk',
        11: 'other-ground',
        12: 'building',
        13: 'fence',
        14: 'vegetation',
        15: 'trunk',
        16: 'terrain',
        17: 'pole',
        18: 'traffic-sign'
    }

    label_map = np.zeros(260) + num_classes
    for i in range(num_classes):
        cls_name = train_label_name_mapping[i]
        label_map[name_label_mapping[cls_name]] = min(num_classes, i)
    return label_map.astype(np.int64)


cmap = np.array([
    [245, 150, 100, 255],
    [245, 230, 100, 255],
    [150, 60, 30, 255],
    [180, 30, 80, 255],
    [255, 0, 0, 255],
    [30, 30, 255, 255],
    [200, 40, 255, 255],
    [90, 30, 150, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [75, 0, 175, 255],
    [0, 200, 255, 255],
    [50, 120, 255, 255],
    [0, 175, 0, 255],
    [0, 60, 135, 255],
    [80, 240, 150, 255],
    [150, 240, 255, 255],
    [0, 0, 255, 255],
])
cmap = cmap[:, [2, 1, 0, 3]]  # convert bgra to rgba


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

def read_pcd(pcd_path):
    lines = []
    num_points = None

    with open(pcd_path, 'rb') as f:
        for line in f:
            lines.append(line.strip())
            if line.startswith('POINTS'):
                num_points = int(line.split()[-1])
    assert num_points is not None
    
    points = []
    for line in lines[-num_points:]:
        x, y, z, i = list(map(float, line.split()))
        #这里没有把i放进去，也是为了后面 x, y, z 做矩阵变换的时候方面
        #但这个理解我选择保留， 因为可能把i放进去也不影响代码的简易程度
        points.append((np.array([x, y, z, 1.0]), i))

    return points

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')

    # parser.add_argument('--velodyne-dir', type=str, default='/hdd/dyd/SemanticPOSS/sequences/05/velodyne')
    parser.add_argument('--velodyne-dir', type=str, default='/hdd/dyd/lidarcap/pointclouds/6')
    
    parser.add_argument('--model', type=str,
                        default='SemanticKITTI_val_SPVCNN@65GMACs')
    # args = parser.parse_args()
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    output_dir = os.path.dirname(args.velodyne_dir) + '/segments_' + os.path.basename(args.velodyne_dir)
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
            
            pc = np.fromfile(point_cloud_file,
                     dtype=np.float32).reshape(-1, 4)   # 读取点云
            label_file_name = os.path.join(args.velodyne_dir.replace(
                'velodyne', 'labels'), point_cloud_name.replace('.bin', '.label'))
            # vis_file_name = point_cloud_name.replace(velodyne_dir'.bin', '.png')
            # gt_file_name = point_cloud_name.replace('.bin', '_GT.png')
            out_file_name = os.path.join(
                output_dir, point_cloud_name.replace('.bin', '.ply'))
            
        elif point_cloud_name.endswith('.pcd'):
            from pypcd import pypcd
            pc_pcd = pypcd.PointCloud.from_path(point_cloud_file)
            pc = np.zeros((pc_pcd.pc_data.shape[0],4)) 
            pc[:, 0] = pc_pcd.pc_data['x']
            pc[:, 1] = pc_pcd.pc_data['y']
            pc[:, 2] = pc_pcd.pc_data['z']
            pc[:, 3] = pc_pcd.pc_data['intensity']
            # make_horizon = np.array([
            #     [0.027556484565, -0.999603390694, 0.005807927810, 14.860293921842],
            #     [0.965788960457, 0.028122182935, 0.257799893618, -6.737565234189],
            #     [-0.257861018181, -0.001494826516, 0.966180920601, 2.864406873059],
            #     [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
            # ])
            make_horizon = np.array([
                [-0.027227737010, - 0.999508678913, - 0.015521888621, 20.598001382560],
                [0.970868110657, - 0.030139083043, 0.237711712718, - 9.102568291975],
                [-0.238062754273, - 0.008597354405, 0.971211731434, 23.602949427876],
                [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]])
            pc[:, :3] = pc[:, :3] @ make_horizon[:3, :3].T + make_horizon[:3, 3]
            label_file_name = None
            out_file_name = os.path.join(output_dir, point_cloud_name)
        else:
            continue

        feed_dict, predictions = inference(
            pc, model, label_file_name=label_file_name)

        output = o3d.geometry.PointCloud()
        output.points = o3d.utility.Vector3dVector(feed_dict['pc'][:,:3])
        output.colors = o3d.utility.Vector3dVector(SEM_COLOR[predictions]/255)
        o3d.io.write_point_cloud(out_file_name, output)
        print(f'\rProcessed {i:d}/{files_num}', end='\r',flush=True)
        # fig = draw_lidar(feed_dict['pc'], predictions.astype(np.int32))
        # mlab.savefig(f'{output_dir}/{vis_file_name}')
        # if label is not None:
            # fig = draw_lidar(feed_dict['pc'], feed_dict['targets_mapped'])
            # mlab.savefig(f'{output_dir}/{gt_file_name}')


