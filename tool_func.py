from re import A
import CSF
import numpy as np
import sys
import os
from core.datasets.semantic_poss import LABEL_DICT, KEPT_LABELS
import json
import open3d as o3d

def filter_ground(xyz):
    csf = CSF.CSF()
    csf.params.bSloopSmooth = False  # 粒子设置为不可移动
    csf.params.cloth_resolution = 0.1  # 布料网格分辨率
    csf.params.rigidness = 3  # 布料刚性参数
    csf.params.time_step = 0.65
    csf.params.class_threshold = 0.05  # 点云与布料模拟点的距离阈值
    csf.params.interations = 300  # 最大迭代次数
    csf.setPointCloud(xyz)
    ground = CSF.VecInt()  # 地面点索引列表
    non_ground = CSF.VecInt() # 非地面点索引列表
    csf.do_filtering(ground, non_ground) # 执行滤波
    return ground, non_ground


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

def read_json_file(file_name):
    """
    Reads a json file
    Args:
        file_name:
    Returns:
    """
    with open(file_name) as f:
        try:
            data = json.load(f)
        except:
            data = None
    return data

def read_pcd(pcd_path):
    """[summary]

    Args:
        pcd_path ([type]): [description]
    """    
    from pypcd import pypcd
    pc_pcd = pypcd.PointCloud.from_path(pcd_path)
    pc = np.zeros((pc_pcd.pc_data.shape[0],4)) 
    pc[:, 0] = pc_pcd.pc_data['x']
    pc[:, 1] = pc_pcd.pc_data['y']
    pc[:, 2] = pc_pcd.pc_data['z']
    pc[:, 3] = pc_pcd.pc_data['intensity']
    return pc

def make_lidarcap_label(folder):
    """ 
    给LiDARCap数据集制作semantic-kitti式的label
    make semantic labels for pcd files in the @folder
    Args:
        folder (str): A folder that contains pcd files
    """    
    if not os.path.exists(folder):
        print ("Folder does not exist")
        return None
    point_clouds = sorted(os.listdir(folder))
    print(f'Generate labels in {folder}')
    for filename in point_clouds:
        pc_file = os.path.join(folder, filename)
        person_file = pc_file.replace(
            'velodyne', 'segment').replace('.pcd', '.ply')

        if not os.path.exists(person_file):
            continue
        
        person = o3d.io.read_point_cloud(person_file)
        pc = o3d.io.read_point_cloud(pc_file)

        obbox = person.get_oriented_bounding_box()
        person_inds = obbox.get_point_indices_within_bounding_box(pc.points)

        label = np.zeros(len(pc.points), dtype=np.int32)
        label[person_inds] = 4 # semantic_poss label 4: person

        label_file = person_file.replace(
            'segment', 'labels').replace('.ply', '.label')
        if not os.path.exists(os.path.dirname(label_file)):
            os.makedirs(os.path.dirname(label_file), exist_ok=True)

        label.tofile(label_file)
        print(f'Label saved to {label_file}', end='\r', flush=True)
    print('')


def save_pcd(folder):
    """ 
    save pcd file based on the label files
    Args:
        folder ([str]): [description]
    """    
    from core.datasets.semantic_poss import SEM_COLOR
    label_map = create_label_map_poss()
    
    point_clouds = sorted(os.listdir(folder))
    for i, filename in enumerate(point_clouds):
        pc_file = os.path.join(folder, filename)

        label_file = pc_file.replace('velodyne', 'labels').split('.')[0] + '.label'
        if not os.path.exists(label_file):
            continue

        pc = o3d.io.read_point_cloud(pc_file)
        all_labels = label_map[np.fromfile(
            label_file, dtype=np.uint32).reshape(-1) & 0xFFFF].astype(np.int64) # semantic labels

        pc.colors = o3d.utility.Vector3dVector(SEM_COLOR[all_labels] / 255)
        out_folder = os.path.join(os.path.dirname(
            folder), 'gt_semantic_' + os.path.basename(folder))
        os.makedirs(out_folder, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(out_folder, filename), pc)
        print(
            f'\rFile saved in {os.path.join(out_folder, filename)}. \
                Processed {i:d}/{len(point_clouds)}', end='\r', flush=True)

def crop_lidarcap(lidarcap_root, seqs = ['6', '25', '26']):
    """根据make horizon旋转矩阵和bounding box, 切割lidarcap点云
    Args:
        lidarcap_root ([str]): [description]
        seqs ([list]): [data folder list]
    """    
    from pypcd import pypcd
    # load rotation matrix to make the frame horizontal 
    rot_mat = {}
    rots = read_json_file(os.path.join(
        lidarcap_root, 'make_horizon.json'))
    for key in rots.keys():
        rot_mat[key] = np.asarray(rots[key])

    # load bounding boxes to crop the frame
    boxes = {}
    boxes_ = read_json_file(os.path.join(
        lidarcap_root, 'bounding_box.json'))
    for key in boxes_.keys():
        boxes[key] = np.asarray(boxes_[key])

    # Load lidarcap data
    # =================================================================
    for seq in seqs:
        pc_dir = os.path.join(lidarcap_root, 'velodyne', seq)
        crop_pc_dir = os.path.join(lidarcap_root, 'velodyne', 'crop_' + seq)
        label_dir = os.path.join(lidarcap_root, 'labels', seq)
        crop_label_dir = os.path.join(lidarcap_root, 'labels', 'crop_' + seq)

        if not os.path.exists(pc_dir):
            continue
        os.makedirs(crop_pc_dir, exist_ok=True)
        os.makedirs(crop_label_dir, exist_ok=True)
        for ii, x in enumerate(sorted(os.listdir(pc_dir))):
            if not x.endswith('.pcd'):
                continue
            
            block = read_pcd(os.path.join(pc_dir, x)).astype(np.float32)
            
            # 掰平地面的矩阵
            if seq in rot_mat.keys():
                make_horizon = rot_mat[key]
                block[:, :3] = block[:, :3] @ make_horizon.T 

            if seq not in boxes.keys():
                continue

            center = np.array(boxes[seq][0]['center'])
            length = np.array(boxes[seq][0]['length'])
            min_bound = center - length/2
            max_bound = center + length/2
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            
            # 根据bounding box切割点云
            crop_inds = box.get_point_indices_within_bounding_box(
                o3d.utility.Vector3dVector(block[:, :3]))


            # save pcd
            croped = pypcd.make_xyz_label_point_cloud(block[crop_inds], label='intensity')
            pypcd.save_point_cloud_bin(croped, os.path.join(crop_pc_dir, x))
            
            # load label and save cropped label
            label_file = os.path.join(label_dir, x.split('.')[0] + '.label')
            crop_label_file = os.path.join(crop_label_dir, x.split('.')[0] + '.label')
            if os.path.exists(label_file):
                label = np.fromfile(
                    label_file, dtype=np.uint32).reshape(-1) & 0xFFFF
                label[crop_inds].tofile(crop_label_file)
            
            print(
                f'\rFile saved in {os.path.join(crop_pc_dir, x)} ({ii:d}/{len(os.listdir(pc_dir))})', end='\r', flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    lidarcap_root = '/hdd/dyd/lidarcap'
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--velodyne-dir', type=str, default='/hdd/dyd/SemanticPOSS/sequences/05/velodyne')
    parser.add_argument('--velodyne-dir', type=str, default='/hdd/dyd/lidarcap/velodyne')
    args, opts = parser.parse_known_args()
    # for pp in sorted(os.listdir(args.velodyne_dir)):
    # make_lidarcap_label(os.path.join(args.velodyne_dir))
    # save_pcd(args.velodyne_dir)

    crop_lidarcap(lidarcap_root)