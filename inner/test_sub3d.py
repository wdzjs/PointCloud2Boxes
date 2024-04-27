"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from inner.data_utils.D3SubDataLoader import ScannetDatasetWholeScene
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import numpy as np

#0	GIS进线套管,1	断路器 ,2	刀闸 ,3	变压器 ,4	电压互感器 ,5	电流互感器 ,6	电容 ,7	电感 ,8	避雷器 ,9	工作人员 ,
# 10	杆塔,11	基坑,12	吊车,13	挖车,14	索道,15	龙门架,16	围栏,17	工作区域。

classes = ['GIS进线套管', '断路器', '刀闸', '变压器', '电压互感器', '电流互感器', '电容', '电感', '避雷器', '工作人员', '杆塔',
           '基坑', '吊车', '挖车' , '索道' , '龙门架' , '围栏' , '工作区域' , '背景' ]
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default='test', help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=1, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            #if weight[b, n] != 0 and not np.isinf(weight[b, n]):
            vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def vald3(args, sor, des, experiment_dir='inner/log/sem_seg/', classifier=None):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    current_dir = os.path.dirname(os.path.realpath(__file__))
    print(current_dir)
    experiment_dir = current_dir+'/log/sem_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 18
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    root = sor

    TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', block_points=NUM_POINT)
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    if classifier is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = BASE_DIR
        sys.path.append(os.path.join(ROOT_DIR, 'models'))
        model_name = "d3_sem_seg"
        MODEL = importlib.import_module(model_name)
        classifier = MODEL.get_model(NUM_CLASSES).cuda()
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',map_location=torch.device('cuda:0'))
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier = classifier.eval()

    with (torch.no_grad()):
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_data.shape[0], NUM_CLASSES))
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 6))

                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    #batch_data[:, :, 3:6] /= 1.0

                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])
                    #print(np.argmax(vote_label_pool))

            pred_label = np.argmax(vote_label_pool, 1)
            sceneoutput = np.c_[whole_scene_data,pred_label]

            outputfileroot = des
            if os.path.exists(outputfileroot):
                sceneoutput = np.loadtxt(outputfileroot)

            np.savetxt(outputfileroot, sceneoutput, fmt='%.4f')
        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    sor = r'H:\data\output\npy'
    des = r'H:\data\output\txt'
    for file in sorted(os.listdir(sor), reverse=True):
        inpath = os.path.join(sor, file)
        outpath = os.path.join(des, file.split('.')[0]+'.txt')
        print(inpath)
        vald3(args, inpath, outpath, experiment_dir='log/sem_seg/')
