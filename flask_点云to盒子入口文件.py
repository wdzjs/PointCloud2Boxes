# coding:utf-8
'''
Created on 2024年4月1日
    接口，用于对接上层应用
@author: yxy
'''
import importlib
import os
import sys
import pandas as pd
import threading

import torch
from flask import Flask, Response

from inner.las2npy import las_trans
from inner.output import sem_label
from inner.pointcloud_sparsification import sparsification
from inner.test_sub3d import vald3, parse_args

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'inner/models'))

mapping_dict = {0: u'GIS进线套管', 1: u'断路器', 2: u'刀闸', 3: u'变压器', 4: u'电压互感器', 5: u'电流互感器', 6: u'电容', 7: u'电感', 8: u'避雷器',
                9: u'工作人员', 10: u'杆塔', 11: u'基坑', 12: u'吊车', 13: u'挖车', 14: u'索道', 15: u'龙门架', 16: u'围栏', 17: u'背景'}


class ModelImport(object):
    '''
    用于311展厅模型
    '''

    def __init__(self, root):

        # 模型加载
        experiment_dir = 'inner/log/sem_seg/test'  # 模型路径
        NUM_CLASSES = 18  # 点云类型数
        model_name = "d3_sem_seg"
        self.MODEL = importlib.import_module(model_name)
        self.classifier = self.MODEL.get_model(NUM_CLASSES).cuda()
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth',
                                map_location=torch.device('cuda:0'))
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier = self.classifier.eval()

        # 生成路径
        self.root = root
        self.original_data = self.root + r'\\original_las\\'  # 原始las路径
        self.sparsed_rotated_data = self.root + r'\\sparsed_rotated_las\\'  # 处理后las路径
        self.output_npy = self.root + r'\\output\npy\\'  # 模型可读取数据路径
        self.output_txt = self.root + r'\\output\txt\\'  # 模型输出路径
        self.output_box = self.root + r'\\output\box\\'  # 盒子输出路径
        self.real_npy = self.root + r'\\real\npy\\'  # 模型带标签数据路径
        self.real_txt = self.root + r'\\real\txt\\'  # 模型真值路径
        self.real_box = self.root + r'\\real\box\\'  # 盒子真值路径

    def get_classifier(self):
        return self.classifier

    def get_box(self, filename, path='real'):
        if path == 'output':
            box_path = self.output_box + filename + r'.txt'
        else:
            box_path = self.real_box + filename + r'.txt'
        df = pd.read_csv(box_path, sep=' ', header=None)
        df.columns = ['label', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax']
        df['label'].replace(mapping_dict, inplace=True)
        json_str = df.to_json(orient='records', indent=4, force_ascii=False)
        return Response(json_str), 200, {'Content-Type': 'application/json; charset=utf-8'}

    def predict(self, filename):
        args = parse_args()

        # 文件保存路径
        original_data = self.original_data + filename + r'.las'  # 原始las路径
        sparsed_rotated_data = self.sparsed_rotated_data + filename + r'.las'  # 处理后las路径
        output_npy = self.output_npy + filename + r'.npy'  # 模型可读取数据路径
        output_txt = self.output_txt + filename + r'.txt'  # 模型输出路径
        output_box = self.output_box + filename + r'.txt'  # 盒子输出路径
        real_npy = self.real_npy + filename + r'.npy'  # 模型带标签数据路径
        real_txt = self.real_txt + filename + r'.txt'  # 模型真值路径
        real_box = self.real_box + filename + r'.txt'  # 盒子真值路径

        # 数据处理
        # sparsification(original_data, sparsed_rotated_data, 1) #稀疏化 调整角度 使用这个函数处理las文件会将处理好的文件替换
        las_trans(sparsed_rotated_data, output_npy)  # 数据转化为模型可识别的npy数据
        vald3(args, output_npy, output_txt, self.classifier)  # 用模型进行预测，输出为带label的点云数据
        sem_label(output_txt, output_box)  # 输出
        return self.get_box(filename, path='output')


@app.route("/start/<filename>", methods=["GET"])
def start(filename):
    # data = request.json  # 自己获取json
    return cs.predict(filename)


@app.route("/predict/<filename>", methods=["GET"])
def predict(filename):
    # data = request.json  # 自己获取json
    global background_thread
    background_thread = threading.Thread(target=cs.predict, args=(filename, ))
    background_thread.start()
    return Response("Start segmentation task"), 200, {'Content-Type': 'text/html; charset=utf-8'}

@app.route("/get_box/<filename>", methods=["GET"])
def get_box(filename):
    # data = request.json  # 自己获取json
    return cs.get_box(filename)


@app.route("/flask", methods=["GET"])
def flask_test():
    return "Flask is OK"


if __name__ == '__main__':
    root = r'H:\\data'
    cs = ModelImport(root)
    app.run(host='0.0.0.0', port=5000)
