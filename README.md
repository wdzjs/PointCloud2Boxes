# PointCloud2Boxes

## 项目名称

该项目为点云识别项目，具有将点云数据识别为设备坐标的功能
 
## 项目描述

输入为识别对象的文件名，输出为json的HTTP报文。
 
## 环境配置

requirements.txt 中包含项目所需依赖的必要库
使用以下命令安装必要库
`pip install -r requirements.txt`
除此之外根据显卡cuda版本还需安装pytorch，在下面地址中根据软硬件配置安装对应的pytorch版本
[“https://pytorch.org/get-started/locally/”](https://pytorch.org/get-started/locally/)

## 项目入口及启动

请在flask_点云to盒子入口文件.py中设置root位置，既数据所在文件夹
https://github.com/wdzjs/PointCloud2Boxes/blob/6f90a343221239c55bfb9f3bb7017103421b0699/flask_%E7%82%B9%E4%BA%91to%E7%9B%92%E5%AD%90%E5%85%A5%E5%8F%A3%E6%96%87%E4%BB%B6.py#L112

服务启动，加载模型使用一下命令
`python flask_点云to盒子入口文件.py`

根据提示在对应端口发送HTTP GET请求，对应的方法为根据地址预测坐标 /predict/"filename"，根据地址获取坐标真值 /get_box/"filename" , 其中“filename”替换为所需文件名
如：
[“http://127.0.0.1:5000/predict/Hongle330dawaji”](http://127.0.0.1:5000/predict/Hongle330dawaji)
[“http://127.0.0.1:5000/get_box/Hongle330dawaji”](http://127.0.0.1:5000/get_box/Hongle330dawaji)

## 重点代码介绍

sparsification() #稀疏化 调整角度 使用这个函数处理las文件会将处理好的文件替换
las_trans()  # 数据转化为模型可识别的npy数据
vald3()  # 用模型进行预测，输出为带label的点云数据
sem_label()  # 输出

**sparsification()函数请尽量不要使用，会改变设定的参数**

inner/test_sub3d.py中的num_votes参数会修改推理次数，推理次数越多，准确率越高，速度越慢。
https://github.com/wdzjs/PointCloud2Boxes/blob/6f90a343221239c55bfb9f3bb7017103421b0699/inner/test_sub3d.py#L37


inner/data_utils/D3SubDataLoader.py中 ScannetDatasetWholeScene类的stride参数会改变模型在推理时对点云的区划范围大小，以米为单位。stride越大，区划范围越大，区划数目越少，速度越快，精度越低。相反stride越小，区划范围越小，区划数目越多，速度越慢，精度越高。
https://github.com/wdzjs/PointCloud2Boxes/blob/6f90a343221239c55bfb9f3bb7017103421b0699/inner/data_utils/D3SubDataLoader.py#L90


inner/output.py中19行的eps参数会改变聚类时同类判定距离，太大导致多个物体识别为同一个，太小导致一个物体识别为多个，推荐为1，可以调整尝试。min_points参数会改变聚类时离散点的判定，太大导致真实物体被识别为噪声，太小导致噪声被误识别为真实物体，推荐为20。
https://github.com/wdzjs/PointCloud2Boxes/blob/6f90a343221239c55bfb9f3bb7017103421b0699/inner/output.py#L19


## 输出格式
如：
[
    {
        "label":"挖车", 
        "xmin":-32.6647,
        "ymin":10.5764,
        "zmin":-2.0766,
        "xmax":-28.7071,
        "ymax":18.2002,
        "zmax":2.0543
    }
]
