## 三、3D点云目标检测

采用PVRCNN模型，PVRCNN源码被作者收集到Open-MMLab/OpenPCDet中。

[GitHub - open-mmlab/OpenPCDet: OpenPCDet Toolbox for LiDAR-based 3D Object Detection.](https://github.com/open-mmlab/OpenPCDet)

### Ⅰ 环境配置

由于PV-RCNN环境需要配置的内容较多，且大多数软件彼此之间有依赖性，在经历了多次尝试后最终决定使用nvidia docker。

#### 1.docker介绍

* 容器级别的虚拟化
* 不是基于硬件虚拟化
* 和宿主机共享操作系统内核和资源
* 轻量
* 方便的构建&部署应用程序
* 采用虚拟化控制

#### 2.Nvidia-docker介绍

docker原生并不支持在他生成的容器中使用Nvidia  GP资源。nvidia-docker是对docker的封装，提供一些必要的组件可以很方便的在容器中用GPU资源执行代码。从下面的图中可以很容器看到nvidia-docker共享了宿主机的CUDA  Driver。

![img](file:///C:\Users\19247\Documents\Tencent Files\1924721559\Image\C2C\AE67A310AB3D97702DDBC81B76AAA490.png)

#### 3.Docker安装

```plain
curl http://get.docker.com | sh \ && sudo systemctl --now enable docker
```

这里有要求docker的版本Docker>=19.03

#### 4.Nvidia-docker安装

设置稳定版本库和GPG密钥

```plain
$ distribution=$(. /ect/os-release;echo $ID$VERSION_ID) \ 
   && curl -s -L http://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L http://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /ect/apt/sources.list
```

在更新软件包列表后，安装nvidia-docker2软件包（以及依赖项）。

```plain
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

重启（不要忘记！）

```plain
sudo systemctl restart docker
```

文档地址：
[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html](

#### 5.创建环境

由于spconv给了可以使用的镜像，我们在此处直接进入

```plain
sudo docker run --rm -it --gpus all scrin/dev-spconv:c0f7284c51205f259cb270747259c01cd03cc2ed /bin/bash
```

（--rm表示这个container运行结束后就会被删除，不加会保留）
此镜像配置python3.8、cuda10.2

#### 6.查看spconv版本

docker镜像里面的文件是没法用图形界面看的，只能在终端里敲命令，源代码被放到了/root/spconv目录里

```plain
cd /root/spconv
cat setup.py
```

本组使用的是spconv1.2版本

#### 7.进入docker环境

首次进入（即创建一个docker）

```
sudo docker run --rm -it --gpus all scrin/dev-spconv:c0f7284c51205f259cb270747259c01cd03cc2ed /bin/bash
```

后续进入（进入到原来的docker中）

长ID获取方法从后续步骤中有提到

```
sudo docker start $docker长ID
sudo docker exec -it $docker长ID /bin/bash
```

### Ⅱ pcdet v0.3安装

OpenPCDet在12月又更新了v0.5，但我们在12月前就进行了学习操作，所以这里采用的使OpenPCDet v0.3。

#### 0.Github克隆源代码

```
git clone https://github.com/open-mmlab/OpenPCDet.git
```

此方法可能会因网络问题而失败。

#### 1.在Github上下载到本地后拷贝到环境中

下载链接：[GitHub - open-mmlab/OpenPCDet: OpenPCDet Toolbox for LiDAR-based 3D Object Detection.](https://github.com/open-mmlab/OpenPCDet)

##### 将本地文件拷贝到docker容器中：

- 打开本地环境终端

- 查找容器

  ```
  sudo docker ps -a
  ```

- 获取容器长id

  ```
  sudo docker inspect -f '{{.ID}}' $NAME
  #$NAME为上一步中所查找到的所使用的docker容器的名称
  ```

- 拷贝本地文件到docker容器

  ```
  sudo docker cp 本地文件路径+文件 长ID:容器路径(+文件)
  
  #将docker容器中的文件拷贝到本地
  sudo docker cp 长ID:容器路径+文件 本地路径(+文件) 
  
  #查看docker容器路径方法
  pwd
  #查看本地文件路径方法 右键->属性
  ```

##### 将OpenPCDet-master压缩包解压

```
unzip OpenPCDet-master.zip
```

#### 2.安装anaconda

注：此步骤在库作者文档中不存在，文档中克隆代码后步骤应为安装依赖包，但是由于docker环境中python=3.8导致后续 mayavi安装接连报错，所以采用安装anaconda来改变docker环境.

如果环境python<=3.7可尝试按库作者文档步骤进行。

但建议使用anaconda环境进行。

 - 在annconda官网下载需要的版本 此链接为2020.2版本python=3.7(注意最好不要下载最新版本)
https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
 - 将下载的安装包按上述拷贝OpenPCDet-master的方法从本地下载目录复制到docker目录下
 - cd到安装包所在的docker路径中(复制到/root路径下忽略此步骤)
 - bash+安装包文件名进行安装
 - 重启docker
 - 验证是否安装成功 conda --version查看版本如正常显示则安装成功

#### 3.安装requirements.txt中的包

```
pip install -r requirements.txt
#不建议使用此方法 因为环境中有些包已经安装且可能会报错
```

- 查看已安装的包(基本上是anaconda自带包)

  ```
  conda list
  ```

- torch和torchvision下载

  - 根据cuda版本选择相应的torch版本进行下载

    [Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)

    本组采用cuda10.1+torch1.5.0+torchvision0.6.0

  ```
  pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html 
  
  conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
  
  #安装过程可能会因网络问题超时 在其后加入
   --default-timeout=1000
  #即
  pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html --default-timeout=1000
  ```

  **注：**如果使用spconv v1.0则下载torch 1.1

  ​        如果使用spconv v1.2则下载torch1.3+ (使用docker则要使torch1.4+)

- 其他包下载

  直接使用pip安装即可(或者conda安装)

  ```
  pip install numba
  ...
  #注意直接安装kornia  pip会安装最新版本导致与torch版本不匹配 
  #如果也使用torch v1.5.0则下载kornia v0.3.0
  pip install kornia==0.3.0
  ```

  **注:** 注意查看numpy版本 使numpy<1.21,>=1.17.否则下一步可能会报错。

#### 4.安装pcdet包

- 切换到OpenPCDet-master路径下 方法：cd+路径

- ```
  python setup.py develop
  ```

### Ⅲ 运行demo

#### 1.下载mayavi

Windows系统下可参考：https://www.zhihu.com/question/383305179 （未尝试，本过程不使用Windows系统，只供后续相关工作参考）

###### Linux系统：

python2环境下可以直接使用pip安装 会将相关依赖包一并下载

```
pip install mayavi
```

python3.7以下可使用以下步骤

```
pip install vtk==8.1.2
pip install mayavi
```

##### 使用anaconda安装：

- 安装vtk 

  - pip安装

  - 官网([Download | VTK](https://vtk.org/download/))下载安装包

    python3.7选择相应版本下载

- 安装mayavi

  [Installation — Viscid 1.0.0 documentation (viscid-hub.github.io)](https://viscid-hub.github.io/Viscid-docs/docs/master/installation.html#installing-mayavi)

  ```
  conda install -c viscid-hub mayavi
  ```

- 在docker环境下需安装libgl1-mesa-gl

  ```
  apt-get update#(每次使用apt-get之前都要进行该步骤)
  apt-get install libgl1-mesa-gl
  ```

- mayavi需要图形化界面包的支持，anaconda自带包pyqt可能不兼容

  ```
  apt-get install python3-pyside
  ```

- 验证mayavi和vtk是否安装成功

  ```
  python
  import vtk
  import mayavi
  ```

  不报错即成功

#### 2.下载KITTI数据集

可在官网下载，但速度慢，建议使用网盘，已整理好。

链接：https://pan.baidu.com/s/1zlM6jOW66-Fhh5S47tbw0A 
提取码：kitt

将数据集按此序列存放：

```
OpenPCDet
├── data
│  ├── kitti
│  │  │── ImageSets
│  │  │── training
│  │  │  ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│  │  │── testing
│  │  │  ├──calib & velodyne & image_2
├── pcdet
├── tools
```

#### 3.下载预训练模型

可根据库作者文档下载，但可能会有网络不畅通的情况，可使用网盘下载。

链接：https://pan.baidu.com/s/10YEXzpFkp3ezwMbRiKEcjw 
提取码：pvrc

下载好的文件放到/tools路径下

#### 4.运行demo.py

切换到tools路径下

```
python demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt pv_rcnn_8369.pth \
    --data_path /tools/OpenPCDet-master/data/kitti/training/velodyne/000000.bin
```

以下为可能的报错情况及解决方法：

###### (1)报错： ModuleNotFoundError: No module named 'importlib_resources'

解决：可能是缺少pyface包
pip install pyface 
  -->importlib-resources 5.4.0 需要 zip>=3.1.0
     pip uninstall zipp(2.2.0)
     pip install zipp(3.6.0)

###### (2)报错： ModuleNotFoundError: No module named 'vtkOpenGLKitPython'

原因：vtk安装失败
按照anaconda方法安装mayavi后
          apt-get update(使用apt-get之前必备)
          apt-get install libgl1-mesa-gl
          !!!apt-get install python3-pyside

###### (3)报错: numpy 1.21.4 is installed but numpy<1.21,>=1.17 is required by {'numba'}

更换numpy版本
安装成功

###### (4)报错：qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.

This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
解决：apt-get install libxcb-xinerama0

###### (5)qt.qpa.screen: QXcbConnection: Could not connect to display  Could not connect to any X display.

解决：vim ~/.bashrc
      加入export QT_QPA_PLATFORM='offscreen'
      source ~/.bashrc

###### (6)ModuleNotFoundError: No module named 'scipy'

解决：pip install scipy

###### (7)ModuleNotFoundError: No module named 'skimage'

解决： pip install scikit-image

### Ⅳ 训练模型

#### 0.坐标转换

使用/pvrcnn/pointconvert.py将自己的.bin点云文件转换为OpenPCDet的标准格式
(cx, cy, cz, dx, dy, dz, heading)

#### 1.统计点云数据

使用/pvrcnn/pc_sum.py统计点云文件数据范围输出final.txt。包含每个点云文件的数据范围。

#### 2.修改kitti_dataset.yaml

修改/tools/cfgs/dataset_configs/kitti_dataset.yaml

第3行，点云范围，根据实际点云范围调整

```python
POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1] 
```

顺序为[x_min, y_min, z_min, x_max, y_max, z_max]

第65-72行，体素参数，根据点云的稠密程度调整

```python
VOXEL_SIZE: [0.05, 0.05, 0.1] #体素大小
MAX_POINTS_PER_VOXEL: 5 #每个体素的最高点数
      MAX_NUMBER_OF_VOXELS: { #体素数量上限
        'train': 16000,
        'test': 40000
      }
```

#### 3.更改点云数据velodyne

  本组的点云数据文件名叫 pointcloud
  将自己的pointcloud文件放入data/kitti/training
  进入/pcdet/datasets/kitti/kitti_object_eval_python/kitti_common.py
  将第47行代码中的'velodyne'改为'pointcloud'

**注：**1.2.3步骤也可以直接使用kitti数据集 本组使用kitti数据集

​       在使用自己的训练集训练时需要进行训练集的点云标注，可参考[(16条消息) 点云标注工具：1.PCAT_m0_37957160的博客-CSDN博客_点云标注系统](https://blog.csdn.net/m0_37957160/article/details/106333630)



#### 4.训练

切换到/tools路径下

使用的参数 batch_size 和 workers 是2， epochs 是5，储存路径命名为 'mydata_1'

```
python train.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml 
                --batch_size 2 
                --workers 2 
                --epochs 5 
                --extra_tag 'mydata_1'
```

根据实际情况进行修改

#### 5.测试

训练的结果储存在 output/kitti_models/pv_rcnn/default/ckpt
文件是 checkpoint_epoch_5.pth ， batch_size 是2
extra_tag 表示储存路径的一个文件夹名，最好和训练的参数保持一致 

```
python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml 
                --batch_size 2 
                --ckpt../output/kitti_models/pv_rcnn/mydata_1/ckpt/checkpoint_epoch_5.pth                 --extra_tag 'mydata_1'
```

#### 6.可视化

使用demo.py中提供的可视化代码

```
python demo.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml 
               --ckpt ../output/kitti_models/pv_rcnn/mydata_1/ckpt/checkpoint_epoch_5.pth                --data_path ../data/kitti/training/pointcloud/Mesh08.bin
```

###### 没有检测到目标 可能报错：

ValueError: zero-size array to reduction operation minimum which has no identity

