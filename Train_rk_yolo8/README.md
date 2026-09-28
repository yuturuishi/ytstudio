### Train_rk_yolo8
* 官网：https://www.yuturuishi.com
* 微信：yuturuishi
* gitee开源地址：https://gitee.com/yuturuishi/BXC_AutoML
* github开源地址：https://github.com/yuturuishi/BXC_AutoML


### 第一步（安装模型训练环境）
* 如下提供三种安装模型训练环境的方式，任选其一

#### 安装方式1：在x86（Linux）安装模型训练环境

~~~
//首先，对于新手而言，强烈建议使用Ubuntu20.04，并确保系统默认Python版本是Python3.8
并不是不支持其它系统或其他Python版本，主要是因为Python依赖库版本很容易不一致，不一致后虽然容易解决，但新手似乎不具备这个能力

//创建虚拟环境
python -m venv venv

//linux系统激活虚拟环境
source venv/bin/activate

//解压rk_yolo8_source.zip
注意注意注意！！！解压rk_yolo8_source.zip到当前文件夹，不要出现多层级，确保解压后根目录下出现了ultralytics文件夹

//开始安装依赖库（本仓库的安装依赖toml文件，并不是传统requirements.txt的方式，请注意区别）

//第1步
python -m ensurepip --default-pip
//第2步
pip install poetry
//第3步
pip install .

//（如果电脑没有英伟达显卡，可以安装cpu版pytorch2.1.0）
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0

//（如果电脑有英伟达显卡，并且安装了显卡驱动，可以安装cuda版pytorch2.1.0加速）
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121


~~~

#### 安装方式2：在x86（Windows）安装模型训练环境

~~~
//首先，对于新手而言，强烈建议使用Python3.10
并不是不支持其他Python版本，主要是因为Python依赖库版本很容易不一致，不一致后虽然容易解决，但新手似乎不具备这个能力

//创建虚拟环境
python -m venv venv

//windows系统激活虚拟环境
venv\Scripts\activate

//升级pip版本（并不是必要的）
python -m pip install --upgrade pip

//解压rk_yolo8_source.zip
注意注意注意！！！解压rk_yolo8_source.zip到当前文件夹，不要出现多层级，确保解压后根目录下出现了ultralytics文件夹

//开始安装依赖库（本仓库的安装依赖toml文件，并不是传统requirements.txt的方式，请注意区别）

//第1步
python -m ensurepip --default-pip
//第2步
pip install poetry
//第3步
pip install .

//（如果电脑没有英伟达显卡，可以安装cpu版pytorch2.1.0）
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0

//（如果电脑有英伟达显卡，并且安装了显卡驱动，可以安装cuda版pytorch2.1.0加速）
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121


~~~

#### 安装方式3：在x86安装docker版模型训练环境
~~~
//暂未发布，建议采用安装方式1或安装方式2

~~~


### 第二步（训练模型）
~~~
//查看已安装的yolo8版本
yolo -V

//训练检测模型（gpu版本）
yolo detect train model=yolov8n.pt data=xxx/data.yaml batch=-1 epochs=1000 imgsz=640 save_period=5 device=cuda 
（注意：关于xxx/data.yaml文件，是指向训练数据集的配置文件，如果不清楚可以先下载一份数据集参考下）

//训练检测模型（cpu版本）
yolo detect train model=yolov8n.pt data=xxx/data.yaml batch=-1 epochs=1000 imgsz=640 save_period=5 device=cpu

//测试模型
yolo detect predict model=runs/train/best.pt source=test.jpg
~~~


### 第三步（模型转换：pt模型->onnx模型）

~~~

//windows系统安装依赖库（建议python3.10，其他python的依赖库版本可能会有所不同，也可以根据报错自行修改版本）：
pip install onnxruntime==1.19.0 onnx==1.16.1  -i https://pypi.tuna.tsinghua.edu.cn/simple

//linux系统依赖库（建议python3.8，其他python的依赖库版本可能会有所不同，也可以根据报错自行修改版本）：
pip install onnxruntime==1.16.3 onnx==1.16.1  -i https://pypi.tuna.tsinghua.edu.cn/simple

//最最最重要的一步，将pt模型转换为rk设备支持的onnx模型(不要怀疑文档的正确性)
yolo export model=yolov8n.pt format=rknn

~~~


### 第四步（模型转换：onnx模型->rknn模型）

~~~
//将rk设备支持的onnx转换为rknn模型（参考文档）
https://gitee.com/yuturuishi/BXC_AutoML/tree/master/onnx2rknn

~~~

### 其他
* rk版yolo8代码和yolo8官方代码训练的pt模型，是可以互相通用的，但是在转换rknn时，是有区别的
* 务必不要从yolo8的github官方仓库下载代码！！！本模块的代码来自于 https://github.com/airockchip/ultralytics_yolov8（https://github.com/airockchip/ultralytics_yolov8 来自于 https://github.com/ultralytics/ultralytics (8.2.82)）


### 训练数据集（免费下载）
* 训练数据集-夸克网盘下载地址：https://pan.quark.cn/s/5dcc2f724bcc
* 检测密集人群数据集20250625
* 检测明厨亮灶数据集20250625
* 检测攀爬数据集20250624
* 检测抽烟数据集20241012
* 检测打架数据集20241012
* 检测反光衣数据集20241013
* 检测粉尘数据集20241013
* 检测火焰烟雾数据集20241012
* 检测人头和安全帽数据集20241013
* 检测人体5动作-站着-摔倒-坐-深蹲-跑数据集20241012
* 检测学生3状态数据集20241022
* 检测睡岗数据集20241210
* 检测手机数据集20251206
* 猫狗2分类数据集-夸克网盘下载地址 https://pan.quark.cn/s/982dd16cb29d
* 车型9分类数据集-夸克网盘下载地址 https://pan.quark.cn/s/f698d0e99a4b
* 打鼾+不打鼾语音识别2分类数据集-夸克网盘下载地址：https://pan.quark.cn/s/4d83dabff0a6