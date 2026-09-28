### Train_yolo8
* 官网：https://www.yuturuishi.com
* 微信：yuturuishi
* gitee开源地址：https://gitee.com/yuturuishi/BXC_AutoML
* github开源地址：https://github.com/yuturuishi/BXC_AutoML


### 安装Python
* torch 2.1.x 仅支持 Python 3.8~3.11，建议用 Python3.10；Python3.12 请改用 torch 2.2.0，见注意事项②）

### 推荐使用Python虚拟环境
* 在使用python开发项目时，推荐使用python的虚拟环境，因为同一台电脑上很可能会安装多个python项目，而不同的python项目可能会使用不同的依赖库，为了避免依赖库不同而导致的冲突，强烈建议使用python虚拟环境
* 关于如何使用python虚拟环境，非常简单，文档最下面提供Windows系统和Linux系统创建和使用虚拟环境的方法


### Python安装 pytorch-cpu版本yolo8（按顺序执行，numpy 必须最后装，原因见下方注意事项）
* pip install ultralytics==8.2.86 -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip install torch==2.1.2 torchvision==0.16.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip install opencv-python==4.8.1.78 -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip install numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple （Windows-Python3.10；必须最后装，见注意事项①）
* pip install numpy==1.24.4 -i https://pypi.tuna.tsinghua.edu.cn/simple （Linux-Python3.8）

### Python安装 pytorch-cuda版本yolo8（按顺序执行，numpy 必须最后装，原因见下方注意事项）
* pip install ultralytics==8.2.86 -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
* pip install opencv-python==4.8.1.78 -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip install numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple （Windows-Python3.10；必须最后装，见注意事项①）
* pip install numpy==1.24.4 -i https://pypi.tuna.tsinghua.edu.cn/simple （Linux-Python3.8）

* 注意①：**torch 2.1.x / 2.2.x 都是基于 NumPy 1.x 编译的，和 numpy 2.x 不兼容**。装 ultralytics 时 pip 会自动带上 numpy 2.x，启动就报 `Failed to initialize NumPy: _ARRAY_API not found`，训练一进数据加载就崩。所以**无论按什么顺序装，最后一步必须把 numpy 钉回 1.x（Python3.9~3.12 用 1.26.4，Python3.8 用 1.24.4）**。以后在这个环境里升级/新装其他库时，若 pip 提示要升级 numpy 到 2.x，记得拒绝。
* 注意②：**Python3.12 装不了上面的 torch 2.1.x 组合**（没有 cp312 的 wheel，cp312 从 torch 2.2.0 才开始提供）。如果用 Python3.12，请改用：`torch==2.2.0 torchaudio==2.2.0 torchvision==0.17.0`（cu121 地址相同），ultralytics 8.2.86 与之兼容。
* 注意③：安装pytorch-gpu训练环境，请根据自己的电脑硬件选择cuda版本，比如我上面选择的https://download.pytorch.org/whl/cu121，并非适用所有电脑设备，请根据自己的设备选择


### 快速开始
~~~
//查看已安装的yolo8版本
yolo -V

//验证环境是否正常（应输出 ultralytics 版本 / cuda True，且无任何 NumPy 警告）
python -c "import ultralytics,torch; print(ultralytics.__version__); print('cuda', torch.cuda.is_available())"

//训练检测模型（gpu版本）
yolo detect train model=yolov8n.pt data=xxx/data.yaml batch=-1 epochs=1000 imgsz=640 save_period=5 device=cuda 
（注意：关于xxx/data.yaml文件，是指向训练数据集的配置文件，如果不清楚可以先下载一份数据集参考下）

//训练检测模型（cpu版本）
yolo detect train model=yolov8n.pt data=xxx/data.yaml batch=-1 epochs=1000 imgsz=640 save_period=5 device=cpu

//测试模型
yolo detect predict model=runs/train/best.pt source=test.jpg

//将pt模型转换为onnx格式模型
//依赖库：pip install onnxruntime==1.19.0 onnx==1.16.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
yolo export model=best.pt format=onnx

//将pt模型转换为openvino格式模型
//依赖库：pip install openvino==2024.3.0 openvino-dev==2024.3.0 onnxruntime==1.19.0 onnx==1.16.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
yolo export model=best.pt format=openvino

~~~


### mo命令将pt模型转换为openvino模型（方式二）
* mo命令是openvino官方提供的模型转换工具
* mo参考文档 https://blog.csdn.net/qq_44632658/article/details/131270531
~~~
//安装mo命令行，将onnx转换为openvino模型
//依赖库：pip install openvino==2024.3.0 openvino-dev==2024.3.0 onnxruntime==1.19.0 onnx==1.16.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
yolo export model=yolov8n.pt format=onnx
mo --input_model yolov8n.onnx  --output_dir yolov8n_openvino_model
~~~

### trtexec命令将pt模型转换为tensorrt模型
~~~
//将pt模型转换为onnx格式模型
yolo export model=yolov8n.pt format=onnx

//将onnx模型转换为tensorrt格式模型(在ovtrt版本的视频行为分析系统的xcms_core文件夹下面有trtexec工具)
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.fp16.engine --fp16
~~~

### yolo8训练相关文档
* yolo8官方训练参数：https://docs.ultralytics.com/zh/modes/train/#train-settings
* yolo8训练参数参考: https://blog.csdn.net/weixin_51692073/article/details/133875143
* yolo8开源地址：https://github.com/ultralytics/ultralytics
* yolo8官方模型-夸克网盘下载地址：https://pan.quark.cn/s/59a17310b205

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
### Windows系统安装Python虚拟环境
~~~

//创建虚拟环境
python -m venv venv

//切换到虚拟环境
venv\Scripts\activate

//更新虚拟环境的pip版本（可以不更新）
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

~~~

### Linux系统安装Python虚拟环境

~~~
//创建虚拟环境
python -m venv venv

//激活虚拟环境
source venv/bin/activate

//更新虚拟环境的pip版本（可以不更新）
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

~~~



