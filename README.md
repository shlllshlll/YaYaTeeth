# YaYaTeeth

## 环境问题

### 环境安装

```bash
# conda导出
# Linux/Mac
conda env export --no-builds | grep -v "prefix" > config/conda_env.yml
# Windows
conda env export --no-builds | findstr -v "prefix" > config/conda_env.yml

# conda导入
conda create -n teeth -f config/conda_env.yml

# Pipenv环境安装
pipenv install
```

### 运行命令

```bash
# 模型训练
# Deeplab
./run.sh -G 0 -P s1 -M deeplab train 0819camonly
# HRNet
./run.sh -G "(0,)" -P s1 -M hrnet -R -1 train 0819camonly
# HRNet-OCR
./run.sh -G "(0,)" -P s1 -M hrnetocr -R -1 train 0819camonly

# 测试模型
./run.sh -G 0 -P s1 eval -c config/eval_s4.json
```

## Deeplab

### ~~conda安装命令~~

```bash
conda create -n teeth python=3.7
conda activate teeth
conda install numpy pandas matplotlib tensorflow-gpu=1.14.0
conda install -c conda-forge python-docx pillow opencv tqdm prettytable scikit-learn
```

## HRNet

### ~~运行命令~~

```bash
# 激活环境
conda activate hrnet

# 训练数据
python hrnet/tools/train.py --cfg config/hrnet.yaml
```

### ~~PyTorch多版本适配~~

```bash
# For PyTorch 0.4.1
PY_CMD="python"
# For PyTorch 1.1.0
PY_CMD="python -m torch.distributed.launch --nproc_per_node=4"
```
