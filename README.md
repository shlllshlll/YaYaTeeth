# YaYaTeeth

## 一些注释

### conda导出导入环境

```bash
# 导出
# Linux
conda env export --no-builds | grep -v "prefix" > environment.yml
# Windows
conda env export --no-builds | findstr -v "prefix" > environment.yml

# 导入
conda create -n env_name -f enviroment.yml
```

## Deeplab

### conda安装命令

```bash
conda create -n teeth python=3.7
conda activate teeth
conda install numpy pandas matplotlib tensorflow-gpu=1.14.0
conda install -c conda-forge python-docx pillow opencv tqdm prettytable scikit-learn
```

## HRNet

### 运行命令

```bash
# 激活环境
conda activate hrnet

# 训练数据
python hrnet/tools/train.py --cfg config/hrnet.yaml
```
