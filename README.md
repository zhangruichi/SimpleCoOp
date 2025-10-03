# SimpleCoOp

一个 [CoOp: Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134) **极简复现**。  
不依赖复杂框架，没有臃肿封装，开箱即用。
---

## 💡 理念
- **如无必要，勿加实体**  
- **删繁就简**：保持核心逻辑直白清晰  
- **最小单元**：仅由 **main / model / data** 三部分构成  

---

## 📂 代码结构
├── CoOp.py           # CoOp模型定义

├── datasets.py       # 数据集封装 (StanfordCars)

├── main.py           # 程序入口

每个文件都支持单独调用，方便快速测试和调试。  
---

## ⚡ 快速开始

### 1. 安装依赖
```bash
pip install torch torchvision   # 请根据设备选择合适版本
pip install numpy Pillow tqdm
pip install git+https://github.com/openai/CLIP.git
```


### 2. 运行

```bash
python main.py \
  --data_root /path/to/stanford_cars \
  --gpu 0 \
  --epochs 200 \
  --num_shots 16
```
常用参数（可在命令行传入，也可以直接修改 main.py 中的默认值）


## 📦 数据集准备

StanfordCars 数据集需提前下载，并确保目录下有 `split_zhou_StanfordCars.json`

## 🙌 致谢

* [CoOp 原论文](https://arxiv.org/abs/2109.01134)
* [OpenAI CLIP](https://github.com/openai/CLIP)


## ⭐ Star 本项目

如果这个项目对你有帮助，欢迎点个 **Star** ⭐

如有问题 或 额外需要 请提出 **issue**  

欢迎各位进行交流学习，共同进步！