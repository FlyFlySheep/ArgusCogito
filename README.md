# ArgusCogito: Chain-of-Thought for Cross-Modal Synergy and Omnidirectional Reasoning in Camouflaged Object Segmentation

## 📂 项目结构
```
├── main.py                 # 主程序入口
├── config/                 # Hydra 配置文件目录
│   └── config.yaml         # 主配置文件
├── src/                    
│   ├── argfocus.py         # ArgusCogito_focus (bbox生成)
│   ├── knowledge.py        # Knowledge factory (知识提取/生成)
│   ├── segment.py          # Sam4Segment (分割模块)
│   ├── prompt.py           # Prompt (提示词)
│   ├── qwen_chat.py        # Qwen_chat（与大模型交互）
│   ├── arglog.py           # Arglog（日志）
└── requirements.txt        # 依赖库
```

---

## ⚙️ 环境依赖
- Python >= 3.9  
- PyTorch >= 2.0 (支持 GPU)  
- 依赖库：
  ```bash
  pip install -r requirements.txt
  ```
  主要依赖：
  - transformers  
  - peft  
  - Pillow  
  - hydra-core  

---

## 🚀 使用方法

### 1. 准备输入数据
- 将 **RGB 图像** 和 **Depth 图像** 放入 `config.yaml` 指定的路径  
- 支持单张图像或批量处理  

### 2. 修改配置文件
在 `config/config.yaml` 中设置：
```yaml
image: null  # image to process only

# If you specify an image, only this image will be inferred (only basename, no extention)

use_knowledge: True
use_knowledge_origin: False # use the knowledge already have
use_bboxes_origin: False

use_sam4mllm: True # use sam4mllm to provide point hints
save_knowlwdge: True
save_bboxes: True

focus_type: left_middle_right

target_long_edge: 1000
max_new_tokens: 1024

input:
  knowledge_origin: # The folder for knowledge of the images
  RGB_image: # dataset rgb image
  Depth_image: # dataset depth image

output:
  log_file: # .log file for logging
  save_file: # folder for saving visible images and masks
  save_knowledge: # folder for saving knowledge
  save_bboxes: # folder for saving bboxes

model:
  qwen2_vl_path: # model path
  sam2_ckpt: facebook/sam2.1-hiera-large # dafault path for sam2
  llava_ckpt: # model path
  sam4mllm_ckpt: /disk4/tan/SAM4MLLM-main/checkpoint/sam4mllm


rounds:
  - id: 1
    words: ['animal or human']
  - id: 2
    words: ['animal or human']


rgb_image_extention: .jpg
depth_image_extention: .png
```

### 3. 运行
```bash
python main.py
```

支持单张图像处理：
```bash
python main.py image=example_001
```

