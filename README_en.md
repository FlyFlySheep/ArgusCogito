# ArgusCogito: Chain-of-Thought for Cross-Modal Synergy and Omnidirectional Reasoning in Camouflaged Object Segmentation  

## ⚙️ Environment Requirements  
- Python >= 3.9  
- PyTorch >= 2.0 (with GPU support)  
- Dependencies:  
  ```bash
  pip install -r requirements.txt
  ```  

  Main dependencies:  
  - transformers  
  - peft  
  - Pillow  
  - hydra-core  

---

## 🚀 Usage  

### 1. Prepare Input Data  
- Place **RGB images** and **Depth images** in the path specified in `config.yaml`  
- Supports both single-image and batch processing  

### 2. Modify Configuration File  
Edit `config/config.yaml`:  
```yaml
image: null  # image to process only  
             # If you specify an image, only this image will be inferred (only basename, no extension)

use_knowledge: True
use_knowledge_origin: False # use the existing knowledge
use_bboxes_origin: False

use_sam4mllm: True # use sam4mllm to provide point hints
save_knowlwdge: True
save_bboxes: True

focus_type: left_middle_right
target_long_edge: 1000
max_new_tokens: 1024

input:
  knowledge_origin: # Folder for image knowledge
  RGB_image:        # Dataset RGB images
  Depth_image:      # Dataset depth images

output:
  log_file:         # .log file for logging
  save_file:        # Folder for saving visualizations and masks
  save_knowledge:   # Folder for saving knowledge
  save_bboxes:      # Folder for saving bounding boxes

model:
  qwen2_vl_path:    # model path
  sam2_ckpt: facebook/sam2.1-hiera-large # default path for sam2
  llava_ckpt:       # model path
  sam4mllm_ckpt: /disk4/tan/SAM4MLLM-main/checkpoint/sam4mllm

rounds:
  - id: 1
    words: ['animal or human']
  - id: 2
    words: ['animal or human']

rgb_image_extention: .jpg
depth_image_extention: .png
```  

### 3. Run  
```bash
python main.py
```  

For single-image processing:  
```bash
python main.py image=example_001
```  

---

## 🔗 Models & Download Links  
- `llava-next`: [llava-next](https://huggingface.co/lmms-lab/llama3-llava-next-8b)  
- `sam4mllm+`: [sam4mllm+](https://drive.google.com/drive/folders/1ytEfGRa6bxThTXQn5MLVKKy4jsxxBo6M)  
- `Qwen2.5-VL-7B-Instruct`: [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)  
- `sam2.1-hiera-large`: Directly pulled by using command `facebook/sam2.1-hiera-large`  

---

## 📖 Citation  
This project builds upon the following work. If you use this project in your research, please also cite the original paper:  
```bibtex
@inproceedings{chen2024sam4mllm,
  title={Sam4mllm: Enhance multi-modal large language model for referring expression segmentation},
  author={Chen, Yi-Chia and Li, Wei-Hua and Sun, Cheng and Wang, Yu-Chiang Frank and Chen, Chu-Song},
  booktitle={European Conference on Computer Vision},
  pages={323--340},
  year={2024},
  organization={Springer}
}
```  
