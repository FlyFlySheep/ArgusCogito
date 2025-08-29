# ArgusCogito: Chain-of-Thought for Cross-Modal Synergy and Omnidirectional Reasoning in Camouflaged Object Segmentation

## 📂 Project Structure
```
├── main.py                 # Main entry point
├── config/                 # Hydra configuration directory
│   └── config.yaml         # Main configuration file
├── src/                    
│   ├── argfocus.py         # ArgusCogito_focus (bbox generation)
│   ├── knowledge.py        # Knowledge factory (knowledge extraction/generation)
│   ├── segment.py          # Sam4Segment (segmentation module)
└── requirements.txt        # Dependencies
```

---

## ⚙️ Environment Requirements
- Python >= 3.9  
- PyTorch >= 2.0 (GPU supported)  
- Dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  Key dependencies:
  - transformers  
  - peft  
  - Pillow  
  - hydra-core  

---

## 🚀 Usage

### 1. Prepare Input Data
- Place **RGB images** and **Depth images** in the directories specified in `config.yaml`.  
- Supports both single-image and batch processing.  

### 2. Modify Configuration File
Edit `config/config.yaml` to set parameters:
```yaml
image: null  # image to process only

# If you specify an image, only this image will be inferred (only basename, no extension)

use_knowledge: True
use_knowledge_origin: False # use pre-existing knowledge
use_bboxes_origin: False

use_sam4mllm: True # use sam4mllm to provide point hints
save_knowlwdge: True
save_bboxes: True

focus_type: left_middle_right

target_long_edge: 1000
max_new_tokens: 1024

input:
  knowledge_origin: # Folder containing prior knowledge files
  RGB_image: # Dataset RGB image folder
  Depth_image: # Dataset depth image folder

output:
  log_file: # Path for log file
  save_file: # Folder for saving visualization images and masks
  save_knowledge: # Folder for saving knowledge files
  save_bboxes: # Folder for saving bboxes

model:
  qwen2_vl_path: # Model path
  sam2_ckpt: facebook/sam2.1-hiera-large # Default SAM2 checkpoint
  llava_ckpt: # Model path
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

Run with a single image:
```bash
python main.py image=example_001
```
