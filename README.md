
<div align="center">
<img src="pics/merged.png" alt="Qilin Logo" width="400"/>
</div>

# Qilin

Qilin is a large-scale multimodal dataset designed for advancing research in search, recommendation, and Retrieval-Augmented Generation (RAG) systems. This repository contains the official implementation of the dataset paper, baseline models, and evaluation tools.

## Dataset Overview

Qilin provides comprehensive data for three main scenarios:

### Search Dataset
- Training set: 44,024 samples
- Testing set: 6,192 samples
- Features:
  - Rich query metadata
  - User interaction logs
  - Ground clicked labels

### Recommendation Dataset
- Training set: 83,437 samples
- Testing set: 11,115 samples
- Features:
  - Detailed user interaction history
  - Candidate note pools
  - Contextual features
  - Ground clicked labels

### Key Characteristics
- Multiple content modalities (text, images, video thumbnails)
- Rich user interaction data
- Comprehensive evaluation metrics
- Support for RAG system development

## Repository Structure

- `baselines/`: Implementation of state-of-the-art baseline models
- `datasets/`: Dataset files and processing scripts
  - `toy_data/`: Small sample dataset for quick exploration
  - `qilin/`: Complete dataset (after downloading)

## Getting Started

### Installation

```bash
pip install -r baselines/requirements.txt
```

### Data and Model Preparation

1. Download the Qilin dataset from [Hugging Face](https://huggingface.co/datasets/THUIR/qilin)
2. Extract and place the dataset in the `datasets/qilin` directory
3. Download the required models:
   - [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
   - [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
   - [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)
4. Place the downloaded models in the `model` directory

## Citation

If you use Qilin in your research, please cite our paper:

```
@misc{chen2025qilinmultimodalinformationretrieval,
      title={Qilin: A Multimodal Information Retrieval Dataset with APP-level User Sessions}, 
      author={Jia Chen and Qian Dong and Haitao Li and Xiaohui He and Yan Gao and Shaosheng Cao and Yi Wu and Ping Yang and Chen Xu and Yao Hu and Qingyao Ai and Yiqun Liu},
      year={2025},
      eprint={2503.00501},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2503.00501}, 
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
