# Qilin Baselines

A comprehensive collection of state-of-the-art baseline model implementations for the Qilin multimodal search, recommendation and Retrieval-Augmented Generation (RAG)dataset. This repository provides researchers and practitioners with robust benchmarks for evaluating and developing multimodal search and recommendation systems.

## 📢 News

**[2024-03-18]** 🔥Image resources are now available for download through [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/af72ab5dbba1460da6c0/)! 

## Overview

Qilin is an extensive dataset specifically designed for advancing research in multimodal search and recommendation tasks. It encompasses:
- Multiple content modalities (text, images, video thumbnails)
- Rich user interaction data
- Comprehensive evaluation metrics
- Support for RAG system development and benchmarking

## Baseline Models

We provide several carefully implemented baseline models, each targeting different aspects of multimodal search and recommendation:

### Traditional Information Retrieval
- **BM25**
  - Implements classic lexical matching algorithms
  - Serves as a strong baseline for text-based retrieval
  - Optimized for efficient search

### Neural Retrieval Models
- **Bi-Encoder**
  - BERT-based architecture for efficient retrieval
  - Independent encoding of queries and notes
  - Optimized for retrieval with dense representations

- **Cross-Encoder**
  - BERT-based architecture for precise reranking
  - Joint encoding of query-note pairs
  - Higher accuracy but computationally more intensive

### Advanced Neural Models
- **DCN (Deep & Cross Network)**
  - Sophisticated feature interaction modeling
  - Combines both dense and sparse features
  - Specialized in user interaction pattern analysis

- **VLM (Vision-Language Model)**
  - State-of-the-art multimodal understanding
  - Seamless integration of text and visual content
  - Enhanced content representation capabilities
  - Support for cross-modal retrieval tasks

## Setup Instructions

### Environment Requirements
- Python 3.10.16
- CUDA-compatible GPU (recommended for neural models)

### Data and Model Preparation
1. Dataset Setup:
   - Download the Qilin dataset from [Hugging Face](https://huggingface.co/THUIR)
   - Extract and place in the `../datasets` directory
   - Verify the dataset structure matches the documentation

2. Model Setup:
   - Download the following models from Hugging Face:
     - [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct): Base vision-language model
     - [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct): Enhanced vision-language model
     - [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese): Foundation model for Chinese text
   - Place all models in the `../model` directory

### Installation
```bash
pip install -r requirements.txt
```

## Dataset Details

### DQA
- Total samples: 6,972
- Features:
  - Structured query information
  - User click signals
  - Comprehensive retrieval results
  - Ground truth annotations

### Search Dataset
- Training set: 44,024 samples
- Testing set: 6,192 samples
- Key components:
  - Rich query metadata
  - User interaction logs
  - Pre-computed neural embeddings
  - Ground clicked labels

### Recommendation Dataset
- Training set: 83,437 samples
- Testing set: 11,115 samples
- Includes:
  - Detailed user interaction history
  - Candidate note pools
  - Contextual features
  - Ground clicked labels

### User Features
- User profiles: 15,482
- Feature types:
  - Demographic information
  - 40-dimensional dense feature vectors
  - Behavioral patterns
  - Preference indicators

### Notes (Content Items)
- Total items: 1,983,938
- Content features:
  - Text content:
    - Titles and descriptions
    - Body text with formatting
  - Media information:
    - Image
  - Engagement metrics:
    - View counts
    - Like/comment statistics
    - User interaction data
  - Pre-computed embeddings:
    - Text embeddings

## Usage Guide

### Training Models
```bash
sh scripts/run.sh --config config/xxx.yaml
```

Replace `xxx.yaml` with the specific configuration file for your chosen model. Here's a comprehensive list of available configuration files:

#### Search Task Configurations
- `search_dpr_config.yaml`: Configuration for Bi-Encoder model
  - Dense passage retrieval architecture
  - Optimized for retrieval
  - Efficient query-note matching

- `search_cross_encoder_config.yaml`: Configuration for Cross-Encoder model
  - Fine-grained relevance scoring
  - BERT-based query-note interaction
  - Suitable for reranking

- `search_dcn_config.yaml`: Configuration for Deep & Cross Network
  - Feature crossing at different levels
  - Combines memorization and generalization

- `search_vlm_config.yaml`: Configuration for Vision-Language Model
  - Multimodal search capabilities
  - Joint text-image understanding
  - Cross-modal retrieval optimization

#### Recommendation Task Configurations
- `recommendation_dpr_config.yaml`: Configuration for Bi-Encoder
  - User-note representation learning
  - Scalable retrieval architecture
  - Efficient candidate generation

- `recommendation_cross_encoder_config.yaml`: Configuration for Cross-Encoder
  - Deep user-note interaction modeling
  - Precise relevance estimation

- `recommendation_dcn_config.yaml`: Configuration for Deep & Cross Network
  - Feature crossing at different levels
  - Combines memorization and generalization

- `recommendation_vlm_config.yaml`: Configuration for Vision-Language Model (2B)
  - Multimodal content understanding
  - Lightweight training option

- `recommendation_vlm7B_config.yaml`: Configuration for Enhanced VLM (7B)
  - Higher capacity model
  - Superior performance

### Monitoring Training
```bash
sh scripts/tensorboard.sh
```
Access TensorBoard visualizations to monitor:
- Training loss curves
- Evaluation metrics
- Model performance statistics

## Development Resources

### Toy Dataset
For quick experimentation and format familiarization, we provide a toy dataset in CSV format located at `../datasets/toy_data`. This smaller dataset maintains the same structure as the full dataset while being more manageable for initial development and testing.
