
# Comparative Analysis of Graph Neural Network and Sequence Models for News Summarization

## 📌 Overview

This project presents a comparative study of four neural architectures for abstractive text summarization using the CNN/DailyMail dataset. The architectures include:

* **TRANSFORMER** (BART)
* **BILSTM**
* **GNN + TRANSFORMER**
* **BILSTM + GNN**

The core goal is to analyze how integrating Graph Neural Networks (GNNs) with traditional sequence models enhances summarization performance by capturing structural and long-range dependencies in documents.

## 🧠 Motivation

Sequence models, including transformers and BiLSTMs, often overlook non-sequential relationships within a document. Graph Neural Networks, by contrast, allow for modeling heterogeneous relationships between words, sentences, and entities, offering the potential to improve summary coherence and informativeness.

## 📂 Dataset

* **Source**: [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)
* **Subset Used**: 0.1% of the dataset (due to compute limitations)
* **Structure**: Pairs of news articles and corresponding human-written summaries

## ⚙️ Preprocessing

* Text cleaning (removing special characters, normalization, tokenization)
* Sentence segmentation and word tokenization using NLTK
* Embedding with GloVe and BERT
* Graph construction for GNN-based models:

  * Word nodes and sentence nodes
  * Edges based on word-sentence occurrence and sentence-sentence similarity (cosine similarity of BERT embeddings)

## 🏗️ Architectures

1. **Transformer (BART)**

   * Fine-tuned on raw article-summary pairs.

2. **BiLSTM**

   * Seq2Seq model with attention-based decoding.

3. **GNN + Transformer**

   * Graph-enhanced encoding followed by BART fine-tuning on GNN-generated summaries.

4. **BiLSTM + GNN**

   * GNN-based encoding followed by training a BiLSTM-based summarization model.

## 🧪 Experimental Setup

* Framework: PyTorch, PyTorch Geometric
* Embeddings: GloVe (300D), BERT-base (768D)
* Hyperparameters:

  * Batch size: 1
  * Epochs: 3
  * Optimizer: Adam (LR = 0.0001)
  * GNN: 3 layers, 8 attention heads, hidden size = 256
  * BiLSTM: 3 layers, hidden size = 512

## 📊 Evaluation

Evaluated using ROUGE metrics:

* **ROUGE-1** (Unigram)
* **ROUGE-2** (Bigram)
* **ROUGE-L** (Longest Common Subsequence)

### Results Summary:

| Model             | ROUGE-1  | ROUGE-2   | ROUGE-L   |
| ----------------- | -------- | --------- | --------- |
| Transformer       | 38.2     | 17.5      | 27.0      |
| BiLSTM            | 17.12    | 16.24     | 23.03     |
| GNN + Transformer | **45.2** | **23.14** | **31.17** |
| BiLSTM + GNN      | 25.56    | 21.23     | 26.14     |

## 💡 Key Insights

* GNN-enhanced models outperform their standalone counterparts across all ROUGE metrics.
* GNNs help capture structural relationships and long-range dependencies that improve the quality of generated summaries.
* The **GNN + Transformer** model offers the most significant improvement.

## 📁 Folder Structure

```
.
├── data/                   # Preprocessed dataset
├── models/                 # Model definitions
├── results/                # Output summaries and metrics
├── notebooks/              # Training and evaluation scripts
└── README.md               # Project overview
```

## 📚 References

* [BART](https://arxiv.org/abs/1910.13461)
* [GloVe](https://nlp.stanford.edu/projects/glove/)
* [CNN/DailyMail Dataset](https://huggingface.co/datasets/cnn_dailymail)
* Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries


