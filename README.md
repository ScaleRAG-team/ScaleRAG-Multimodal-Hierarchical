# ScaleRAG: Multimodal Hierarchical RAG for Scientific Papers

## Team Information
- **Team Name**: ScaleRAG
- **Members**:
  - Mahdi Saleh Tabesh (mt3846, Columbia University)
  - Manush Kalwari (mmk2266, Columbia University)

---

## 1. Problem Statement

Retrieval-Augmented Generation (RAG) systems struggle to scale to large scientific corpora while preserving document structure and multimodal evidence such as figures and tables. The effectiveness of RAG over large scientific corpora is constrained by limited retrieval recall, increasing latency, and the growing multimodality of research papers, making it challenging to scale retrieval while preserving relevance and cost efficiency. This project investigates scalable, compute-efficient RAG pipelines that maintain retrieval quality, grounding, and predictable latency when applied to long, structured scientific papers.

---

## 2. Model Description

We implement and evaluate two complementary RAG architectures under a unified framework:

### A. Multimodal Hierarchical RAG (RAPTOR-based)
- **Retrieval**: Three-level hierarchical retrieval (paper → section → chunk)
- **Embeddings**:
  - Text: MiniLM (384-D)
  - Vision: CLIP vision encoder (concatenated with caption embeddings)
- **Summarization**: RAPTOR-style recursive abstractive summarization (offline)
- **Indexing**: FAISS cosine similarity search
- **Generation**: LLaVA-1.5-7B (4-bit quantized)
- **Frameworks**: PyTorch, FAISS, SentenceTransformers, Docling

### B. Unimodal Hierarchical RAG (Text-only with Visual Distillation)
- **Retrieval**: Coarse-to-fine hierarchical RAG with SPI-Lite–style efficiency
- **Embeddings**: GTE-Large
- **Visual Handling**: Figures and tables summarized offline into text using GPT-based models
- **Generation**: Phi-3.5-mini-instruct
- **Frameworks**: PyTorch, FAISS, SentenceTransformers, Docling

No end-to-end training is performed; all models are used in inference-only mode with offline preprocessing.

---

## 3. Final Results Summary

| Metric                          | Value (ScaleRAG)            |
|---------------------------------|-----------------------------|
| Recall@5 (1023 papers)          | ~75%                        |
| nDCG@5                          | ~0.60                       |
| Grounding Accuracy              | 80–83%                      |
| Avg. Retrieval Latency          | ~150 ms                     |
| Context Length Stability        | Stable across corpus sizes  |
| Device                          | Single GPU (T4)      |

Hierarchical retrieval maintains high recall and stable latency as corpus size scales, significantly outperforming flat baselines in large collections.

---

## 4. Reproducibility Instructions

### A. Requirements

Install dependencies:
```bash
pip install -r requirements.txt 
```

Key dependencies include:
- PyTorch
- FAISS
- SentenceTransformers
- Docling
- Transformers
- NumPy / Pandas

---

### B. WandB Dashboard

Evaluation and profiling were performed locally.

---

### C. Training vs Inference

This project is **inference-only**. All models are pre-trained and used without fine-tuning.

---

### D. Evaluation

To run evaluation notebooks:
```bash
jupyter notebook rag_v1/
jupyter notebook rag_v2/
```

Metrics reported include Recall@k, nDCG@k, retrieval latency, and grounding accuracy.

---

### E. Quickstart: Minimum Reproducible Result

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Prepare data
jupyter notebook RAG_Data_Preparation.ipynb

# Step 3: Run hierarchical RAG
jupyter notebook rag_v2/
```

---

## 5. Notes

- Data processing relies on **Docling** for robust PDF parsing.
- Hierarchical summaries and embeddings are computed once and cached.
- GPT-based APIs are used offline to generate high-quality, image-aware textual summaries of figures and tables.
- Experiments were designed to reflect realistic single-GPU constraints.
- See `Project_Report.pdf` for full methodology, results, and analysis.

---

## Repository Structure

```
ScaleRAG-Multimodal-Hierarchical/
├── 00-frontend/        # Web interface
├── 01-backend/         # API and serving logic
├── data/               # PDFs, parsed JSON, embeddings
├── rag_v1/             # Baseline & multimodal RAG
├── rag_v2/             # Hierarchical ScaleRAG
├── utils/              # Helper scripts
├── RAG_Data_Preparation.ipynb # Data Preparation 
├── Project_Report.pdf  # Full technical report
└── README.md
```

---

## Citation

```
@project{ScaleRAG2025,
  title     = {ScaleRAG: Multimodal Hierarchical RAG for Scientific Papers},
  author    = {Mahdi Saleh Tabesh and Manush Kalwari},
  year      = {2025},
  institution = {Columbia University}
}
```
