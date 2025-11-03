# ScaleRAG – Multimodal Hierarchical RAG for Scientific Papers

**Team:**  
- Mahdi Saleh Tabesh (mt3846)  
- Manush Kalwari (mmk2266)  

**Course:** Scaling LLMs: Systems, Optimization, and Emerging Paradigms - COMSE6998 (Columbia University, Fall 2025)

---

## Overview

**ScaleRAG** is a reproducible framework for **multimodal, hierarchical Retrieval-Augmented Generation (RAG)** over scientific papers—especially those focused on **LLM scaling laws**.  
The system processes text, figures, tables, and equations from research PDFs into structured, searchable datasets for grounded question answering and reasoning.

Our goal is to enable **evidence-grounded, multimodal retrieval** and efficient **inference serving** using modern large language models.

---

## Current Progress (Milestone 1 – Retrieval Pipeline)

The repository currently contains two main notebooks that together form the **data-to-retrieval backbone**:

### 1. `RAG_Data_Preparation.ipynb`
Transforms raw arXiv PDFs into structured multimodal datasets:
- Downloads papers listed in `core_papers.csv`
- Converts PDFs to structured **Docling JSON** format
- Extracts and enriches **paragraphs, tables, figures, and equations**
- Attaches image crops to RAG blocks
- Produces clean merged chunks in `data/rag_chunks/`

**Output directories:**

```
data/pdf/ → Raw PDFs  
data/docling_json*/ → Structured outputs  
data/rag_json*/ → RAG-ready JSON blocks  
data/rag_assets/images/ → Extracted figures & tables  
data/rag_chunks/ → Final merged chunks  
```



### 2. `RAG_v1.ipynb`
Implements the **first complete multimodal retrieval pipeline**:
- Loads and merges preprocessed chunks
- Generates text (384-D) and multimodal (896-D) embeddings  
  using **SentenceTransformer (MiniLM)** and **OpenAI CLIP**
- Builds and saves **FAISS** indexes for fast cosine similarity search
- Defines **query & retrieval functions** for text and image search
- Integrates **RAG context builder** and GPU-based **Gemma/LLaVA** inference setup

**Output directories:**
```
data/RAG/embeddings/ → Stored embeddings (.json / .pkl)
data/RAG/indexes/ → FAISS indexes for text and images
```

---

## Upcoming Work

- [ ] Extend RAG to hierarchical retrieval (RAPTOR-style summarization)
- [ ] Implement document-level reasoning chains for multi-hop QA
- [ ] Integrate vLLM serving for scalable inference benchmarking
- [ ] Evaluate accuracy, grounding, latency, and cost trade-offs
- [ ] Prepare final report and demo notebook

---

## Core Idea

Traditional RAG pipelines flatten documents into plain text, losing visual and structural cues crucial for **technical papers**.  
**ScaleRAG** preserves both structure and modality — allowing large language models to retrieve and reason across **figures, equations, and hierarchical sections**, not just text.

---

## References

Key works inspiring this project:
- Lewis et al. (2020) – [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401)  
- Sarthi et al. (2024) – [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2404.01744)  
- Microsoft Research (2024) – [GraphRAG: Knowledge-Graph-Guided Retrieval](https://aka.ms/graphrag)  
- Yan et al. (2024) – [Corrective Retrieval-Augmented Generation (CRAG)](https://arxiv.org/abs/2403.05989)  
- Jiang et al. (2023) – [LLMLingua / RECOMP](https://arxiv.org/abs/2310.06839)  
- Kwon et al. (2023) – [vLLM: PagedAttention for Efficient LLM Inference](https://arxiv.org/abs/2309.06180)

---

## Environment

Developed on a **GCP instance (T4 / A100 GPU)** with:
- Python 3.10  
- PyTorch 2.x  
- FAISS  
- SentenceTransformers  
- OpenAI CLIP  
- Docling  
- Pandas / NumPy / Matplotlib

---

## Repository Structure

```
ScaleRAG-Multimodal-Hierarchical/
│
├── RAG_Data_Preparation.ipynb   # PDF → Structured multimodal data
├── RAG_v1.ipynb                 # Embedding + FAISS retrieval pipeline
├── data/                        # PDFs, JSONs, embeddings, indexes
├── utils/                       # Helper modules (download, parsing, etc.)
└── README.md
```


## Citation

If you use or extend this work, please cite:

```
@project{ScaleRAG2025,
  author       = {Mahdi Tabesh and Manush Kalwari},
  title        = {ScaleRAG: Multimodal Hierarchical RAG for Scientific Papers},
  year         = {2025},
  institution  = {Columbia University},
  course       = {Scaling Large Language Models}
}
```

---

_This README will be updated as the project progresses._

