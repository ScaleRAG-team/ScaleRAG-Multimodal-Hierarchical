
# Multimodal RAG Evaluation Report
Generated on: 2025-11-17 18:31:11

## Model & Modes
- **Model:** LLaVA 1.5 - 7B (4-bit quantized)
- **Generation Mode:** Top-p Sampling (Option B)

## Generation Hyperparameters
- **max_new_tokens**: 480
- **min_new_tokens**: 120
- **do_sample**: True
- **temperature**: 0.4
- **top_p**: 0.92
- **repetition_penalty**: 1.1
- **eos_token_id**: 0
- **pad_token_id**: 0
- **no_repeat_ngram_size**: 3

## Retrieval Settings
- Top-k Text Retrieval: 3
- Top-k Image Retrieval: 2
- Text Encoder: sentence-transformers/all-MiniLM-L6-v2
- Image Encoder: CLIP ViT-B/32
- FAISS Index Type: Flat (L2-normalized)

---

### 1. LLM in a Flash (ACL'24)
**Q:** How does the system enable long-context inference on memory-limited devices using NVMe, and what are the key scheduling/prefetching ideas?

**A:** In the paper "LLM in a flash" presented at ACL'23, the authors propose a method to enable long context inference on limited memory devices using an NVME storage device. They develop a novel cost model based on the characteristics of flash and D RAM, which allows them to optimize the use of both storage media. The key ideas behind the scheduling and prefetching include windowing and row-column bundle optimization. These techniques help reduce the time spent on data loading and improve overall inference efficiency. The authors showcase the effectiveness of their proposed methods by demonstrating improvements in inference performance when compared to conventional loading methods on CPU and GPU.

_Generated in 10.24s_

---
### 2. FlashAttention-2 (2023)
**Q:** What kernel-level changes make FA-2 faster than FA-1, and how do tiling/parallelization reduce memory traffic?

**A:** Flash Attention-1 and Flash Attension-2 have different kernel-based optimizations that improve their performance. The authors of the paper implemented various optimizations in Flash Attentions, including better parallelization and work distribution. They also utilized tiling and parallelization techniques to reduce memory bandwidth usage. These optimizations allowed them to achieve higher speedups compared to the original Flash Attenion. By carefully examining the code and understanding the underlying mechanisms, they were able to identify areas where they could improve the efficiency of the algorithms, leading to increased performance in both forward and backward passes.

_Generated in 9.55s_

---
### 3. vLLM: PagedAttention (2023)
**Q:** Explain how PagedAttention reduces memory fragmentation and improves throughput during batched decoding.

**A:** Paper: "vLLM" - Page 2, Section "Introduction":
The authors introduce "vLTM," a distributed language model serving engine designed to improve the efficiency of large language models. One key aspect of vLTM is its use of "PagedAttentions," which operate on non-consecutive memory pages similar to how operating systems manage virtual memory. This helps reduce memory fragmentaion and improve throughput when performing batched decode operations. By utilizing these techniques, vLTN is able to serve larger language models while maintaining high levels of accuracy.

_Generated in 10.15s_

---
### 4. Speculative Decoding (Leviathan et al., 2023)
**Q:** Summarize the draft-verify process and the acceptance criterion; what governs speedup vs. quality?

**A:** In the paper "Speculative decoding" by Leviathan, et al. (2019), the draft verification process involves using a smaller language model to generate candidate sentences based on the input sentence. The acceptance criteria involve evaluating the quality and diversity of the generated sentences compared to the original ones. The goal is to strike a balance between generating diverse sentences that align well with the original distribution and minimizing the overall error rate. This is achieved through a combination of factors such as the size and architecture of the language model used, the temperature of the sampling process, and the specifications of the acceptability metric. The trade-off between speedup and quality is governed by these factors, and it requires careful tuning to optimize the performance of the system.

_Generated in 11.90s_

---
### 5. H2O: Efficient LLM Inference (2023)
**Q:** Why is KV-cache eviction in H2O not equivalent to Belady's cache policy, and what heuristics does H2O use?

**A:** The KV caching mechanism used in H ² O is different from Belady’s cache policies. While Belady proposed a cache-based approach to reduce memory usage in deep learning models, H20 employs a more efficient technique called live migration of LLC requests. This involves using the KLV (Knowledge Level Value) embeddingen of the last token as a reference point to ensure that the KC (KV Cache) remains consistent throughout the entire inference process. As a result, H²O can maintain high-quality performance without incurring the same memory overheads as traditional cache-policies.

_Generated in 9.70s_
