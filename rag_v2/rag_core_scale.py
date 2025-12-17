
# rag-core-scale.py

import time
import re
from pathlib import Path

import faiss
import numpy as np
import ujson as json
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


class ScaleRAGPipeline:
    def __init__(
        self,
        index_l1_path,
        emb_l1_path,
        meta_l1_path,
        meta_l2_path,
        meta_l3_path,
        l2_manifest_path,
        l3_manifest_path,
        max_new_tokens: int = 512,
        generator_model_id="microsoft/Phi-3.5-mini-instruct",
        device="gpu",
    ):
        self.max_new_tokens = max_new_tokens
        # --------------------
        # Load metadata
        # --------------------
        self.level1 = json.load(open(meta_l1_path))
        self.level2 = json.load(open(meta_l2_path))
        self.level3 = json.load(open(meta_l3_path))

        # --------------------
        # Load embeddings + FAISS
        # --------------------
        self.emb_l1 = np.load(emb_l1_path)
        self.idx_l1 = faiss.read_index(str(index_l1_path))

        self.l2_manifest = json.load(open(l2_manifest_path))
        self.l3_manifest = json.load(open(l3_manifest_path))

        # --------------------
        # Index caches
        # --------------------
        self.l2_index_cache = {}
        self.l3_index_cache = {}

        # --------------------
        # Encoder
        # --------------------
        self.encoder = SentenceTransformer(
            "Alibaba-NLP/gte-large-en-v1.5",
            device="cuda",
            trust_remote_code=True,
        )

        # --------------------
        # Generator
        # --------------------
        self.tok = AutoTokenizer.from_pretrained(generator_model_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            generator_model_id,
            device_map="cuda",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        self.SYSTEM = (
            "Answer ONLY from <chunk> context. "
            "Read EVERY chunk.\n"
            "Step 1: Extract EVERY distinct method/model by its PROPER NAME.\n"
            "Step 2: Write a concise answer.\n"
            "If nothing relevant: Not found.\n"
        )

        self.gen_cfg = dict(
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.01,
            no_repeat_ngram_size=8,
        )

        # --------------------
        # Retrieval hyperparams (internal)
        # --------------------
        self.k_max = {"l1": 10, "l2": 20, "l3": 40}
        self.tau = {"l1": 0.65, "l2": 0.60, "l3": 0.55}

        self.config = {
            "encoder_model": "Alibaba-NLP/gte-large-en-v1.5",
            "generator_model": generator_model_id,
            "adaptive_retrieval": True,
            "similarity_thresholds": self.tau,
            "k_max": self.k_max,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_l2_index(self, paper_id):
        if paper_id not in self.l2_index_cache:
            entry = self.l2_manifest.get(paper_id)
            if entry is None:
                return None
            self.l2_index_cache[paper_id] = faiss.read_index(entry["index_path"])
        return self.l2_index_cache[paper_id]

    def _get_l3_index(self, paper_id, section_id):
        key = f"{paper_id}__{section_id}"
        if key not in self.l3_index_cache:
            entry = self.l3_manifest.get(key)
            if entry is None:
                return None
            self.l3_index_cache[key] = faiss.read_index(entry["index_path"])
        return self.l3_index_cache[key]

    def _choose_depth(self, query):
        q = query.lower()
        if re.search(r"(figure|eqn|equation|derive|algorithm)", q):
            return 3
        words = len(query.split())
        if words <= 6:
            return 1
        if words <= 15:
            return 2
        return 3

    def _embed_query(self, query):
        return self.encoder.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

    # ------------------------------------------------------------------
    # Adaptive SPI Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query):
        timing = {}
        t0 = time.perf_counter()

        depth = self._choose_depth(query)
        qemb = self._embed_query(query)
        timing["embed_query_ms"] = (time.perf_counter() - t0) * 1000

        # ---------- Level 1 ----------
        t = time.perf_counter()
        D1, I1 = self.idx_l1.search(qemb, self.k_max["l1"])

        papers = []
        for score, idx in zip(D1[0], I1[0]):
            if score < self.tau["l1"]:
                break
            papers.append(self.level1[idx])

        timing["l1_search_ms"] = (time.perf_counter() - t) * 1000

        if depth == 1 or not papers:
            timing["retrieval_ms"] = sum(timing.values())
            return {"level": 1, "papers": papers}, timing

        # ---------- Level 2 ----------
        t = time.perf_counter()
        sections = []
        for p in papers:
            idx = self._get_l2_index(p["paper_id"])
            if idx is None:
                continue

            D2, I2 = idx.search(qemb, self.k_max["l2"])
            meta_idxs = self.l2_manifest[p["paper_id"]]["meta_indices"]

            for score, i in zip(D2[0], I2[0]):
                if score < self.tau["l2"]:
                    break
                sections.append(self.level2[meta_idxs[i]])

        timing["l2_search_ms"] = (time.perf_counter() - t) * 1000

        if depth == 2 or not sections:
            timing["retrieval_ms"] = sum(timing.values())
            return {"level": 2, "papers": papers, "sections": sections}, timing

        # ---------- Level 3 ----------
        t = time.perf_counter()
        chunks = []
        for s in sections:
            idx = self._get_l3_index(s["paper_id"], s["section_id"])
            if idx is None:
                continue

            D3, I3 = idx.search(qemb, self.k_max["l3"])
            meta_idxs = self.l3_manifest[
                f"{s['paper_id']}__{s['section_id']}"
            ]["meta_indices"]

            for score, i in zip(D3[0], I3[0]):
                if score < self.tau["l3"]:
                    break
                chunks.append(self.level3[meta_idxs[i]])

        timing["l3_search_ms"] = (time.perf_counter() - t) * 1000
        timing["retrieval_ms"] = sum(timing.values())

        return {
            "level": 3,
            "papers": papers,
            "sections": sections,
            "chunks": chunks,
        }, timing

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _build_context(self, result, max_chars=1500):
        blocks = []
        for sec in result.get("sections", [])[:3]:
            blocks.append(f"<chunk>\n{sec['text'][:max_chars]}\n</chunk>")
        for ch in result.get("chunks", [])[:10]:
            blocks.append(f"<chunk>\n{ch['text'][:max_chars]}\n</chunk>")
        return "\n".join(blocks)

    def generate(self, query, context):
        if not context.strip():
            return "Not found."

        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{self.SYSTEM}\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{query}\n\n{context}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.01,
            no_repeat_ngram_size=8,
            use_cache=False,
            eos_token_id=[
                self.tok.eos_token_id,
                self.tok.convert_tokens_to_ids("<|eot_id|>")
            ],
            pad_token_id=self.tok.pad_token_id,
        )
        
        gen_ids = out[:, inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen_ids[0], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, query):
        t0 = time.perf_counter()

        result, retrieval_timing = self.retrieve(query)
        t1 = time.perf_counter()

        context = self._build_context(result)
        answer = self.generate(query, context)
        t2 = time.perf_counter()

        return {
            "answer": answer,
            "retrieval_level": result["level"],
            "contexts": result,
            "timing": {
                **retrieval_timing,
                "generation_ms": (t2 - t1) * 1000,
                "total_ms": (t2 - t0) * 1000,
            },
            "config": self.config,
            "query": query,
        }

