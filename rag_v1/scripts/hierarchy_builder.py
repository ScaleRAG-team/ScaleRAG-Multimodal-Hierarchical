import os, math, json, pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global dictionaries to hold the generated summaries and mappings
paper_summaries: dict = {}
section_summaries: dict = {}
paper_to_sections: dict = {}
section_to_chunks: dict = {}

def build_hierarchy(
    papers,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    max_sec_summary_len: int = 96,
    max_paper_summary_len: int = 192,
    batch_size: int = 1,
    resume: bool = True,
    summary_cache_path: str = "../data2/RAG/Version_V2/hierarchy/hierarchy_summaries.json"
):

    global paper_summaries, section_summaries, paper_to_sections, section_to_chunks

    # 1. Load model in FP16
    print(f"[build_hierarchy] Loading summarization model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

    # 2. Resume from cache
    processed_ids = set()
    if resume and os.path.exists(summary_cache_path):
        try:
            with open(summary_cache_path, 'r') as f:
                cached = json.load(f)
            for pid, text in cached.get("paper_summaries", {}).items():
                paper_summaries[pid] = text
                processed_ids.add(pid)
            for sid, text in cached.get("section_summaries", {}).items():
                section_summaries[sid] = text
            paper_to_sections.update(cached.get("paper_to_sections", {}))
            section_to_chunks.update(cached.get("section_to_chunks", {}))
            print(f"[build_hierarchy] Loaded cache with {len(processed_ids)} papers already summarized. Resuming...")
        except Exception as e:
            print(f"[build_hierarchy] Warning: Failed to load cache from {summary_cache_path} ({e}). Starting fresh.")
            processed_ids = set()

    # 3. Iterate papers
    print("[build_hierarchy] Beginning hierarchical summarization...")
    paper_count = 0

    for paper in papers:
        paper_id = paper["paper_id"]
        if paper_id in processed_ids:
            continue

        sections = paper.get("sections", [])
        paper_to_sections[paper_id] = []
        section_texts = []
        section_ids = []

        # Reduce tokenizer context 2048 → 768 (prevents OOM)
        for sec in sections:
            sec_id = sec.get("section_id")
            if sec_id is None:
                sec_index = len(paper_to_sections[paper_id])
                sec_id = f"{paper_id}::section{sec_index}"

            paper_to_sections[paper_id].append(sec_id)
            section_to_chunks[sec_id] = sec.get("chunk_ids", [])

            sec_content = " ".join(sec.get("chunks", []))

            encoded = tokenizer(
                sec_content,
                truncation=True,
                max_length=768,
                return_tensors=None
            )
            sec_content = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)

            section_texts.append(sec_content)
            section_ids.append(sec_id)

        # 4. Section summaries
        section_sum_texts = []
        for batch_start in range(0, len(section_texts), batch_size):
            batch_texts = section_texts[batch_start:batch_start + batch_size]
            batch_sec_ids = section_ids[batch_start:batch_start + batch_size]
            if not batch_texts:
                continue

            prompts = []
            for text in batch_texts:
                prompt = (
                    f"<|im_start|>system\nYou are a helpful assistant specialized in summarizing academic papers.<|im_end|>\n"
                    f"<|im_start|>user\nSummarize the following section of a research paper:\n{text}\n<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            # Disable KV-cache to reduce VRAM
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_sec_summary_len,
                    num_beams=1,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=False,          # <<<<<< FIXED
                    tokenizer=tokenizer,
                    stop_strings=["<|im_end|>", "</s>"],
                )

            summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for sum_text in summaries:
                section_sum_texts.append(sum_text.strip())

            for sid, stext in zip(batch_sec_ids, section_sum_texts[-len(batch_sec_ids):]):
                section_summaries[sid] = stext

        # 5. Paper-level summary
        full_input = ""
        for sec, sec_id in zip(sections, section_ids):
            title = sec.get("title") or sec.get("section_title") or ""
            sec_summary = section_summaries.get(sec_id, "")
            if title:
                full_input += f"{title}: {sec_summary}\n"
            else:
                full_input += f"{sec_summary}\n"

        prompt = (
            f"<|im_start|>system\nYou are an assistant summarizing research papers.<|im_end|>\n"
            f"<|im_start|>user\nHere are summaries of each section of a paper. Provide a concise overall summary:\n{full_input}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_paper_summary_len,
                num_beams=1,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,             # <<<<<< FIXED
                tokenizer=tokenizer,
                stop_strings=["<|im_end|>", "</s>"],
            )

        paper_summary = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        paper_summaries[paper_id] = paper_summary

        paper_count += 1

        # Save checkpoint every 50
        if paper_count % 50 == 0:
            try:
                cache_data = {
                    "paper_summaries": paper_summaries,
                    "section_summaries": section_summaries,
                    "paper_to_sections": paper_to_sections,
                    "section_to_chunks": section_to_chunks
                }
                with open(summary_cache_path, 'w') as f:
                    json.dump(cache_data, f)
                print(f"[build_hierarchy] Saved intermediate summaries for {paper_count} papers.")
            except Exception as e:
                print(f"[build_hierarchy] Warning: Failed to save intermediate cache ({e}).")

    # Final save
    try:
        cache_data = {
            "paper_summaries": paper_summaries,
            "section_summaries": section_summaries,
            "paper_to_sections": paper_to_sections,
            "section_to_chunks": section_to_chunks
        }
        with open(summary_cache_path, 'w') as f:
            json.dump(cache_data, f)
        print(f"[build_hierarchy] Completed summaries for {len(paper_summaries)} papers.")
    except Exception as e:
        print(f"[build_hierarchy] Warning: Could not save final summaries ({e}).")

    return paper_summaries, section_summaries
