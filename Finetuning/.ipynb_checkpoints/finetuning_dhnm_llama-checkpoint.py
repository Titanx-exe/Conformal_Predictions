import os
import json
import math
import random
import time
import gc
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm.auto import tqdm
import faiss


# Configuration

SEED = 1
MODEL_ID = "meta-llama/Llama-3.1-8B"

DATASET_PATH = "blink_training_1000000.jsonl"
OUTPUT_DIR = os.path.join("finetuned_models", "llama-3.1-8b-lora-dhnm_attn_blink_testttt_1m")
LOSS_LOG_PATH = os.path.join("Loss", "dhnm_llama_8b_attn_blink_1m_testttt.jsonl")

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
NEGATIVES_PER_QUERY = 15
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
MAX_SEQ_LENGTH = 2048
TEMPERATURE = 0.05

# DHNM
MINE_EVERY_N_STEPS = 100

# Logging
LOG_EVERY_N_STEPS = 10
SAVE_EVERY_EPOCH = True

# Distributed setup
process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
accelerator = Accelerator(
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    kwargs_handlers=[process_group_kwargs]
)
DEVICE = accelerator.device

# Seeding
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Using device: {DEVICE}")


# Utilities

def last_token_pooling(last_hidden_state, attention_mask):
    """Extract the last non-padding token representation."""
    # Sum of attention mask gives sequence length; -1 for last token index
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device), 
        sequence_lengths
    ]

# Data

class EntityLinkingDataset(Dataset):
    def __init__(self, path, max_samples=None):
        self.pairs = []
        self.unique_entities = set()
        self.doc_to_gold_entities = {}
        
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                obj = json.loads(line.strip())
                doc_text = obj["document"]
                ent_text = obj["entity"]
                
                self.pairs.append({
                    "document": doc_text,
                    "gold_entity": ent_text,
                    "negatives": []
                })
                self.unique_entities.add(ent_text)
                
                if doc_text not in self.doc_to_gold_entities:
                    self.doc_to_gold_entities[doc_text] = set()
                self.doc_to_gold_entities[doc_text].add(ent_text)
                
        random.shuffle(self.pairs)
        self.unique_entities_list = sorted(list(self.unique_entities))
        
        if accelerator.is_main_process:
            print(f"Loaded {len(self.pairs)} document-entity pairs.")
            print(f"Found {len(self.unique_entities_list)} unique candidate entities.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch, tokenizer, max_length):
    """Build contrastive batch with hard negatives."""
    queries = []
    candidates = set()
    corrects = []
    
    for doc in batch:
        queries.append(doc["document"])
        correct = doc["gold_entity"]
        corrects.append(correct)
        
        # Sample hard negatives if available
        if len(doc["negatives"]) >= NEGATIVES_PER_QUERY:
            cands = random.sample(doc["negatives"], NEGATIVES_PER_QUERY)
        elif len(doc["negatives"]) > 0:
            cands = doc["negatives"].copy()
        else:
            cands = []
            
        cands.append(correct)
        candidates.update(cands)
        
    candidates = list(candidates)
    random.shuffle(candidates)
    
    # Create labels pointing to gold entity indices in candidate list
    labels = [candidates.index(el) for el in corrects]

    doc_tokens = tokenizer(
        queries, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )
    ent_tokens = tokenizer(
        candidates, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )
    
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return doc_tokens, ent_tokens, labels_tensor


# Model Setup

def setup_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModel.from_pretrained(
        MODEL_ID, 
        dtype=dtype,
        attn_implementation="flash_attention_2"
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    model = get_peft_model(base_model, lora_config)
    
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    model.to(DEVICE)
    
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    
    return model, tokenizer


# Hard Negative Mining

def mine_hard_negatives(model, tokenizer, dataset, start_idx=None, end_idx=None):
    """
    Distributed hard negative mining using FAISS.
    Each GPU encodes a shard of entities, gathers full set, builds local FAISS index,
    then encodes document shards to mine negatives.
    """
    model.eval()

    world_size = accelerator.num_processes
    rank = accelerator.process_index
    is_distributed = dist.is_initialized() and world_size > 1
    target_bs = 64
    entities = dataset.unique_entities_list

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = len(dataset.pairs)

    total_entities = len(entities)
    total_docs = end_idx - start_idx

    print(f"[Rank {rank}/{world_size}] DHNM starts | {total_entities} entities | {total_docs} documents | distributed={is_distributed}")

    # Distributed Entity Encoding
    t0 = time.time()

    if is_distributed:
        ents_per_gpu = math.ceil(total_entities / world_size)
        my_ent_start = rank * ents_per_gpu
        my_ent_end = min(my_ent_start + ents_per_gpu, total_entities)
    else:
        my_ent_start = 0
        my_ent_end = total_entities

    my_entities = entities[my_ent_start:my_ent_end]
    print(f"[Rank {rank}] Entity shard: [{my_ent_start}:{my_ent_end}] = {len(my_entities)} entities")

    local_ent_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(my_entities), target_bs),
                      desc=f"[R{rank}] Encoding Entities",
                      disable=(rank != 0)):
            batch_ents = my_entities[i : i + target_bs]
            tokens = tokenizer(batch_ents, max_length=MAX_SEQ_LENGTH, padding=True, truncation=True, return_tensors="pt")
            tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
            outputs = model(**tokens)
            embeds = last_token_pooling(outputs.last_hidden_state, tokens["attention_mask"])
            embeds = F.normalize(embeds, p=2, dim=-1)
            local_ent_embeds.append(embeds)

    if local_ent_embeds:
        local_ent_tensor = torch.cat(local_ent_embeds, dim=0)
    else:
        hidden_dim = model.config.hidden_size
        local_ent_tensor = torch.empty(0, hidden_dim, device=DEVICE, dtype=torch.bfloat16)

    print(f"[Rank {rank}] Local entity tensor shape: {local_ent_tensor.shape}, dtype: {local_ent_tensor.dtype}")

    if is_distributed:
        local_count = torch.tensor([local_ent_tensor.shape[0]], device=DEVICE, dtype=torch.long)
        all_counts = [torch.zeros(1, device=DEVICE, dtype=torch.long) for _ in range(world_size)]
        dist.all_gather(all_counts, local_count)
        counts_list = [int(c.item()) for c in all_counts]
        max_count = max(counts_list)
        hidden_dim = local_ent_tensor.shape[1]

        print(f"[Rank {rank}] Entity counts per GPU: {counts_list}, max_count: {max_count}, hidden_dim: {hidden_dim}")

        # Pad for all_gather
        padded = torch.zeros(max_count, hidden_dim, device=DEVICE, dtype=local_ent_tensor.dtype)
        padded[:local_ent_tensor.shape[0]] = local_ent_tensor

        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)

        # Trim padding; concatenate in rank order
        trimmed = [gathered[i][:counts_list[i]] for i in range(world_size)]
        all_ent_embeds = torch.cat(trimmed, dim=0).float().cpu().numpy()

        print(f"[Rank {rank}] Gathered entity embeddings shape: {all_ent_embeds.shape}")
        del local_ent_tensor, padded, gathered, trimmed, local_ent_embeds
    else:
        all_ent_embeds = local_ent_tensor.float().cpu().numpy()
        del local_ent_tensor, local_ent_embeds

    torch.cuda.empty_cache()
    t1 = time.time()
    print(f"[Rank {rank}] Phase 1 complete | Entity embeds shape: {all_ent_embeds.shape} | Time: {t1 - t0:.2f}s")

    assert all_ent_embeds.shape[0] == total_entities, \
        f"[Rank {rank}] ENTITY COUNT MISMATCH: got {all_ent_embeds.shape[0]}, expected {total_entities}"

    # Build FAISS Index (every GPU builds independently)
    vector_dim = all_ent_embeds.shape[1]
    index = faiss.IndexFlatIP(vector_dim)
    index.add(all_ent_embeds)

    print(f"[Rank {rank}] Phase 2 complete | FAISS index vectors: {index.ntotal}, dimension: {vector_dim}")

    # Distributed Document Encoding + FAISS Search
    t2 = time.time()

    if is_distributed:
        docs_per_gpu = math.ceil(total_docs / world_size)
        my_doc_start = start_idx + rank * docs_per_gpu
        my_doc_end = min(my_doc_start + docs_per_gpu, end_idx)
    else:
        my_doc_start = start_idx
        my_doc_end = end_idx

    my_doc_count = my_doc_end - my_doc_start
    print(f"[Rank {rank}] Document shard: [{my_doc_start}:{my_doc_end}] = {my_doc_count} documents")

    my_negatives = {}
    total_faiss_time = 0.0
    total_encode_time = 0.0

    with torch.no_grad():
        for i in tqdm(range(my_doc_start, my_doc_end, target_bs),
                      desc=f"[R{rank}] Mining Negatives",
                      disable=(rank != 0)):
            batch_end = min(i + target_bs, my_doc_end)
            batch_docs = [dataset.pairs[j]["document"] for j in range(i, batch_end)]

            enc_start = time.time()
            tokens = tokenizer(batch_docs, max_length=MAX_SEQ_LENGTH, padding=True, truncation=True, return_tensors="pt")
            tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
            outputs = model(**tokens)
            doc_embeds = last_token_pooling(outputs.last_hidden_state, tokens["attention_mask"])
            doc_embeds = F.normalize(doc_embeds, p=2, dim=-1).float().cpu().numpy()
            enc_end = time.time()
            total_encode_time += (enc_end - enc_start)

            k = 30
            faiss_start = time.time()
            scores, indices = index.search(doc_embeds, k)
            faiss_end = time.time()
            total_faiss_time += (faiss_end - faiss_start)

            for doc_batch_idx in range(len(batch_docs)):
                global_idx = i + doc_batch_idx
                doc_str = dataset.pairs[global_idx]["document"]
                all_valid_entities = dataset.doc_to_gold_entities[doc_str]

                hard_negs = []
                for ent_idx in indices[doc_batch_idx]:
                    candidate_str = entities[ent_idx]
                    if candidate_str not in all_valid_entities:
                        hard_negs.append(candidate_str)

                my_negatives[global_idx] = hard_negs

    t3 = time.time()
    print(f"[Rank {rank}] Phase 3 complete | Doc encoding: {total_encode_time:.2f}s | FAISS search: {total_faiss_time:.2f}s | Total: {t3 - t2:.2f}s")
    print(f"[Rank {rank}] Mined negatives for {len(my_negatives)} documents (indices {my_doc_start} to {my_doc_end - 1})")

    # Gather mined negatives from all GPUs
    if is_distributed:
        accelerator.wait_for_everyone()

        all_neg_dicts = [None] * world_size
        dist.all_gather_object(all_neg_dicts, my_negatives)

        total_updated = 0
        for gpu_idx, neg_dict in enumerate(all_neg_dicts):
            count = len(neg_dict)
            total_updated += count
            print(f"[Rank {rank}] Received {count} negatives from GPU {gpu_idx}")

        for neg_dict in all_neg_dicts:
            for global_idx, hard_negs in neg_dict.items():
                dataset.pairs[global_idx]["negatives"] = hard_negs

        t4 = time.time()
        print(f"[Rank {rank}] Phase 4 complete | Merged {total_updated} total negatives | Gather time: {t4 - t3:.2f}s")
    else:
        for global_idx, hard_negs in my_negatives.items():
            dataset.pairs[global_idx]["negatives"] = hard_negs
        t4 = time.time()

    total_time = t4 - t0
    print(f"[Rank {rank}] DHNM COMPLETE | Total time: {total_time:.2f}s")

    docs_with_negatives = sum(1 for p in dataset.pairs[start_idx:end_idx] if len(p["negatives"]) > 0)
    print(f"[Rank {rank}] Verification: {docs_with_negatives}/{total_docs} documents have negatives\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Training Loop

def train_model(model, tokenizer, dataset):
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, MAX_SEQ_LENGTH),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    total_steps = (len(dataloader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    scheduler = accelerator.prepare(scheduler)

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"  Dataset size:       {len(dataset)}")
        print(f"  Epochs:             {NUM_EPOCHS}")
        print(f"  Effective batch:    {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Target Candidates:  {NEGATIVES_PER_QUERY + 1}")
        print(f"  DHNM Frequency:     Every {MINE_EVERY_N_STEPS} steps")
        print(f"{'='*60}\n")

        os.makedirs(os.path.dirname(LOSS_LOG_PATH), exist_ok=True)
        with open(LOSS_LOG_PATH, "w") as f:
            f.write(json.dumps({
                "event": "training_hyperparameters",
                "model_id": MODEL_ID,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "batch_size_effective": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "negatives_per_query": NEGATIVES_PER_QUERY,
                "warmup_ratio": WARMUP_RATIO,
                "weight_decay": WEIGHT_DECAY,
                "max_seq_length": MAX_SEQ_LENGTH,
                "temperature": TEMPERATURE,
                "dhnm_frequency": MINE_EVERY_N_STEPS
            }) + "\n")

    global_step = 0
    cuda_warmup_done = False
    model.train()

    for epoch in range(NUM_EPOCHS):
        # DistributedSampler handles shuffling; set_epoch ensures different permutation each epoch
        dataloader.set_epoch(epoch)
        
        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", disable=not accelerator.is_main_process)

        for step, (doc_batch, ent_batch, labels) in enumerate(progress_bar):
            if step > 0 and step % MINE_EVERY_N_STEPS == 0:
                if accelerator.is_main_process:
                    with open(LOSS_LOG_PATH, "a") as f:
                        f.write(json.dumps({"event": "dhnm_mining", "epoch": epoch+1, "step": step}) + "\n")
                
                unwrapped_model = accelerator.unwrap_model(model)
                mine_hard_negatives(unwrapped_model, tokenizer, dataset)
                model.train()

            with accelerator.accumulate(model):
                doc_batch = {k: v.to(DEVICE) for k, v in doc_batch.items()}
                ent_batch = {k: v.to(DEVICE) for k, v in ent_batch.items()}
                labels = labels.to(DEVICE)

                doc_outputs = model(**doc_batch)
                ent_outputs = model(**ent_batch)

                doc_embeds = last_token_pooling(doc_outputs.last_hidden_state, doc_batch["attention_mask"])
                ent_embeds = last_token_pooling(ent_outputs.last_hidden_state, ent_batch["attention_mask"])
                
                doc_embeds = F.normalize(doc_embeds, p=2, dim=-1)
                ent_embeds = F.normalize(ent_embeds, p=2, dim=-1)

                # Contrastive loss
                similarity = (doc_embeds @ ent_embeds.T) / TEMPERATURE
                loss = F.cross_entropy(similarity, labels)
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if not cuda_warmup_done:
                torch.cuda.empty_cache()
                cuda_warmup_done = True
                
            gathered_loss = accelerator.gather(loss.detach().clone())
            current_loss = gathered_loss.mean().item()


            if accelerator.is_main_process:
                with open(LOSS_LOG_PATH, "a") as f:
                    f.write(json.dumps({
                        "event": "loss",
                        "epoch": epoch+1,
                        "total_epochs": NUM_EPOCHS,
                        "step": step,
                        "loss": current_loss
                    }) + "\n")
            
            epoch_loss += current_loss
            num_batches += 1
            

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % LOG_EVERY_N_STEPS == 0 and accelerator.is_main_process:
                    avg_loss = epoch_loss / max(num_batches, 1)
                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")


        # Save adapter after each epoch
        if SAVE_EVERY_EPOCH:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                epoch_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(epoch_dir)
                tokenizer.save_pretrained(epoch_dir)
                print(f"  -> Adapter saved to {epoch_dir}")


    # Final Save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(OUTPUT_DIR, "final")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"\nTraining complete! Final adapter saved to: {final_dir}")
        
    return model



def main():
    dataset = EntityLinkingDataset(DATASET_PATH, max_samples=None)
    model, tokenizer = setup_model_and_tokenizer()
    model = train_model(model, tokenizer, dataset)


if __name__ == "__main__":
    main()