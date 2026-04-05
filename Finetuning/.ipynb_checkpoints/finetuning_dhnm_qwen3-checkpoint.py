import os
import gc
import json
import math
import random
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm.auto import tqdm
import faiss
import numpy as np
from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta

# Global Accelerate Init
process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
accelerator = Accelerator(
    gradient_accumulation_steps=4,
    kwargs_handlers=[process_group_kwargs]
)
DEVICE = accelerator.device

# Reproducibility
SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

import gc
print(f"Using device: {DEVICE}")

# ── Model ──
MODEL_ID = "Qwen/Qwen3-Embedding-4B"

# ── Paths ──
DATASET_PATH = "aida_finetune.jsonl"
OUTPUT_DIR = os.path.join("finetuned_models", "qwen3-4b-lora-dhnm_attn_mlp_aida_complete")
LOSS_LOG_PATH = os.path.join("Loss", "dhnm_qwen3_4b_attn_mlp_aida_complete.jsonl")

# ── LoRA Parameters ──
LORA_R = 16                
LORA_ALPHA = 32            
LORA_DROPOUT = 0.05        
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    # "q_proj", "k_proj", "v_proj", "o_proj"
    # "gate_proj", "up_proj", "down_proj"
]

# ── Training Hyperparameters ──
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
BATCH_SIZE = 8                  # Number of queries per step
GRADIENT_ACCUMULATION_STEPS = 4 # Enable gradient accumulation if OOM issues persist
NEGATIVES_PER_QUERY = 15            # Number of hard negatives sampled per query
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
MAX_SEQ_LENGTH = 2048
TEMPERATURE = 0.05                  

# ── DHNM Parameters ──
MINE_EVERY_N_STEPS = 2895          # How often to rebuild the FAISS index and find new negatives

# ── Logging ──
LOG_EVERY_N_STEPS = 10
SAVE_EVERY_EPOCH = True

class EntityLinkingDataset(Dataset):
    def __init__(self, path, max_samples=None):
        self.pairs = []
        self.unique_entities = set()
        
        # New: Track all gold entities for a single document to prevent false "hard negatives"
        self.doc_to_gold_entities = {}
        
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                obj = json.loads(line.strip())
                doc_text = obj["document"]
                ent_text = obj["entity"]
                
                # Each item holds the document, the gold entity, and an initially empty list of negatives
                self.pairs.append({
                    "document": doc_text,
                    "gold_entity": ent_text,
                    "negatives": [] 
                })
                self.unique_entities.add(ent_text)
                
                if doc_text not in self.doc_to_gold_entities:
                    self.doc_to_gold_entities[doc_text] = set()
                self.doc_to_gold_entities[doc_text].add(ent_text)
                
        # Shuffle initial pairs
        random.shuffle(self.pairs)
        self.unique_entities_list = sorted(list(self.unique_entities))
        
        # Only print loading stats on the main process
        if accelerator.is_main_process:
            print(f"Loaded {len(self.pairs)} document-entity pairs.")
            print(f"Found {len(self.unique_entities_list)} unique candidate entities.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def collate_fn(batch, tokenizer, max_length):
    """
    Constructs the batch. For each document, it pairs it with its gold entity and N sampled hard negatives.
    """
    queries = []
    candidates = set()
    labels = []
    corrects = []
    
    for doc in batch:
        queries.append(doc["document"])
        correct = doc["gold_entity"]
        corrects.append(correct)
        
        # Sample hard negatives if they exist, else sample random entities (fallback for initial epochs)
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
    
    # Create integer labels pointing to the index of the gold entity in the candidates list
    for el in corrects:
        labels.append(candidates.index(el))

    doc_tokens = tokenizer(
        queries, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )
    ent_tokens = tokenizer(
        candidates, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )
    
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return doc_tokens, ent_tokens, labels_tensor

def mean_pooling(last_hidden_state, attention_mask):
    # Mean pooling over non-padding tokens.
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def setup_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModel.from_pretrained(
        MODEL_ID, 
        dtype=dtype,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
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
    
    # CRITICAL: enable_input_require_grads ensures the embedding layer's outputs
    # have requires_grad=True, which is REQUIRED for use_reentrant=True checkpointing.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    model.to(DEVICE)
    
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    
    return model, tokenizer

def mine_hard_negatives(model, tokenizer, dataset, start_idx=None, end_idx=None):
    """
    1. Encodes all unique entities.
    2. Builds FAISS index.
    3. Encodes training documents.
    4. Finds Top-K confusing entities for each document and saves them in the dataset.
    """
    model.eval()
    
    if accelerator.is_main_process:
        print("\n[DHNM] Mining hard negatives...")
        
        # 1. Encode All Entities
        # We do this in batches to avoid OOM
        entity_embeds_list = []
        target_bs = 64
        entities = dataset.unique_entities_list
        
        with torch.no_grad():
            for i in tqdm(range(0, len(entities), target_bs), desc="Encoding KB"):
                batch_ents = entities[i : i + target_bs]
                tokens = tokenizer(batch_ents, max_length=MAX_SEQ_LENGTH, padding=True, truncation=True, return_tensors="pt")
                tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
                outputs = model(**tokens)
                embeds = mean_pooling(outputs.last_hidden_state, tokens["attention_mask"])
                
                # Crucial: Normalize for FAISS IP -> Cosine Similarity
                embeds = F.normalize(embeds, p=2, dim=-1)
                # Store in CPU memory
                entity_embeds_list.append(embeds.cpu().numpy())
                
        # shape: (Num_Entities, 2048) or similar
        all_ent_embeds = np.concatenate(entity_embeds_list, axis=0) 
        
        # 2. Build FAISS Index (Inner Product for Cosine Similarity since vectors are normalized)
        vector_dim = all_ent_embeds.shape[1]
        index = faiss.IndexFlatIP(vector_dim)
        index.add(all_ent_embeds)
        
        # 3. Mine Negatives for Documents
        if start_idx is None: start_idx = 0
        if end_idx is None: end_idx = len(dataset.pairs)
        
        with torch.no_grad():
            for i in tqdm(range(start_idx, end_idx, target_bs), desc="Mining Negatives"):
                # Grab next batch of document strings
                batch_docs = [dataset.pairs[j]["document"] for j in range(i, min(i+target_bs, end_idx))]
                tokens = tokenizer(batch_docs, max_length=MAX_SEQ_LENGTH, padding=True, truncation=True, return_tensors="pt")
                tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
                outputs = model(**tokens)
                doc_embeds = mean_pooling(outputs.last_hidden_state, tokens["attention_mask"])
                
                # Crucial: Normalize query documents
                doc_embeds = F.normalize(doc_embeds, p=2, dim=-1).cpu().numpy()
                
                # Ask FAISS for the highest similarity candidates. 
                # Request Top 30 (to ensure we still have candidates if the top 1 is the gold entity)
                k = 30
                scores, indices = index.search(doc_embeds, k)
                
                for doc_batch_idx in range(len(batch_docs)):
                    global_idx = i + doc_batch_idx
                    doc_str = dataset.pairs[global_idx]["document"]
                    
                    # Fetch ALL valid entities for this specific document
                    all_valid_entities = dataset.doc_to_gold_entities[doc_str]
                    
                    # Filter out ALL valid entities from the retrieved candidates
                    hard_negs = []
                    for score_idx, ent_idx in enumerate(indices[doc_batch_idx]):
                        candidate_str = entities[ent_idx]
                        if candidate_str not in all_valid_entities:
                            hard_negs.append(candidate_str)
                            
                    # Update the dataset with the newly discovered hard mistakes
                    dataset.pairs[global_idx]["negatives"] = hard_negs
                    
        # Model back to train mode done in training loop
        print("[DHNM] Finished mining.\n")
    
    # Halt all GPUs until Rank 0 finishes indexing and mining
    accelerator.wait_for_everyone()
    
    # Broadcast the newly discovered negatives uniformly to all processes
    objects = [dataset.pairs] if accelerator.is_main_process else [None]
    if dist.is_initialized():
        dist.broadcast_object_list(objects, src=0)
    dataset.pairs = objects[0]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train_model(model, tokenizer, dataset):
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer, MAX_SEQ_LENGTH),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Let accelerator handle distribution, device assignment, and sharding
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Calculate steps per epoch based on the newly chunked local dataloader
    total_steps = (len(dataloader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
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
        # DistributedSampler handles shuffling; set_epoch ensures a different permutation each epoch.
        dataloader.set_epoch(epoch)
        
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", disable=not accelerator.is_main_process)

        for step, (doc_batch, ent_batch, labels) in enumerate(progress_bar):
            
            # Trigger DHNM periodically based on raw batch steps
            if step > 0 and step % MINE_EVERY_N_STEPS == 0:
                if accelerator.is_main_process:
                    with open(LOSS_LOG_PATH, "a") as f:
                        f.write(json.dumps({"event": "dhnm_mining", "epoch": epoch+1, "step": step}) + "\n")
                
                # Unwrap model so it doesn't trigger DDP hooks during native inference loop
                unwrapped_model = accelerator.unwrap_model(model)
                mine_hard_negatives(unwrapped_model, tokenizer, dataset)
                model.train() # Resume training

            with accelerator.accumulate(model):
                doc_batch = {k: v.to(DEVICE) for k, v in doc_batch.items()}
                ent_batch = {k: v.to(DEVICE) for k, v in ent_batch.items()}
                labels = labels.to(DEVICE)

                doc_outputs = model(**doc_batch)
                ent_outputs = model(**ent_batch)

                doc_embeds = mean_pooling(doc_outputs.last_hidden_state, doc_batch["attention_mask"])
                ent_embeds = mean_pooling(ent_outputs.last_hidden_state, ent_batch["attention_mask"])
                
                doc_embeds = F.normalize(doc_embeds, p=2, dim=-1)
                ent_embeds = F.normalize(ent_embeds, p=2, dim=-1)

                # Contrastive Loss Calculation
                similarity = (doc_embeds @ ent_embeds.T) / TEMPERATURE
                
                # Compute cross entropy
                loss = F.cross_entropy(similarity, labels)
                
                # accelerate handles dividing loss properly inside context
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if not cuda_warmup_done:
                torch.cuda.empty_cache()
                cuda_warmup_done = True
                
            # Gather loss strictly for reporting
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
    # 1. Load Data
    dataset = EntityLinkingDataset(DATASET_PATH, max_samples=None)

    # 2. Setup Model
    model, tokenizer = setup_model_and_tokenizer()

    # 3. Train
    model = train_model(model, tokenizer, dataset)

if __name__ == "__main__":
    main()