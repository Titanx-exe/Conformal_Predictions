import inspect
from transformers. models.llama. modeling_llama import apply_rotary_pos_emb, LlamaModel

# Print the source of apply_rotary_pos_emb
print("apply_rotary_pos_emb source:")
print(inspect.getsource(apply_rotary_pos_emb))

print("\n" + "="*80 + "\n")

# Check LlamaModel's forward to see how it computes position_embeddings
print("LlamaModel forward signature and rotary handling:")
print(inspect.getsource(LlamaModel.forward))
