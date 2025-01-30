# GPTX
Reproducing GPTX from scratch using PyTorch

## TODO 
- [ ] Implement the model architecture
   - [ ] Implement the MHA layer in `layers.attention.attention.py`
   - [ ] Implement the MLA layer 
- [ ] Implement BPE tokenizer
   - [ ] train
   - [ ] tokenize 


## References
1. The architectures 
   1. LitGPT: https://github.com/Lightning-AI/litgpt
   2. nanoGPT: https://github.com/karpathy/nanoGPT
   2. llama3 from scratch: https://github.com/naklecha/llama3-from-scratch
   3. LLMs from scratch: https://github.com/rasbt/LLMs-from-scratch
   4. x-transformers: https://github.com/lucidrains/x-transformers
1. minbpe: https://github.com/karpathy/minbpe
1. RLHF
   1. PPO: https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py
   1. DPO: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py
1. Use Case
   1. chess GPT: https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html

## Run Tests
```bash
bash scripts/run_tests.sh
```

## Check Code Quality
```bash
bash scripts/codeqa.sh
```
