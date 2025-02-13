# Synthetic Answer Generation with SGLang  

This repository contains Python code to generate synthetic answers from questions using SGLang. The implementation is largely based on the [generate_reasoning.py](https://github.com/huggingface/open-r1/blob/main/scripts/generate_reasoning.py) script from the **open-r1** project. A huge thanks to **Hugging Face** for their excellent work!  

## Enhancements  

We have added a **continue generation** option, which checks whether a UUID has already been generated. If the UUID exists, the script will skip it. This is useful when resuming an interrupted run. *(Note: This feature only works when execution is stopped midway.)*  

## Usage  

### 1. Start the SGLang Server  

Follow the [SGLang installation guide](https://docs.sglang.ai/start/install.html) to set up the environment.  

#### **For a small model:**  
```sh
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --port 30000 \
  --host 0.0.0.0 \
  --dp 4
```

#### **For a large model:**
```sh
python -m sglang.launch_server \
  --model-path simplescaling/s1-32B \
  --port 30000 \
  --host 0.0.0.0 \
  --tp 4
```

### 2. Generate Synthetic Answers

Run the following command to start generating synthetic answers:
```
python sglang.py \
  --dataset-name "OpenMedical/medical-data-dedup" \
  --output-file "medical_data_generations.jsonl" \
  --prompt-column "question" \
  --uuid-column "question" \
  --api-addr "127.0.0.1:30000" \
  --num-generations 3 \
  --max-tokens 16384 \
  --max-concurrent 200
```