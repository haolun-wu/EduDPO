# EduDPO Data Generation Pipeline


## Prerequisites

- Python 3.10

## Data Generation Steps

### 1. Student Solution Simulation

```bash
python scripts/simulate_student.py \
    --input_file data/raw/questions.json \
    --output_file data/simulated/questions_stu_answers.json \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --num_simulations 5 \
    --removal_probability 0.3 \
    --number_modification_range 0.2 \
    --number_modification_chance 0.5
```

### 2. LLM Feedback Generation

```bash
python scripts/generate_llm_feedback.py \
    --input_file data/simulated/questions_stu_answers.json \
    --output_file data/simulated/questions_llm_feedbacks.json \
    --model_names allenai/OLMo-2-1124-7B-Instruct meta-llama/Llama-3.1-8B-Instruct
```


### 3. TA Feedback Generation

```bash
python scripts/generate_ta_feedback.py \
    --input_file data/simulated/questions_llm_feedbacks.json \
    --output_file data/simulated/questions_ta_feedbacks.json \
    --model_name microsoft/Phi-4-mini-instruct \
    --max_prompt_tokens 8000
```


### 4. DPO Format Conversion

Convert the feedback data to DPO training format:

```bash
python scripts/convert_to_dpo.py \
    --input_file data/simulated/questions_ta_feedbacks.json \
    --output_file data/processed/dpo_samples.json
```
