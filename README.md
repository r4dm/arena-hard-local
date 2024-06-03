# Arena-Hard-Local
Arena-Hard-Local is an adaptation of [lm-sys/arena-hard-auto](https://github.com/lm-sys/arena-hard-auto) to use the local model as a judge and also for multilingual use (Russian by default).
The repository uses a simplified evaluation prompt in which the model does not have to also answer the question that the evaluated models answered. This was done intentionally, since first of all the project is intended for quick evaluation of checkpoints after fine-tuning. In addition, we noticed that the performance of the local 70b model is sometimes not enough for a full prompt. For full evaluation, please use the original repository.

Check out our blog post for more details about how Arena Hard Auto v0.1 works -> [Blog post link](https://lmsys.org/blog/2024-04-19-arena-hard/).

## Local Model
To evaluate models, it is proposed to use Meta-Llama-3-70B-Instruct or fine-tuned for these purposes [Llama-3-70B-Instruct-AH-AWQ](https://huggingface.co/radm/Llama-3-70B-Instruct-AH-AWQ), trained to judge the stronger model gpt-4-1106-preview.

We use original hard arena questions translated into Russian [Vikhrmodels/arena_hard_ru](https://huggingface.co/datasets/Vikhrmodels/arena_hard_ru).
To use another language, replace the file **data/arena-hard-v0.1/question.jsonl** and make sure that the resulting json is saved with the required encoding (line 103 in **gen_answer.py**).

### Llama-3-70B-Instruct as judge:
```console
Llama-3-Instruct-8B-SimPO                          | score: 78.3  | 95% CI:   (-1.5, 1.2)   | average #tokens: 545
SELM-Llama-3-8B-Instruct-iter-3                    | score: 72.8  | 95% CI:   (-2.1, 1.4)   | average #tokens: 606
Meta-Llama-3-8B-Instruct-f16                       | score: 65.3  | 95% CI:   (-1.8, 2.1)   | average #tokens: 560
suzume-llama-3-8B-multilingual-orpo-borda-half     | score: 63.5  | 95% CI:   (-1.6, 2.1)   | average #tokens: 978
Phi-3-medium-128k-instruct                         | score: 50.0  | 95% CI:   (0.0, 0.0)    | average #tokens: 801
suzume-llama-3-8B-multilingual                     | score: 48.1  | 95% CI:   (-2.2, 1.8)   | average #tokens: 767
aya-23-8B                                          | score: 48.0  | 95% CI:   (-2.0, 2.1)   | average #tokens: 834
Vikhr-7B-instruct_0.5                              | score: 19.6  | 95% CI:   (-1.3, 1.5)   | average #tokens: 794
alpindale_gemma-2b-it                              | score: 11.2  | 95% CI:   (-1.0, 0.8)   | average #tokens: 425
```
### Llama-3-70B-Instruct-AH-AWQ as judge:
```console
Llama-3-Instruct-8B-SimPO                          | score: 83.8  | 95% CI:   (-1.4, 1.3)   | average #tokens: 545
SELM-Llama-3-8B-Instruct-iter-3                    | score: 78.8  | 95% CI:   (-1.7, 1.9)   | average #tokens: 606
suzume-llama-3-8B-multilingual-orpo-borda-half     | score: 71.8  | 95% CI:   (-1.7, 2.4)   | average #tokens: 978
Meta-Llama-3-8B-Instruct-f16                       | score: 69.8  | 95% CI:   (-1.9, 1.7)   | average #tokens: 560
suzume-llama-3-8B-multilingual                     | score: 54.0  | 95% CI:   (-2.1, 2.1)   | average #tokens: 767
aya-23-8B                                          | score: 50.4  | 95% CI:   (-1.7, 1.7)   | average #tokens: 834
Phi-3-medium-128k-instruct                         | score: 50.0  | 95% CI:   (0.0, 0.0)    | average #tokens: 801
Vikhr-7B-instruct_0.5                              | score: 14.2  | 95% CI:   (-1.3, 1.0)   | average #tokens: 794
alpindale_gemma-2b-it                              | score:  7.9  | 95% CI:   (-0.9, 0.8)   | average #tokens: 425
```

## Install Dependencies
```
[git clone https://github.com/lm-sys/arena-hard.git](https://github.com/radaevm/arena-hard-local.git)
cd arena-hard-local
pip install -r requirements.txt
pip install -r requirements-optional.txt  # Optional dependencies (e.g., anthropic sdk)
```

## Run bench
Just run
```console
python3 show_result.py --judge-name Llama-3-70B-Instruct-AH-AWQ --baseline Phi-3-medium-128k-instruct
```
Running `show_results.py` will save generated battles into `data/arena_hard_battles.jsonl` and bootstrapping statistics into `data/bootstrapping_results.jsonl`. If you don't want to regenerate battles or bootstrapping statistics, simply toggle argument `--load-battles` or `--load-bootstrap`, respectively.

## Evaluate a new model on Arena-Hard-Local

### Step 1. Set up the endpoint config to your model

Fill in your API endpoint in `config/api_config.yaml`. We support OpenAI compatible API server. You can specify `parallel` to indicate the number of concurrent API requests (default: 1).
```yaml
# example
aya-23-8B:
     model_name: /home/jupyter/models/aya-23-8B
     endpoints:
         - api_base: http://localhost:8010/v1
           api_key: token-abc123
     api_type: openai
     parallel: 16
```
You may use inference engine such as [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) or [SGLang](https://github.com/sgl-project/sglang?tab=readme-ov-file#using-local-models) to host your model with an OpenAI compatible API server.


### Step 2. Generate Model Answers

In `config/gen_answer_config.yaml`, add your model name in `model_list`.
```yaml
bench_name: arena-hard-v0.1
temperature: 0.0
max_tokens: 4096
num_choices: 1

model_list:
  - [YOUR-MODEL-NAME]
```
Run the command to generate answers:
```console
python3 gen_answer.py
```
Caching feature is implemented. The code will skip generating an answer when there is already an existing answer/judgment to the same prompt. 

### Step 3. Generate Judgments

In `config/judge_config.yaml`, add your model name in `model_list`.
```yaml
...
# Add your model below for evaluation
model_list:
  - gpt-3.5-turbo-0125
  - [YOUR-MODEL-NAME]
```

Run the command to generate judgments:
```console
python3 gen_judgment.py
```
Judgment caching is also implemented. It will skip generating judgments that has already been generated or lacks one of the model answers.  

### Step 4. Show result
Output model win rates.  Optionally, use `--full-stats` for detailed results.
```console
> python3 show_result.py --judge-name Llama-3-70B-Instruct-AH-AWQ --baseline Phi-3-medium-128k-instruct
```
### Step 5. Arena Hard UI
You can review individual judgment results using our UI code.
```console
> python3 qa_broswer.py --share
```


