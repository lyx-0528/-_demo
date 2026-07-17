# Causal TTL Experiment Scaffold

This directory now contains two layers:

1. A runnable reference implementation for the PPT method.
2. A local-first experiment scaffold so you can prepare everything on your own
   machine before moving the real SLM runs to cloud GPUs.

The design target is:
- local: data preparation, dry runs, config validation, artifact layout
- cloud: real SLM fine-tuning and large-scale inference

## What Already Works

The reference pipeline still maps the PPT into three stages:

1. `Stage 1: multi-path reasoning`
   - `MockEdgeReasoner.generate_reasoning_paths(...)`
2. `Stage 2: cluster + cloud diagnosis`
   - `cluster_reasoning_paths(...)`
   - `HeuristicCloudDiagnoser.diagnose(...)`
3. `Stage 3: front-door style TTL objective`
   - `evaluate_interventions(...)`
   - compares treated vs control interventions

The current implementation is still a mock pipeline, but it is useful for:
- validating prompts and data flow
- validating experiment configs
- validating output formats
- validating cloud run scripts before renting GPUs

## Why This Helps With MedThink-Style Work

`MedThink/PrecisionBoost` already gives us:
- a teacher-guided workflow
- usable medical QA formats
- a clear "student answer -> teacher judge" pattern

This scaffold extends that into:
- reusable experiment configs
- normalized local datasets
- repeatable dry runs
- cloud bootstrap scripts

## Directory Layout

```text
causal_ttl/
  causal_path.py
  config.py
  data.py
  evaluation.py
  exports/
    config.py
    frontdoor.py
    records.py
    runner.py
  frontdoor/
    clustering.py
    diagnosers.py
    interfaces.py
    interventions.py
    reasoners.py
    workflow.py
  inference/
    config.py
    generate.py
    prompts.py
    runner.py
    runtime.py
  knowledge_tokens.py
  schema.py
  teachers/
    base.py
    cache.py
    factory.py
    heuristic.py
    openai_api.py
    parsing.py
  text_utils.py
  training/
    config.py
    dataset.py
    runner.py
    runtime.py
    trainer.py
  adapters.py
  clustering.py
  hf_inference.py
  hf_training.py
  pipeline.py
  supervision.py
  teacher.py
  ttl.py
configs/
  qa_mock.yaml
  digestive_mock.yaml
scripts/
  bootstrap_autodl.sh
  run_autodl_experiment.sh
  run_local_dry_run.ps1
.env.example
requirements.txt
prepare_dataset.py
baselines.py
metrics.py
experiment_runner.py
export_supervision.py
train_causal_ttl.py
predict_causal_ttl.py
evaluate_causal_ttl.py
run_causal_workflow.py
run_demo.py
```

Top-level files such as `teacher.py`, `hf_training.py`, `hf_inference.py`, `supervision.py`, `pipeline.py`, and `adapters.py` are now compatibility shims. The main implementations live in the new subpackages.

## Local Workflow

### 1. Install dependencies

```powershell
python .\因果推断\prepare_dataset.py --help
python -m pip install -r .\因果推断\requirements.txt
```

### 2. Normalize data locally

Single dataset split:

```powershell
python .\因果推断\prepare_dataset.py `
  --input .\MedThink\PrecisionBoost\q&a.csv `
  --output-dir .\因果推断\data\processed\qa
```

Explicit train and test inputs:

```powershell
python .\因果推断\prepare_dataset.py `
  --train-input .\MedThink\PrecisionBoost\data\digestive_Qwen2_72b_Instruction_train.json `
  --test-input .\MedThink\PrecisionBoost\data\digestive_Qwen2_72b_Instruction_test.json `
  --output-dir .\因果推断\data\processed\digestive
```

### 3. Run a local dry run

```powershell
python .\因果推断\experiment_runner.py --config .\因果推断\configs\qa_mock.yaml
python .\因果推断\experiment_runner.py --config .\因果推断\configs\digestive_mock.yaml
```

These dry runs use mock reasoning so you can check:
- dataset loading
- config parsing
- artifact output
- end-to-end control flow

## Cloud Workflow

After the local dry run works, move the project to the cloud machine and run:

```bash
bash scripts/bootstrap_autodl.sh /root/project/causal_ttl_project
bash scripts/run_autodl_experiment.sh /root/project/causal_ttl_project /root/project/causal_ttl_project/configs/digestive_mock.yaml
```

The intended future replacement is:
- local mock baseline -> real cloud SLM baseline
- heuristic diagnoser -> real teacher API or large teacher model

## What Is Ready Vs Stubbed

Ready now:
- dataset loading
- local dataset normalization
- config-driven dry runs
- baseline registry
- metric export to JSON and CSV
- cloud bootstrap script

Stubbed by design:
- real SLM generation
- real cloud LLM diagnosis
- stage-1 / stage-2 LoRA fine-tuning
- online serving deployment

## Next Replacement Points

To move from local prep to real experiments, replace:

- `MockEdgeReasoner`
  with a real student model adapter
- `HeuristicCloudDiagnoser`
  with a teacher API or large teacher model adapter
- mock baselines
  with paper baselines such as `OneShot Distill`, `AnswerFix`, and `MedThink`

## CloudDeviceTTL Migration

This project now includes a first-stage migration path for `CloudDeviceTTL`-style causal TTL:

- structured teacher payloads (`causal_factors`, `confounders`, `steps`, `facts`, `counterfactuals`, `answer`)
- explicit `<knowledge>` markers and causal path formatting
- export of `answer`, `counterfactual`, and `policy` supervision records without depending on LLaMA-Factory
- optional front-door analysis traces from the PPT pipeline (`multi-path -> cluster -> diagnose -> intervene`)
- weighted training that can use both task type weights and sample-level front-door weights

The export presets map to the CloudDeviceTTL variants like this:

- `fact`: similar to `causal_collaborative_ttl_fact`
- `v6`: similar to `causal_collaborative_ttl_v6`
- `v7`: similar to `causal_collaborative_ttl_v7`

### Export structured supervision locally

Use the heuristic teacher for a dry run:

```powershell
python .\因果推断\export_supervision.py `
  --dataset .\MedThink\PrecisionBoost\q&a.csv `
  --limit 10 `
  --mode v7 `
  --teacher-backend heuristic `
  --output-dir .\因果推断\outputs\supervision_v7
```

Enable the PPT front-door analysis while exporting:

```powershell
python .\因果推断\export_supervision.py `
  --dataset .\MedThink\PrecisionBoost\q&a.csv `
  --limit 10 `
  --mode v7 `
  --teacher-backend heuristic `
  --enable-frontdoor `
  --output-dir .\因果推断\outputs\supervision_v7_frontdoor
```

Artifacts written by this command:

- `supervision_dataset.jsonl`: combined answer / counterfactual / policy records
- `answer_records.jsonl`
- `counterfactual_records.jsonl`
- `policy_records.jsonl`
- `teacher_payloads.jsonl`
- `frontdoor_traces.jsonl` when `--enable-frontdoor` is used
- `supervision_summary.json`

### Switch to a real API teacher

The same exporter can use an OpenAI-compatible teacher endpoint:

```powershell
python .\因果推断\export_supervision.py `
  --dataset .\MedThink\PrecisionBoost\q&a.csv `
  --mode v7 `
  --teacher-backend api `
  --teacher-api-base $env:TEACHER_API_BASE `
  --teacher-model $env:TEACHER_MODEL `
  --output-dir .\因果推断\outputs\supervision_api_v7
```

This keeps your current project as the main workspace while pulling over the most reusable parts of the CloudDeviceTTL algorithm.

## Local Train / Predict / Evaluate

The project now includes a minimal local-first execution workflow so `因果推断` is no longer limited to mock dry runs.

### 1. Train on exported supervision records

```powershell
python .\因果推断\train_causal_ttl.py `
  --train-records .\因果推断\outputs\supervision_v7\supervision_dataset.jsonl `
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct `
  --output-dir .\因果推断\outputs\model_v7 `
  --task-types answer counterfactual policy `
  --answer-loss-weight 1.0 `
  --counterfactual-loss-weight 1.0 `
  --policy-loss-weight 1.0
```

If the exported records include `sample_weight` from front-door analysis, the trainer will
use them by default. Add `--no-sample-weight` to disable that behavior.

### 2. Predict on a prepared test split

```powershell
python .\因果推断\predict_causal_ttl.py `
  --dataset .\因果推断\data\processed\qa_test\test.json `
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct `
  --adapter-path .\因果推断\outputs\model_v7 `
  --mode v7 `
  --teacher-backend heuristic `
  --predictions-path .\因果推断\outputs\model_v7\predictions.jsonl
```

### 3. Evaluate predictions

```powershell
python .\因果推断\evaluate_causal_ttl.py `
  --predictions .\因果推断\outputs\model_v7\predictions.jsonl
```

### 4. Run the full workflow from one config

Example configs:

- [configs/causal_ttl_v7_workflow.yaml](/D:/25-26Spr/科研资料/逻辑链/因果推断/configs/causal_ttl_v7_workflow.yaml)
- [configs/causal_ttl_fact_workflow.yaml](/D:/25-26Spr/科研资料/逻辑链/因果推断/configs/causal_ttl_fact_workflow.yaml)

Run end to end:

```powershell
python .\因果推断\run_causal_workflow.py --config .\因果推断\configs\causal_ttl_v7_workflow.yaml
```

This local workflow gives your project the same broad structure as CloudDeviceTTL:

- teacher-guided supervision export
- LoRA training on answer / counterfactual / policy samples
- direct or interactive inference
- evaluation on saved predictions

It also now supports the PPT method more directly:

- front-door style multi-path diagnosis is exported into `frontdoor_traces.jsonl`
- each supervision record can carry `frontdoor_*` summary fields and `sample_weight`
- training can combine task-level weights with those sample-level front-door weights
