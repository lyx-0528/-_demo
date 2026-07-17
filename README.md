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
  adapters.py
  clustering.py
  config.py
  data.py
  pipeline.py
  schema.py
  text_utils.py
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
run_demo.py
```

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
