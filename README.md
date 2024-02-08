# llm-meta-analysis
Automating meta-analysis of clinical trials (randomized controlled trials)

## SETUP

Create conda environment from the environment.yml: `conda env create -f environment.yml`

Activate the conda environment: `conda activate llm-meta-analysis`

Install additional packages from source. This is needed for running open-source models:

```
pip install flash-attn --no-build-isolation
pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
```
## EXPERIMENTS

### Running Tasks

### Running Evaluation/Calculating Metrics

## CITATION
