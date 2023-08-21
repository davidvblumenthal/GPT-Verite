# GPT-Verite
This is the main repository of the master thesis: GPT-Vérité - A model for trustworthy and flexible text generation. All code used is contained in this repository. It is essentially a agregated self contained version of many individual sub-repositories used across this thesis. In the section below, I provide links to the individual GitHub repositories contained here.

## Artefacts
Here are all the links to the created artefacts such as training dataset, model checkpoints etc.
### Data
[Complete Training Dataset (LesFaits)](https://huggingface.co/datasets/davidvblumenthal/LesFaits)

[Subset used for training](https://doi.org/10.5281/zenodo.8242895)

### Model Weights
[Weights including optimizer states](https://doi.org/10.5281/zenodo.8242804)

[HuggingFace versions](https://huggingface.co/davidvblumenthal)

## Training related
This part links all code repositories related to training models.
### Main Training Repository
[Folder in this repository](training/continued_pretraining)
[Link to GitHub repository](https://github.com/davidvblumenthal/gpt-verite_)

### HuggingFace related Trainings
[Folder in this repository](training/scratch_training)
[Link to GitHub repository](https://github.com/davidvblumenthal/Truthfulness_Study_Hug)

## Dataset Creation
[Folder in this repository](dataset/LesFaits)
[Link to GitHub repository](https://github.com/davidvblumenthal/LesFaits)

## Evaluation related
### FACTUALITYPROMPTS
#### Generate the continuation to the prompts
[Folder in this repository](evaluation/generation_evaluation)
[Link to GitHub repository](https://github.com/davidvblumenthal/generation_evaluation)

#### Evaluate the Generations
[Folder in this repository](evaluation/FactualityPrompts)
[Link to GitHub repository](https://github.com/davidvblumenthal/FactualityPrompt)

### TruthfulQA and Downstream Task Evaluation
[Folder in this repository](evaluation/lm-evaluation-harness)
[Link to GitHub repository](https://github.com/davidvblumenthal/lm-evaluation-harness)

### Real Toxicity Prompts
[Folder in this repository](evaluation/toxicity_prompts)
[Link to GitHub repository](https://github.com/davidvblumenthal/toxicity_prompts)
