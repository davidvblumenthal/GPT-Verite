# FactualityPrompt
![License: Apache](https://img.shields.io/badge/License-Apache2.0-yellow.svg) 
  
This repository contains the test prompts and evaluation pipeline for the improved version of FactualityPrompts. 


## Code Overview
* `fever_athene`: contains fact-checking pipeline code (Wiki document retriever, Wiki sentence selector, etc) from UKPLab/fever-2018-team-athene [github](UKPLab/fever-2018-team-athene). We utilize and build on top of their Wiki document retriever in our work. (Refer to their github for citation details)
* `prompts`: contains our FactualityPrompt testset utilized in our paper.
* `src`: codes for evaluating the factualtiy of LM generation (For files adapted from other publicly available codebases, we included the pointer to the original code file)

## 1. Setup 
1. Install dependencies by running `pip install -r requirements.txt`
2. Download Wikipedia processed dump (knowledgesource.json) from [KILT-github](https://github.com/facebookresearch/KILT#kilt-knowledge-source) into `data` directory (Refer to their repository for citation details)
```bash
  mkdir data
  cd data
  wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
```
3. Create the DB file from Wikipedia dump by running:

```bash
  PYTHONPATH=fever_athene python3 fever_athene/scripts/build_db_kilt.py data/knowledgesource.json data/kilt_db.db
```
This script will create kilt_db.db into `data` directory. 

4. Configure `src/const.py` file. 

## 2. Run evaluation script
Main [Evaluation Script](run_parallel.sh)

### Factuality Metrics

```bash
PYTHONPATH=. python src/auto_evaluation.py --prompt_type ${PROMPT_TYPE} --gen_path ${GEN_TO_EVALUATE_NAME} \
        --gen_dir ${GEN_FOLDER} \
        --use_additional_ne \
        --dedub_generation \
        --entailment_model large_mnli
```

### Repetition

```bash
for PROMPT_TYPE in factual nonfactual
do
    GEN_TO_EVALUATE_NAME=${PROMPT_TYPE}-CUSTOM-GEN-NAME.jsonl
    python src/repetition.py ${GEN_TO_EVALUATE_NAME}  --final
done
``` 

