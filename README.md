# kolmev
Evaluation for korean language models (e.g. bert, roberta, bart, t5, gpt2...)
## How to setup the environment?
```bash
# cuda 11.3.x
pip install .
```
## How to use?
1. model과 task에 맞는 softlink를 project root에 만듭니다.

```bash
python main.py --model gpt2 --task nsmc
```
```bash
# softlink
nsmc.py -> ${PROJECT_PATH}/src/kolmev/gpt/nsmc.py
```

2. 환경에 맞게 실행합니다.
```bash
python nsmc.py --help
```
```bash
torchrun --nnodes 1 --nproc_per_node 8 nsmc.py --pretrained_model_name_or_path ${PRETRAINED_MODEL_NAME_OR_PATH}
```
