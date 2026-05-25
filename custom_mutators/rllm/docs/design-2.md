# CovRL-Fuzz

This is the design document for the implementation of CovRL-Fuzz(ref;CovRL-Fuzz) in AFL++.

We divide CovRL-Fuzz into three seperate components; data/model/training. This leaves the implementation extendable for alternative data processing, different models, and different training algorithms.

In our work we extend CovRL-Fuzz with a reinforcement learning policy gradient algorithm called Group Relative Policy Optimization (GRPO) implementing it as its own trainer



### Preprocessing & preperation of run
- data/preprocessing.py - *preprocess the seed queue*
- data/validity.py - *Extracting validity of programs*
- config.py - *Configuring fuzzing campaign*
    - configs/default.json 
- run_rllm.sh - *run CovRL-Fuzz standalone*

### Main files needed for run
- rllm.py - *AFL++ API shim*
- mutator.py - *Orchistrator of fuzzing campaign*
- data/masking.py - *Masking strategy*
- model/llm.py - *Model serving mutator*
- model/tokenizer.py - *Model tokenization for mutator*

### Trainers used to finetune language model during run

training/covrl_trainer.py
model/critic.py

train/rewarding.py
exit_hook.c + exit_hook.so + Makefile - *Capture validity during fuzzing*
train/rollout.py
