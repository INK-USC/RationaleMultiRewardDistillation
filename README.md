# Tailoring Self-Rationalizers with Multi-Reward Distillation

This is the code and associated datasets for the paper titled 

>[Tailoring Self-Rationalizers with Multi-Reward Distillation. *Sahana Ramnath, Brihi Joshi, Skyler Hallinan, Ximing Lu, Liunian Harold Li, Aaron Chan, Jack Hessel, Yejin Choi, Xiang Ren.*](https://openreview.net/forum?id=t8eO0CiZJV)

Project website: [inklab.usc.edu/MaRio](https://inklab.usc.edu/MaRio/)

<img src="https://github.com/INK-USC/RationaleMultiRewardDistillation/assets/17588365/ca7ec674-dd5f-4e4d-a33b-9bf4cd935a0a" width=700>

[README in-progress]

## Dataset
We provide our train/val/test data splits in folder ```data/[dataset-name]/raw```.
The subfolders ```gpt3```, ```llama``` and ```flant5``` in ```data/[dataset-name]``` have the sampled test-set responses from GPT-3, LLaMa 7B/65B and FLAN-T5-L/XL/XXL for all datasets, as well as the sampled train-set responses from GPT-3 which we use as silver-standard training data.

## Training commands 
### The base model (```SFT```)
``python scripts/train_new.py --dataset-name=numersense \
--dataset-train="data/numersense/raw/train.jsonl" \
--dataset-val="data/numersense/raw/dev.jsonl" --dataset-test="data/numersense/raw/test.jsonl" \
--num-epochs=25 --batch_size=4 --save_number=1 --output-dir="save/"``

### Reward model for ```consistency```
I2O model:
``python train_rq_models.py \
--dataset-name=numersense --model-type="t5-base" --eval-interval=1000 \
--dataset-train="data/numersense/raw/train.jsonl" --dataset-val="data/numersense/raw/dev.jsonl" \
--dataset-test="data/numersense/raw/test.jsonl" --num-epochs=10 --batch_size=4 \
--if_warmup=0 --gen-mode=i2o --use_demonstrations=0 \
--save_number=1``

IR2O model:
``python train_rq_models.py \
--dataset-name=numersense --model-type="t5-base" --eval-interval=7000 \
--dataset-train="data/numersense/raw/train.jsonl" --dataset-val="data/numersense/raw/dev.jsonl" \
--dataset-test="data/numersense/raw/test.jsonl" --num-epochs=5 --batch_size=4 \
--if_warmup=0 --gen-mode=ir2o --use_demonstrations=0 \
--save_number=2``

### Training ```MaRio```
``python main_n_new.py --save_number=numersense_1 --dataset-name=numersense --dataset-train="data/numersense/raw/train.jsonl" --dataset-val="data/numersense/raw/dev.jsonl" --dataset-test="data/numersense/raw/test.jsonl" --ref-model="path-to-SFT" --lr=3e-5 --task_correctness=no --sample-interval=4000 --batch_size=4 --max-grad-norm=1.0 --top_p=0.7 --kl_coef=0.1 --entropy_coef=0.0 --reward_name="numersense-rationale-quality-continuous,accuracy,plausibility,diversity" --n_extra_tokens="5,5,5,5" --train_option="quark,quark,quark,quark" --reward_filter_score="-1,-1,-1,-1" --total-steps=50000 --num_policy_rollout=1 --task_loss_coef=0.0 --output-dir="save"  --step_to_start_sampling=0 --expert_all_at_first=1``

The validation and test set reward / accuracy scores will be saved to ``path-to-model-dir/reward/reward_scores_[val-or-test]_greedy.txt``, and the predicted rationales will be saved to ``path-to-model-dir/reward/eval_output[val-or-test]_greedy_[ckp-num].jsonl``.

## Acknowledgements
This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022-22072200006. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.
