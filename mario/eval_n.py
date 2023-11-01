import os
import csv
import torch
import json
import time
import random
import logging
import random
import argparse
import pickle
import numpy as np
from typing import List
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dython.nominal import correlation_ratio

from arguments import get_args
from main_n_new import load_model
from policy_n import Policy
from main_n_new import PromptDataset, PromptCollator_WithGold
from reward import Reward, RevReward
from utils.utils import load_jsonl, ensure_dir, batchify
from flanT5_scores import FlanT5Processor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_control_code(input_ids, num_control_codes, best_control_code=None,
    random_control_codes_list=None, random_tags_for_eval="no"):
    if num_control_codes > 0:
        if random_tags_for_eval == "no":
            return input_ids.new(best_control_code * len(input_ids)).reshape(-1, num_control_codes)
        else:
            random_control_code = np.zeros((len(input_ids), num_control_codes))
            for i in range(len(input_ids)):
                for j in range(num_control_codes):
                    random_control_code[i][j] = random.choice(random_control_codes_list[j])
            print("random_control_code:", random_control_code[:5, :])
            return input_ids.new(random_control_code)
    else:
        return None

def get_accuracy(pred_responses, gold_responses):
    correct = 0
    correctness = []
    for i in range(len(pred_responses)):
        gold_label = gold_responses[i].split('So the answer is')[1].strip().strip('.').lower()
        split_text_pred = pred_responses[i].split('So the answer is')
        if len(split_text_pred) > 1:
            pred_label = split_text_pred[1].strip().strip('.').lower()
        else:
            pred_label = ''

        if pred_label == gold_label:
            correct += 1
            correctness.append(1)
        else:
            correctness.append(0)

    accuracy = float(correct)/len(pred_responses)
    return accuracy*100, correctness

def get_pred_labels(pred_responses):
    pred_labels = []
    for i in range(len(pred_responses)):
        split_text_pred = pred_responses[i].split('So the answer is')
        if len(split_text_pred) > 1:
            pred_label = split_text_pred[1].strip().strip('.').lower()
        else:
            pred_label = ''
        pred_labels.append(pred_label)

    return pred_labels

def main():
    args = get_args()

    # seeds and cuda
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    num_gpus = torch.cuda.device_count()
    print(f'Detect {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # save dir and checkpoints
    save_dir = args.save_dir_for_eval
    args.reward_dir = os.path.join(save_dir, 'reward')
    args.model_dir = os.path.join(save_dir, 'model')
    ckp_num = args.ckp_num_for_eval
    if ckp_num == "all":
        ckp_list = [0]
        tmp = os.listdir(args.model_dir)
        ckp_list += [int(x.replace("ckp_","").replace(".pth","")) for x in tmp]
    else:
        ckp_list = [int(x) for x in ckp_num.split(',')]
    ckp_list = sorted(ckp_list)

    # reward tokens
    args.n_extra_tokens_list = [int(x) for x in args.n_extra_tokens.split(',')]
    args.rewards_list = args.reward_name.split(',')
    args.reward_filter_scores_list = [float(x) for x in args.reward_filter_score.split(',')]
    train_option = args.train_option.split(',')
    print("train option:", args.train_option)
    assert len(args.rewards_list)==len(args.n_extra_tokens_list)
    assert len(args.rewards_list)==len(args.reward_filter_scores_list)
    assert len(args.rewards_list)==len(train_option)

    tree_tokens = {}
    score_model = {}
    reward_filter_scores = {}
    args.train_option = {}
    flant5_model_flag = 0
    flant5_model = None
    for i in range(len(args.n_extra_tokens_list)):
        if args.rewards_list[i] == "accuracy":
            # this is always calculated, don't need a reward calculator for it 
            # also don't need tree_tokens because task_correctness_tokens below takes care of it
            continue        
        ct_tree_tokens = ['_TOKEN'+str(i)+'_{}'.format(str(idx).zfill(5)) for idx in range(args.n_extra_tokens_list[i])]
        if 0: #int(args.save_number) < 161: # removr later !!!!
            ct_tree_tokens = ct_tree_tokens + ['_TOKEN'+str(i)+'_ZERO_COMMENTS']
        tree_tokens[args.rewards_list[i]] = ct_tree_tokens
        # if args.rewards_list[i] in ['flan-t5-factuality', 'flan-t5-factuality-full', 'flan-t5-factuality-w-qn', 'flan-t5-completeness'] and flant5_model_flag==0:
        if 'flan-t5' in args.rewards_list[i] and flant5_model_flag==0:  
            # since flan-t5 is so big, the same flan-t5 model is used for all flan-t5 based rewards
            # to optimize gpu's used
            # for now, I assume all flan-t5 based rewards use the same size of flan-t5
            flant5_model = FlanT5Processor(args.flan_t5_size)
            flant5_model_flag = 1

        if args.rewards_list[i] == "rev":
            ct_reward = RevReward(dataset_name=args.dataset_name, baseline_rat_model_path=args.baseline_rat_model_path, \
                        evaluator_model_rb_path=args.evaluator_model_rb_path, evaluator_model_b_path=args.evaluator_model_b_path, \
                        save_path=args.reward_dir, batch_size=args.reward_batch_size, device=num_gpus - 1)
        else:
            ct_reward = Reward(save_path=args.reward_dir, batch_size=args.reward_batch_size,
                            reward_name=args.rewards_list[i], device=num_gpus - 1, flant5_model=flant5_model)
        score_model[args.rewards_list[i]] = ct_reward
        reward_filter_scores[args.rewards_list[i]] = args.reward_filter_scores_list[i]
        args.train_option[args.rewards_list[i]] = train_option[i]
    task_correctness_tokens = ['WRONG_LABEL', 'CORRECT_LABEL']

    if args.reward_as_product == 1:
        tree_tokens = {}
        tree_tokens['reward_product'] = ['_TOKEN'+str(i)+'_{}'.format(str(idx).zfill(5)) for idx in range(args.n_extra_tokens_list[0])]

    # new rewards if any
    if len(args.reward_names_other) > 0:
        other_rewards_list = args.reward_names_other.split(',')
        for i in range(len(other_rewards_list)):
            # if other_rewards_list[i] in ['flan-t5-factuality', 'flan-t5-factuality-full', 'flan-t5-factuality-w-qn', 'flan-t5-completeness'] and flant5_model_flag==0:
            if 'flan-t5' in other_rewards_list[i] and flant5_model_flag==0:
                # since flan-t5 is so big, the same flan-t5 model is used for all flan-t5 based rewards
                # to optimize gpu's used
                # for now, I assume all flan-t5 based rewards use the same size of flan-t5
                flant5_model = FlanT5Processor(args.flan_t5_size)
                flant5_model_flag = 1
            if other_rewards_list[i] == "rev":
                ct_reward = RevReward(dataset_name=args.dataset_name, baseline_rat_model_path=args.baseline_rat_model_path, \
                            evaluator_model_rb_path=args.evaluator_model_rb_path, evaluator_model_b_path=args.evaluator_model_b_path, \
                            save_path=args.reward_dir, batch_size=args.reward_batch_size, device=num_gpus - 1)
            else:
                ct_reward = Reward(save_path=args.reward_dir, batch_size=args.reward_batch_size,
                                reward_name=other_rewards_list[i], device=num_gpus - 1, flant5_model=flant5_model)
            score_model[other_rewards_list[i]] = ct_reward

    if args.actual_rewards_to_calculate == '':
        actual_rewards_to_calculate = list(score_model.keys())
    else:
        actual_rewards_to_calculate = args.actual_rewards_to_calculate.split(',')

    print(f'Initializing models ...')
    #model, tokenizer = load_model(args.init_model)
    model, tokenizer = load_model(args.ref_model)
    ref_policy = Policy(model=deepcopy(model), tokenizer=deepcopy(tokenizer), temperature=args.temperature, device=device)
    policy = Policy(model=model, tokenizer=tokenizer, temperature=args.temperature, device=device,
                    reward_cond=True, tree_tokens=tree_tokens, task_correctness_tokens=task_correctness_tokens)

    print('Loading data..')
    prompt_collator_with_gold = PromptCollator_WithGold(tokenizer=policy.tokenizer)
    val_dataset = PromptDataset(args.dataset_val, args.dataset_name)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=prompt_collator_with_gold)
    print(f'Load val set with {len(val_dataset)} examples\n\n')

    test_dataset = PromptDataset(args.dataset_test, args.dataset_name)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=prompt_collator_with_gold)
    print(f'Load test set with {len(test_dataset)} examples\n\n')

    # best control code
    best_cat = {}
    best_cat_id = {}
    num_control_codes = 0
    best_control_code = []
    if args.reward_as_product:
        rew = 'reward_product'
        num_control_codes = 1
        best_cat[rew] = tree_tokens[rew][0]
        best_cat_id[rew] = policy.tokenizer.convert_tokens_to_ids(best_cat[rew])
        best_control_code = [best_cat_id[rew]]
    else:
        for rew in tree_tokens.keys():
            if args.train_option[rew] in ['quark', 'filter_quark']: # add control code only for this case
                if rew == "accuracy":
                    best_cat[rew] = task_correctness_tokens[-1]
                    best_cat_id[rew] = policy.tokenizer.convert_tokens_to_ids(best_tc)
                else:
                    best_cat[rew] = tree_tokens[rew][0]
                    best_cat_id[rew] = policy.tokenizer.convert_tokens_to_ids(best_cat[rew])
                num_control_codes += 1
                best_control_code.append(best_cat_id[rew]) 
        if args.task_correctness == 'yes':
            best_tc = task_correctness_tokens[-1]
            best_tc_id = policy.tokenizer.convert_tokens_to_ids(best_tc)
            num_control_codes += 1
            best_control_code.append(best_tc_id)
    print("best control code:", best_control_code)

    # random control code
    rand_cat = {}
    rand_cat_id = {}
    random_control_codes_list = []
    for rew in tree_tokens.keys():
        if rew=='reward_product':
            args.train_option[rew] = 'quark' # temporary fix
        if args.train_option[rew] in ['quark', 'filter_quark']: # add control code only for this case
            if rew == "accuracy":
                rand_cat[rew] = deepcopy(task_correctness_tokens)
                rand_cat_id[rew] = [policy.tokenizer.convert_tokens_to_ids(ct_rand_cat) for ct_rand_cat in rand_cat[rew]]
            else:                
                rand_cat[rew] = tree_tokens[rew]
                rand_cat_id[rew] = [policy.tokenizer.convert_tokens_to_ids(ct_rand_cat) for ct_rand_cat in rand_cat[rew]]
            random_control_codes_list.append(rand_cat_id[rew])
    if args.task_correctness == 'yes':
        rand_tc = deepcopy(task_correctness_tokens)
        rand_tc_id = [policy.tokenizer.convert_tokens_to_ids(ct_rand_tc) for ct_rand_tc in rand_tc]
        random_control_codes_list.append(rand_tc_id)
    print("random control codes list:", random_control_codes_list)

    batch_size = 64

    # need to evaluate all checkpoints in list
    for ckp_num in ckp_list:
        if ckp_num != 0:
            ckp_num_path = os.path.join(args.model_dir, 'ckp_' + str(ckp_num) + '.pth')
            checkpoint = torch.load(ckp_num_path, map_location='cpu')
            policy.model.load_state_dict(checkpoint['policy_model'])
            print("----------------------------")
            print("LOADED CHECKPOINT:", ckp_num)
            print("----------------------------")

        for eval_type in ['val', 'test']:
            if eval_type == 'val':
                ct_dataloader = val_dataloader
            else:
                ct_dataloader = test_dataloader

            # greedy decoding
            prompts, responses, responses_gold = [], [], []
            num_sents, num_words = [], []
            for i, (ct_queries, input_ids, attention_mask, response_gold, gold_label_sequence) in enumerate(tqdm(ct_dataloader)):
                with torch.no_grad():
                    if ckp_num == 0:
                        rollouts = ref_policy.sample(step=ckp_num, input_ids=input_ids, attention_mask=attention_mask,
                                                    max_len=args.decoding_len,
                                                    do_sample=False)
                    else:
                        rollouts = policy.sample(step=ckp_num, input_ids=input_ids, attention_mask=attention_mask,
                                                max_len=args.decoding_len,
                                                control_code=get_control_code(input_ids, num_control_codes,
                                                    best_control_code, random_control_codes_list,
                                                    args.random_tags_for_eval),
                                                do_sample=False,
                                                training_in_stages=args.training_in_stages,
                                                training_in_stages_mode=args.training_in_stages_mode,
                                                stages_interval=args.stages_interval)
                    forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                                      'query_mask': rollouts['query/mask'],
                                      'response_input_ids': rollouts['response/input_ids'],
                                      'response_mask': rollouts['response/mask']}

                    prompt, response = rollouts['query/text'], rollouts['response/text']
                    prompts.extend(ct_queries)#prompt)
                    responses.extend(response)
                    responses_gold.extend(response_gold)
                    num_sents.extend([x.count('.') for x in response])
                    num_words.extend([len(x.split(' ')) for x in response])

            accuracy, correctness_list = get_accuracy(responses, responses_gold)

            # to save the individual datapoints
            to_save_dict = {'question': prompts,
                            'model_response': responses,
                            'correctness': correctness_list,
                            'num_sents': num_sents,
                            'num_words': num_words}

            scores = {}
            reward_scores_str = ''
            for rew in actual_rewards_to_calculate: #score_model.keys():
                if rew == "accuracy":
                    continue # correctness list is already calculated
                scores[rew] = score_model[rew].get_reward(prompts, responses, responses_gold, f'eval_eval_' + eval_type + 'greedy', split=eval_type)
                to_save_dict[rew] = scores[rew]
                reward_score = np.nanmean([s for s in scores[rew] if isinstance(s, float)])
                #reward_scores_str += rew + ': ' + str(round(reward_score,2)) + ' ' 
                reward_scores_str += rew.replace('flan-t5-', '').replace('roscoe-', '') + ': ' + str(round(reward_score,3)) + ' '
                # correlation
                scores_for_correlation = [s if isinstance(s, float) else np.nan for s in scores[rew]]
                calculated_correlation_score = correlation_ratio(correctness_list, scores_for_correlation,
                                                                    nan_strategy='drop')
                #reward_scores_str += 'corr: ' + str(round(calculated_correlation_score,2)) + ' '
                print(eval_type + " " + rew + " greedy reward = ", reward_score, " correlation w. accuracy = ",
                        calculated_correlation_score)
            reward_scores_str += 'num_sents' + ': ' + str(round(np.mean(num_sents),2)) + ' '
            reward_scores_str += 'num_words' + ': ' + str(round(np.mean(num_words),2)) + ' '
            print(eval_type + " greedy accuracy = ", accuracy)
            print(eval_type + " greedy num_sents = ", np.mean(num_sents))
            print(eval_type + " greedy num_words = ", np.mean(num_words))

            # saving the scores
            eval_mode = ''
            if args.random_tags_for_eval == "yes":
                eval_mode = 'random_'
            with open(os.path.join(args.reward_dir, eval_mode + 'eval_reward_scores_' + eval_type + '_greedy.txt'), 'a') as f:
                f.write('Step ' + str(ckp_num)+' - '+ reward_scores_str + ' Accuracy: ' + str(round(accuracy,2)) + '\n')

            # saving the individual datapoints in csv
            with open(os.path.join(args.reward_dir,'eval_output' + eval_type + '_greedy_' + str(ckp_num) + '.csv'), 'w') as csv_f:
                writer = csv.writer(csv_f)
                key_list = list(to_save_dict.keys())
                limit = len(to_save_dict[key_list[0]])
                
                writer.writerow(key_list)
                for lt in range(limit):
                    writer.writerow([to_save_dict[keykey][lt] for keykey in key_list])

if __name__ == "__main__":
    main()


