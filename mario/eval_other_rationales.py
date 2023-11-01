# to evaluate rationales obtained from other sources/models
# this code is just reward evaluation

import os
import csv
import torch
import json
import time
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
from main_n import load_model
from policy_n import Policy
from main_n import PromptDataset, PromptCollator_WithGold
from reward import Reward, RevReward
from utils.utils import load_jsonl, ensure_dir, batchify
from flanT5_scores import FlanT5Processor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_accuracy(pred_labels, gold_labels):
    correct = 0
    correctness = []
    for i in range(len(pred_labels)):
        gold_label = gold_labels[i]
        pred_label = pred_labels[i]

        if pred_label == gold_label:
            correct += 1
            correctness.append(1)
        else:
            correctness.append(0)

    accuracy = float(correct)/len(pred_labels)
    return accuracy*100, correctness

def main():
    args = get_args()

    num_gpus = torch.cuda.device_count()
    print(f'Detect {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # save dir and checkpoints
    save_dir = args.save_dir_for_eval
    args.reward_dir = os.path.join(save_dir, args.dataset_name, 'reward')
    for d in [save_dir, args.reward_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # reward tokens
    args.rewards_list = args.reward_name.split(',')

    score_model = {}
    flant5_model_flag = 0
    flant5_model = None
    for i in range(len(args.rewards_list)):
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
    task_correctness_tokens = ['WRONG_LABEL', 'CORRECT_LABEL']

    # new rewards if any
    if len(args.reward_names_other) > 0:
        other_rewards_list = args.reward_names_other.split(',')
        for i in range(len(other_rewards_list)):
            ct_reward = Reward(save_path=args.reward_dir, batch_size=args.reward_batch_size,
                            reward_name=other_rewards_list[i], device=num_gpus - 1)
            score_model[other_rewards_list[i]] = ct_reward

    # here is where I load the gold data
    gold_data_loaded = pd.read_json(args.outputs_other_sources_gold_file, orient='records')
    if args.dataset_name != 'strategyqa':
        gold_data_loaded = gold_data_loaded.T
    else:
        gold_classes = {0 : "no", 1 : "yes"}
        if "train" in args.outputs_other_sources_gold_file:
            gold_data_loaded = gold_data_loaded.T
    gold_q_and_label = {}
    gold_q_and_rationale = {}
    for i in range(len(gold_data_loaded)):
        ct_gold_question = gold_data_loaded.iloc[i]['question']
        ct_gold_label = gold_data_loaded.iloc[i]['answer']
        ct_gold_rationale = gold_data_loaded.iloc[i]['rationale']
        if args.dataset_name == 'strategyqa' and "train" not in args.outputs_other_sources_gold_file:
            ct_gold_label = gold_classes[ct_gold_label]
        if args.dataset_name not in ['quarel', 'qasc']:
            ct_gold_label = ct_gold_label.lower()
        gold_q_and_label[ct_gold_question] = ct_gold_label
        gold_q_and_rationale[ct_gold_question] = ct_gold_rationale

    # here is where I load the new files
    predicted_outputs = []
    predicted_labels = []
    gold_labels = []
    predicted_rationales = []
    questions = []
    num_sents = []
    num_words = []
    gold_responses = []
    if args.outputs_other_sources_type == "jsonl":
        data_loaded = pd.read_json(args.outputs_other_sources_file, orient='records')
        #if args.dataset_name!= "strategyqa":
        data_loaded = data_loaded.T
        for i in range(len(data_loaded)):
            if data_loaded.iloc[i]['question'] not in gold_q_and_rationale:
                continue
            questions.append(data_loaded.iloc[i]['question'])
            gold_label = data_loaded.iloc[i]['gold_label']
            if args.dataset_name in ["quarel", "qasc", "numersense"]:
                gold_label = gold_label.upper()
            gold_labels.append(gold_label)
            pred_label = data_loaded.iloc[i]['predicted_label']
            if args.dataset_name in ["quarel", "qasc", "numersense"]:
                pred_label = pred_label.upper()
            predicted_labels.append(pred_label)
            predicted_rationales.append(data_loaded.iloc[i]['predicted_rationale'])
            ct_predicted_output = data_loaded.iloc[i]['predicted_rationale'] + " So the answer is " + pred_label + '.'
            predicted_outputs.append(ct_predicted_output)
            # gold_responses.append("dummy The answer is " + data_loaded.iloc[i]['gold_label'])
            gold_responses.append(
                gold_q_and_rationale[data_loaded.iloc[i]['question']] + \
                " So the answer is " + gold_label + '.')
            num_sents.append(ct_predicted_output.count('.'))
            num_words.append(len(ct_predicted_output.split(' ')))
            if i<5:
                print(i,':',ct_predicted_output, gold_responses[-1])
    
    if args.outputs_other_sources_type == "llama_json":
        data_loaded = pd.read_json(args.outputs_other_sources_file, lines=True)
        #if args.dataset_name!= "strategyqa":
        #data_loaded = data_loaded.T
        for i in range(len(data_loaded)):
            if data_loaded.iloc[i]['question'] not in gold_q_and_rationale:
                continue # ?? wut
            ct_question = data_loaded.iloc[i]['question']
            questions.append(ct_question)
            gold_label = gold_q_and_label[ct_question]
            if args.dataset_name in ["quarel", "qasc", "numersense"]:
                gold_label = gold_label.upper()
            gold_labels.append(gold_label)
            if args.dataset_name=="strategyqa":
                tmp = data_loaded.iloc[i]['trunc_output'].split("\nA: So the answer is")
                pred_rationale = tmp[0]
                try:
                    pred_label = tmp[1].replace(".", "").lstrip().rstrip()
                except:
                    pred_label = ""
            else:
                pred_rationale = data_loaded.iloc[i]['trunc_output'].split("So the answer is")[0]
                try:
                    pred_label = data_loaded.iloc[i]['trunc_output'].replace(":","").split("So the answer is ")[1].replace(".", "").lstrip().rstrip() #data_loaded.iloc[i]['answer']
                    #pred_label = data_loaded.iloc[i]['trunc_output'].split("So the answer is: ")[1].replace(".", "").lstrip().rstrip() #data_loaded.iloc[i]['answer']
                except:
                    pred_label = ""
            if args.dataset_name in ["quarel", "qasc", "numersense"]:
                pred_label = pred_label.upper()
            predicted_labels.append(pred_label)
            predicted_rationales.append(pred_rationale)
            ct_predicted_output = pred_rationale + " So the answer is " + pred_label + '.'
            predicted_outputs.append(ct_predicted_output)
            # gold_responses.append("dummy The answer is " + data_loaded.iloc[i]['gold_label'])
            gold_responses.append(
                gold_q_and_rationale[data_loaded.iloc[i]['question']] + \
                " So the answer is " + gold_label + '.')
            num_sents.append(ct_predicted_output.count('.'))
            num_words.append(len(ct_predicted_output.split(' ')))
            if i<5:
                print(i,':',ct_predicted_output, gold_responses[-1])

    if args.outputs_other_sources_type == "quark_json":
        data_loaded = pd.read_json(args.outputs_other_sources_file)
        #if args.dataset_name!= "strategyqa":
        #data_loaded = data_loaded.T
        for i in range(len(data_loaded)):
            if data_loaded.iloc[i]['question'] not in gold_q_and_rationale:
                continue # ?? wut
            ct_question = data_loaded.iloc[i]['question']
            questions.append(ct_question)
            gold_label = gold_q_and_label[ct_question]
            if args.dataset_name in ["quarel", "qasc", "numersense"]:
                gold_label = gold_label.upper()
            gold_labels.append(gold_label)
            
            tmp = data_loaded.iloc[i]['model_response'].split("So the answer is")
            pred_rationale = tmp[0]
            try:
                pred_label = tmp[1].replace(".", "").lstrip().rstrip()
            except:
                pred_label = ""

            if args.dataset_name in ["quarel", "qasc", "numersense"]:
                pred_label = pred_label.upper()
            predicted_labels.append(pred_label)
            predicted_rationales.append(pred_rationale)
            ct_predicted_output = pred_rationale + " So the answer is " + pred_label + '.'
            predicted_outputs.append(ct_predicted_output)
            # gold_responses.append("dummy The answer is " + data_loaded.iloc[i]['gold_label'])
            gold_responses.append(
                gold_q_and_rationale[data_loaded.iloc[i]['question']] + \
                " So the answer is " + gold_label + '.')
            num_sents.append(ct_predicted_output.count('.'))
            num_words.append(len(ct_predicted_output.split(' ')))
            if i<5:
                print(i,':',ct_predicted_output, gold_responses[-1])

    if args.outputs_other_sources_type == "gold_data":
        for i in range(len(gold_data_loaded)):
            ct_gold_question = gold_data_loaded.iloc[i]['question']
            ct_gold_label = gold_q_and_label[ct_gold_question]
            ct_gold_rationale = gold_q_and_rationale[ct_gold_question]

            questions.append(ct_gold_question)
            gold_labels.append(ct_gold_label)
            predicted_labels.append(ct_gold_label)
            predicted_rationales.append(ct_gold_rationale)
            ct_predicted_output = ct_gold_rationale + " So the answer is " + ct_gold_label  + '.'
            predicted_outputs.append(ct_predicted_output)
            gold_responses.append(ct_predicted_output)
            num_sents.append(ct_predicted_output.count('.'))
            num_words.append(len(ct_predicted_output.split(' ')))
            if i<5:
                print(i,':',ct_predicted_output, gold_responses[-1])


    if args.outputs_other_sources_type == "csv":
        # # gold data stuff
        # gold_data_loaded = pd.read_json(args.outputs_other_sources_gold_file, orient='records')
        # if args.dataset_name != 'strategyqa':
        #     gold_data_loaded = gold_data_loaded.T
        #     gold_classes = {0 : "No", 1 : "Yes"}
        # gold_q_and_label = {}
        # for i in range(len(gold_data_loaded)):
        #     ct_gold_question = gold_data_loaded.iloc[i]['question']
        #     ct_gold_label = gold_data_loaded.iloc[i]['answer'].lower()
        #     if args.dataset_name == 'strategyqa':
        #         ct_gold_label = gold_classes[ct_gold_label].lower()
        #     gold_q_and_label[ct_gold_question] = ct_gold_label

        data_loaded = pd.read_csv(args.outputs_other_sources_file)
        if args.dataset_name == 'strategyqa':
            lll = ["The answer is", "So the answer is", "So, the answer is", "Thus the answer is", "Thus, the answer is", "Of the choices, the most appropriate answer is"]
        else:
            lll = ["The answer is (", "So the answer is (", "So, the answer is (", "Thus the answer is (", "Thus, the answer is (", "Of the choices, the most appropriate answer is (", "The correct answer is ("]
        lll_lower = [ct_lll.lower() for ct_lll in lll]

        if args.dataset_name == 'strategyqa':
            lll_lower.remove("the answer is") # since this will be there in all of them, it has to be the last thing we resort to
            lll_lower.append("the answer is")
        else:
            lll_lower.remove("the answer is (") # since this will be there in all of them, it has to be the last thing we resort to
            lll_lower.append("the answer is (")

        for i in range(len(data_loaded)):
            ct_question = data_loaded.iloc[i]['question']
            ct_gold_label = gold_q_and_label[ct_question]
            ct_response_text = data_loaded.iloc[i]['model_response']

            #print(ct_response_text)
            tmp = []
            for ct_lll in lll + lll_lower:
                if ct_lll in ct_response_text.replace(':', ''):
                    tmp = ct_response_text.replace(':', '').split(ct_lll)
                    break
            if len(tmp) == 0:
                tmp = ct_response_text.split("So the answer is") # dummy
            ct_predicted_rationale = tmp[0].strip()
            if ct_predicted_rationale[-1] != '.': 
                ct_predicted_rationale = ct_predicted_rationale + '.'
            if args.dataset_name != 'strategyqa':
                ct_predicted_label = '(' + tmp[1].strip()[:2].lower()
            else:
                if tmp[1].strip().lower().startswith('no'):
                    ct_predicted_label = 'no'
                elif tmp[1].strip().lower().startswith('yes'):
                    ct_predicted_label = 'yes'
                else:
                    ct_predicted_label = ""

            questions.append(ct_question)
            gold_labels.append(ct_gold_label)
            predicted_labels.append(ct_predicted_label)
            predicted_rationales.append(ct_predicted_rationale)
            ct_predicted_output = ct_predicted_rationale + " So the answer is " + ct_predicted_label + '.'
            predicted_outputs.append(ct_predicted_output)
            # gold_responses.append("dummy The answer is " + ct_gold_label)
            gold_responses.append(
                gold_q_and_rationale[ct_question] + \
                " So the answer is " + ct_gold_label)
            num_sents.append(ct_predicted_output.count('.'))
            num_words.append(len(ct_predicted_output.split(' ')))
            if i<5:
                print(i,':',ct_predicted_output, gold_responses[-1])


    if args.outputs_other_sources_type == 'json':
        # json files will be the gpt-3 
        data_loaded = json.load(open(args.outputs_other_sources_file, 'r'))
        for i in range(len(data_loaded)):
            ct_question = data_loaded[i]['question'].replace('Q: ', '').replace('A: ', '').replace('\n', ' ').replace('  ', ' ')
            questions.append(ct_question)
            gold_labels.append(data_loaded[i]['gold_label'].lower())
            if args.outputs_other_name == 'gpt-3':
                ct_predicted_output = data_loaded[i]['gpt_answer']

            predicted_outputs.append(ct_predicted_output)
            num_sents.append(ct_predicted_output.count('.'))
            num_words.append(len(ct_predicted_output.split(' ')))
            if 'The answer is' in ct_predicted_output:
                tmp = ct_predicted_output.split('The answer is')
                ct_predicted_label = tmp[1].strip().strip('.').lower()
                ct_predicted_rationale = tmp[0].strip()
            elif 'So the answer is: ' in ct_predicted_output:
                tmp = ct_predicted_output.split('So the answer is: ')
                ct_predicted_label = tmp[1].strip().strip('.').lower()
                ct_predicted_rationale = tmp[0].strip()
            else:
                ct_predicted_label = ''
                ct_predicted_rationale = ct_predicted_output
            predicted_labels.append(ct_predicted_label)
            predicted_rationales.append(ct_predicted_rationale)

    # calculating accuracy and rewards
    accuracy, correctness_list = get_accuracy(predicted_labels, gold_labels)

    # to save the individual datapoints
    to_save_dict = {'question': questions,
                    'model_response': predicted_outputs,
                    'correctness': correctness_list,
                    'num_sents': num_sents,
                    'num_words': num_words}

    scores = {}
    reward_scores_str = ''
    for rew in score_model.keys():
        assert len(predicted_rationales)>0
        scores[rew] = score_model[rew].get_reward(questions, predicted_outputs, gold_responses,
            f'eval_eval_others_' + args.outputs_other_name, split="val", rationales=predicted_rationales)
        to_save_dict[rew] = scores[rew]
        reward_score = np.nanmean([s for s in scores[rew] if isinstance(s, float)])
        reward_scores_str += rew + ': ' + str(round(reward_score,2)) + ' ' 
        # correlation
        scores_for_correlation = [s if isinstance(s, float) else np.nan for s in scores[rew]]
        calculated_correlation_score = correlation_ratio(correctness_list, scores_for_correlation,
                                                            nan_strategy='drop')
        reward_scores_str += 'corr: ' + str(round(calculated_correlation_score,2)) + ' '
        print(args.outputs_other_name + " overall " + rew + " = ", reward_score, " correlation w. accuracy = ",
                calculated_correlation_score)

    # using majority predicted label as basis
    dict_pred_labels = defaultdict(list)
    dict_gold_labels = {}
    dict_scores = {}
    for rew in list(score_model.keys()) + ['num_sents', 'num_words']:
        dict_scores[rew] = defaultdict(list)
    for iii in range(len(questions)):
        ct_q = questions[iii]
        dict_pred_labels[ct_q].append(predicted_labels[iii])
        dict_gold_labels[ct_q] = gold_labels[iii]
        for rew in score_model.keys():
            dict_scores[rew][ct_q].append(scores[rew][iii])
        dict_scores['num_sents'][ct_q].append(num_sents[iii])
        dict_scores['num_words'][ct_q].append(num_words[iii])

    # need to average scores and do majority voting for labels
    unique_questions_list = list(dict_gold_labels.keys())
    all_reward_scores = defaultdict(list)
    majority_accuracy = 0
    for ct_q in unique_questions_list:
        tmp = dict_pred_labels[ct_q]
        majority_pred_label = max(set(tmp), key = tmp.count) 
        ct_gold_label = dict_gold_labels[ct_q]
        if majority_pred_label == ct_gold_label:
            majority_accuracy += 1

        for rew in score_model.keys():
            tmp_rew_score = []
            for sub_idx in range(len(tmp)):
                if tmp[sub_idx]==majority_pred_label: # considering only these scores for averaging
                    if isinstance(dict_scores[rew][ct_q][sub_idx], float):
                        tmp_rew_score.append(dict_scores[rew][ct_q][sub_idx])
            if len(tmp_rew_score) > 0:
                all_reward_scores[rew].append(np.nanmean(tmp_rew_score))
            else:
                all_reward_scores[rew].append('')
        for an_gen in ["num_sents", "num_words"]:
            tmp_nums = []
            for sub_idx in range(len(tmp)):
                if tmp[sub_idx]==majority_pred_label: # considering only these scores for averaging
                    tmp_nums.append(dict_scores[an_gen][ct_q][sub_idx])
            all_reward_scores[an_gen].append(np.nanmean(tmp_nums))

    averaged_reward_score = {}
    reward_scores_str = ''
    for rew in list(score_model.keys()) + ["num_sents", "num_words"]:
        averaged_reward_score[rew] = np.nanmean([ss for ss in all_reward_scores[rew] if isinstance(ss, float)])
        reward_scores_str += rew + ': ' + str(round(averaged_reward_score[rew],2)) + ' ' 
    accuracy = float(majority_accuracy)/len(unique_questions_list)*100

    # reward_scores_str += 'num_sents' + ': ' + str(round(np.mean(num_sents),2)) + ' '
    # reward_scores_str += 'num_words' + ': ' + str(round(np.mean(num_words),2)) + ' '
    print(args.outputs_other_name + " accuracy = ", accuracy)
    print(reward_scores_str)
    # print(args.outputs_other_name + " num_sents = ", np.mean(num_sents))
    # print(args.outputs_other_name + " num_words = ", np.mean(num_words))

    # saving the scores
    with open(os.path.join(args.reward_dir, 'eval_reward_scores_other_outputs.txt'), 'a') as f:
        f.write('Name ' + args.outputs_other_name +' - '+ reward_scores_str + ' Accuracy: ' + str(round(accuracy,2)) + '\n')

    # saving the individual datapoints in csv
    with open(os.path.join(args.reward_dir,'eval_output_other' + args.outputs_other_name + '.csv'), 'w') as csv_f:
        writer = csv.writer(csv_f)
        key_list = list(to_save_dict.keys())
        limit = len(to_save_dict[key_list[0]])
        
        writer.writerow(key_list)
        for lt in range(limit):
            writer.writerow([to_save_dict[keykey][lt] for keykey in key_list])

if __name__ == "__main__":
    main()
