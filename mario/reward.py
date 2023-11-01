import json
import os
import sys

import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from typing import List, Iterable, Dict, Any
import torch
import logging
import pdb

from utils.utils import batchify, load_jsonl, load_data_strategyqa_template, init_model

from datasets import load_dataset, Dataset
import transformers 

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from flanT5_scores import FlanT5Processor, calc_and_return_flanT5_scores, calc_and_return_flanT5_completeness_scores
from csqa_acceptability import csqa_acc_model_and_tokenizer, csqa_acc_return_scores
from rq_score import load_supervised_rq, get_rq_scores

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def get_accuracy_individual(pred_responses, gold_responses):
    correct = 0
    indiv_correct = []
    for i in range(len(pred_responses)):
        gold_label = gold_responses[i].split('So the answer is ')[1].strip().strip('.').lower()
        split_text_pred = pred_responses[i].split('So the answer is ')
        if len(split_text_pred) > 1:
            pred_label = split_text_pred[1].strip().strip('.').lower()
        else:
            pred_label = ''

        if pred_label == gold_label:
            correct += 1
            indiv_correct.append(1.0)
        else:
            indiv_correct.append(0.0)

    accuracy = float(correct)/len(pred_responses)
    return indiv_correct

def _ngram_counts(text, ngram): # for reward: diversity
    token_list = text.strip().split()
    start_idx, end_idx = 0, ngram
    total_num = 0
    ngram_set = set()
    while end_idx <= len(token_list):
        one_ngram_list = token_list[start_idx:end_idx]
        assert len(one_ngram_list) == ngram
        one_ngram = ' '.join(one_ngram_list)
        total_num += 1
        ngram_set.add(one_ngram)
        start_idx += 1
        end_idx += 1
    return len(ngram_set), total_num
    
def compute_repetition(continuations, start_n=2): # for reward: diversity
    out = []
    for continuation in continuations:
        stats = {}
        for n in range(start_n, 5):
            unique, total = _ngram_counts(continuation, n)
            if not total:
                break
            stats['rep_%d' % n] = 1. - unique / total

        if any(['rep_%d' % n not in stats for n in range(2, 5)]):
            for n in range(2, 5):
                if 'rep_%d' % n not in stats:
                    stats['rep_%d' % n] = 0.
        if start_n == 2:
            stats['diversity'] = (1. - stats['rep_2']) * (1. - stats['rep_3']) * (1. - stats['rep_4'])
        elif start_n == 3:
            stats['diversity'] = (1. - stats['rep_3']) * (1. - stats['rep_4'])
        out.append(stats)
    return out


def product_rewards(reward_list: List[List[float]]):
    return [np.prod(x) for x in list(zip(*reward_list))]


# for rationale-quality reward
rq_model_dict = {'i2o': {}, 'ir2o': {}, 'choices': {}}
rq_model_dict['i2o']['quarel'] = 'save/quarel/i2o/08-27-2023_16-30-59_1/model/ckp_5000.pth' 
rq_model_dict['ir2o']['quarel'] = 'save/quarel/ir2o/08-27-2023_16-30-44_2/model/ckp_5000.pth' 
rq_model_dict['i2o']['obqa'] = 'save/obqa/i2o/09-25-2023_17-39-09_1/model/ckp_45000.pth' 
rq_model_dict['ir2o']['obqa'] = 'save/obqa/ir2o/09-25-2023_17-39-09_2/model/ckp_45000.pth' 
rq_model_dict['i2o']['strategyqa'] = 'save/strategyqa/i2o/08-24-2023_10-54-29_1/model/ckp_1000.pth'
rq_model_dict['ir2o']['strategyqa'] = 'save/strategyqa/ir2o/08-24-2023_10-54-29_2/model/ckp_8000.pth'


rq_model_dict['i2o']['csqa'] = 'save/csqa/i2o/09-02-2023_16-55-06_1/model/ckp_15000.pth'
rq_model_dict['ir2o']['csqa'] = 'save/csqa/ir2o/09-02-2023_16-55-24_2/model/ckp_22000.pth'
rq_model_dict['i2o']['wg'] = 'save/wg/i2o/09-05-2023_22-37-11_1/model/ckp_28000.pth'
rq_model_dict['ir2o']['wg'] = 'save/wg/ir2o/09-06-2023_10-38-08_2/model/ckp_40000.pth'
rq_model_dict['i2o']['qasc'] = 'save/qasc/i2o/09-17-2023_19-43-43_1/model/ckp_37000.pth'
rq_model_dict['ir2o']['qasc'] = 'save/qasc/ir2o/09-17-2023_19-43-33_2/model/ckp_37000.pth' 
rq_model_dict['i2o']['numersense'] = 'save/numersense/i2o/10-13-2023_13-22-01_1/model/ckp_11000.pth'
rq_model_dict['ir2o']['numersense'] = 'save/numersense/ir2o/10-13-2023_14-43-27_2/model/ckp_11000.pth'

rq_model_dict['i2o']['coinflip'] = 'save/coinflip/i2o/09-20-2023_02-26-44_1/model/ckp_18000.pth'
rq_model_dict['ir2o']['coinflip'] = 'save/coinflip/ir2o/09-20-2023_09-46-49_2/model/ckp_18000.pth'
rq_model_dict['i2o']['penguins_in_a_table'] = 'save/penguins_in_a_table/i2o/09-14-2023_14-52-25_1/model/ckp_6000.pth'
rq_model_dict['ir2o']['penguins_in_a_table'] = 'save/penguins_in_a_table/ir2o/09-14-2023_15-02-00_2/model/ckp_6000.pth'

rq_model_dict['choices']['quarel'] = ['(A)', '(B)']
rq_model_dict['choices']['wg'] = ['(A)', '(B)']
rq_model_dict['choices']['qasc'] = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)"]
rq_model_dict['choices']['numersense'] = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)", "(K)", "(L)"]
rq_model_dict['choices']['penguins_in_a_table'] = ["(A)", "(B)", "(C)", "(D)", "(E)"]
rq_model_dict['choices']['obqa'] = ['(a)', '(b)', '(c)', '(d)']
rq_model_dict['choices']['csqa'] = ['(a)', '(b)', '(c)', '(d)', '(e)']
rq_model_dict['choices']['strategyqa'] = ['no', 'yes']
rq_model_dict['choices']['coinflip'] = ['no', 'yes']
class Reward:
    def __init__(self, save_path: str, batch_size: int, reward_name: str, device: int,
        flant5_model: FlanT5Processor):
        self.path = save_path
        self.batch_size = batch_size
        self.reward_name = reward_name
        self.device = f'cuda:{device}'

        if 'roscoe' in self.reward_name and '_and_' not in self.reward_name:
            self.score_name = self.reward_name.split('-')[1]
            self.roscoe_evaluator = load_roscoe_for_quark(self.score_name)

        if self.reward_name == 'cola_grammar':
            self.score_name = self.reward_name
            cola_model_name = "textattack/roberta-base-CoLA"
            self.cola_tokenizer = RobertaTokenizer.from_pretrained(cola_model_name)
            self.cola_model = RobertaForSequenceClassification.from_pretrained(cola_model_name).to('cuda:0')

        if self.reward_name=='num_words':
            self.score_name = self.reward_name
        if 'length' in self.reward_name:
            if '_and_' not in self.reward_name:
                self.score_name = self.reward_name
                # length-more: just as verbose as possible
                # length-int: have a length of value int
                if 'min' in self.reward_name and 'max' in self.reward_name:
                    min_idx = self.reward_name.index('min-')
                    max_idx = self.reward_name.index('max-')
                    #self.min_length = float(self.reward_name[min_idx + 4])
                    #self.max_length = float(self.reward_name[max_idx + 4])
                    tmp = self.reward_name.replace('min-', '').replace('max-', '').split('-')
                    self.min_length = float(tmp[1])
                    self.max_length = float(tmp[2])
                    print(self.reward_name, self.min_length, self.max_length)
            elif '_and_roscoe-missing_step' in self.reward_name:
                self.score_name = self.reward_name
                self.roscoe_evaluator = load_roscoe_for_quark('missing_step')
            elif '_and_diversity-3-4' in self.reward_name:
                self.score_name = self.reward_name

        if self.reward_name == 'flan-t5-factuality':
            # applying flan-t5 at the sentence level, and averaging scores of sentences within a rationale
            self.score_name = self.reward_name
            self.eval_type = 'sent'
            self.flant5_model = flant5_model

        if self.reward_name == 'flan-t5-factuality-full':
            # applying flan-t5 at the level of the entire rationale
            self.score_name = self.reward_name
            self.eval_type = 'full'
            self.flant5_model = flant5_model

        if self.reward_name == 'flan-t5-factuality-full_and_roscoe-missing_step':
            # taking product of two scores
            self.score_name = self.reward_name
            self.roscoe_evaluator = load_roscoe_for_quark('missing_step')
            self.eval_type = 'full'
            self.flant5_model = flant5_model

        if self.reward_name == 'flan-t5-factuality-w-qn':
            # applying flan-t5 to the question + rationale + answer
            self.score_name = self.reward_name
            self.eval_type = 'full_w_qn'
            self.flant5_model = flant5_model

        if self.reward_name == 'flan-t5-completeness':
            # applying flan-t5 completeness
            self.score_name = self.reward_name
            self.flant5_model = flant5_model

        if self.reward_name == 'flan-t5-completeness-1':
            # applying flan-t5 completeness
            self.score_name = self.reward_name
            self.flant5_model = flant5_model
        
        if self.reward_name in ['diversity', 'diversity-3-4']:
            # calculating 2-gram, 3-gram and 4-gram repetition rates and taking product of (1 - repetition rate)
            self.score_name = self.reward_name

        if self.reward_name in ['csqa-acceptability', 'csqa-acceptability-and-accuracy']:
            # only for csqa
            # acceptability score from https://arxiv.org/pdf/2112.08674.pdf
            self.score_name = self.reward_name
            self.csqa_acc_details_dict = csqa_acc_model_and_tokenizer()

        if 'rationale-quality' in self.reward_name:
            self.score_name = 'rationale-quality'
            tmp = self.reward_name.split('-rationale-quality-')
            self.rq_dataset_name = tmp[0]
            self.rq_mode = tmp[1] # discrete or continuous
            num_gpus = torch.cuda.device_count()
            device_num = f"cuda:{num_gpus-1}"
            self.rq_device = torch.device(device_num)
            self.rq_i2o_model, self.rq_i2o_tokenizer = load_supervised_rq(rq_model_dict['i2o'][self.rq_dataset_name], 'large', self.rq_device)
            self.rq_ir2o_model, self.rq_ir2o_tokenizer = load_supervised_rq(rq_model_dict['ir2o'][self.rq_dataset_name], 'large', self.rq_device)
            self.rq_dataset_choices = rq_model_dict['choices'][self.rq_dataset_name]

        if 'plausibility' in self.reward_name:# == 'plausibility':
            # VERA from https://arxiv.org/pdf/2305.03695.pdf
            self.score_name = self.reward_name
            self.vera_tokenizer = transformers.AutoTokenizer.from_pretrained('liujch1998/vera')
            self.vera_model = transformers.T5EncoderModel.from_pretrained('liujch1998/vera')
            self.vera_model.D = self.vera_model.shared.embedding_dim

            self.vera_linear = torch.nn.Linear(self.vera_model.D, 1)
            self.vera_linear.weight = torch.nn.Parameter(self.vera_model.shared.weight[32099, :].unsqueeze(0))
            self.vera_linear.bias = torch.nn.Parameter(self.vera_model.shared.weight[32098, 0].unsqueeze(0))
            self.vera_model.eval()

            num_gpus = torch.cuda.device_count()
            self.vera_device = torch.device(f"cuda:1")#{num_gpus-1}")
            self.vera_model.to(self.vera_device)
            self.vera_linear.to(self.vera_device)
            self.vera_t = self.vera_model.shared.weight[32097, 0].item() # temperature for calibration

    def get_reward(self, prompts: List[str], responses: List[str], responses_gold: List[str], epoch: str, split: str, rationales: List[str] = []) -> List[float]:
        perspective_file = f'{self.path}/perspective_{self.reward_name}_{epoch}.json'
       
        print(self.reward_name)
        print(self.score_name)
        if 'roscoe' in self.reward_name and '_and_' not in self.reward_name:
            scores_dict = calc_roscoe_for_quark(evaluator=self.roscoe_evaluator,
                score_name=self.score_name,
                contexts=prompts,
                hypos=responses,
                references=responses_gold)
            scores = []
            for i in range(len(prompts)):
                scores.append(scores_dict[prompts[i]][self.score_name])

        if self.reward_name == 'cola_grammar':
            texts = [res.split('So the answer is')[0].strip() for res in responses]
            #texts = [t.strip() for t in responses]
            scores = []
            batch_size = 20
            steps = int(len(texts)/batch_size) + 1
            for i in tqdm(range(steps)):
                try:
                    batch = texts[i*batch_size:(i+1)*batch_size]
                except:
                    batch = texts[i*batch_size:]
                if len(batch)==0: continue

                ct_inputs = self.cola_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to('cuda:0')
                with torch.no_grad():
                    logits = self.cola_model(**ct_inputs).logits
                    probs = logits.softmax(dim=-1)
                scores.extend(probs[:, 1].tolist())

        if self.reward_name=='num_words':
            scores = [float(len(x.split(' '))) for x in responses]
        if 'length' in self.reward_name:
            if '_and_' not in self.reward_name:
                if self.reward_name == 'length-more':
                    # crude score
                    scores = [float(x.count('.')) for x in responses]
                elif 'min' in self.reward_name and 'max' in self.reward_name:
                    lengths = [float(x.count('.')) for x in responses]
                    scores = []
                    for iii in range(len(responses)):
                        ct_l = lengths[iii]
                        ct_score = (ct_l - self.min_length)*(self.max_length - ct_l)
                        scores.append(ct_score)
            elif '_and_roscoe-missing_step' in self.reward_name:
                # length scores
                if 'length-more' in self.reward_name:
                    length_scores = [float(x.count('.')) for x in responses]
                # missing-step
                scores_dict = calc_roscoe_for_quark(evaluator=self.roscoe_evaluator,
                    score_name='missing_step',
                    contexts=prompts,
                    hypos=responses,
                    references=responses_gold)
                roscoe_scores = []
                for i in range(len(prompts)):
                    roscoe_scores.append(scores_dict[prompts[i]]['missing_step'])
                # product
                scores = []
                for iii in range(len(length_scores)):
                    if isinstance(length_scores[iii], float) and isinstance(roscoe_scores[iii], float):
                        scores.append(length_scores[iii]*roscoe_scores[iii])
                    else:
                        scores.append('')
            elif '_and_diversity-3-4' in self.reward_name:
                # length scores
                if 'length-more' in self.reward_name:
                    length_scores = [float(x.count('.')) for x in responses]
                # diversity scores
                div_scores_1 = compute_repetition(responses, start_n=3)
                div_scores = [d['diversity'] for d in div_scores_1]
                # product
                scores = []
                for iii in range(len(length_scores)):
                    if isinstance(length_scores[iii], float) and isinstance(div_scores[iii], float):
                        scores.append(length_scores[iii]*div_scores[iii])
                    else:
                        scores.append('')

        if self.reward_name in ['flan-t5-factuality', 'flan-t5-factuality-full', 'flan-t5-factuality-w-qn']:
            scores = calc_and_return_flanT5_scores(flant5_model=self.flant5_model,
                questions=prompts, pred_rationales=responses,
                eval_type=self.eval_type)

        if self.reward_name == 'flan-t5-completeness':
            scores = calc_and_return_flanT5_completeness_scores(flant5_model=self.flant5_model,
                questions=prompts, pred_rationales=responses)

        if self.reward_name == 'flan-t5-completeness-1':
            scores = calc_and_return_flanT5_completeness_scores(flant5_model=self.flant5_model,
                questions=prompts, pred_rationales=responses, comp_type='type1')
        
        if self.reward_name == 'flan-t5-factuality-full_and_roscoe-missing_step':
            # factuality
            factuality_scores = calc_and_return_flanT5_scores(flant5_model=self.flant5_model,
                questions=prompts, pred_rationales=responses,
                eval_type=self.eval_type)
            # missing-step
            scores_dict = calc_roscoe_for_quark(evaluator=self.roscoe_evaluator,
                score_name='missing_step',
                contexts=prompts,
                hypos=responses,
                references=responses_gold)
            roscoe_scores = []
            for i in range(len(prompts)):
                roscoe_scores.append(scores_dict[prompts[i]]['missing_step'])
            # product
            scores = []
            for iii in range(len(factuality_scores)):
                if isinstance(factuality_scores[iii], float) and isinstance(roscoe_scores[iii], float):
                    scores.append(factuality_scores[iii]*roscoe_scores[iii])
                else:
                    scores.append('')

        if self.reward_name == 'diversity':
            rationales = [res.split('So the answer is')[0].strip() for res in responses]
            div_scores = compute_repetition(rationales)#responses)
            scores = [d['diversity'] for d in div_scores]
        
        if self.reward_name == 'diversity-3-4':
            div_scores = compute_repetition(responses, start_n=3)
            scores = [d['diversity'] for d in div_scores]

        if self.reward_name == 'csqa-acceptability':
            scores = csqa_acc_return_scores(pred_rationales=responses,
                questions=prompts,
                details_dict=self.csqa_acc_details_dict,
                split=split)

        if self.reward_name == 'csqa-acceptability-and-accuracy':
            acceptability_scores = csqa_acc_return_scores(pred_rationales=responses,
                questions=prompts,
                details_dict=self.csqa_acc_details_dict,
                split=split)
            accuracy_scores = get_accuracy_individual(responses, responses_gold)
            scores = []
            for iii in range(len(acceptability_scores)):
                if isinstance(acceptability_scores[iii], float) and isinstance(accuracy_scores[iii], float):
                    scores.append(acceptability_scores[iii]*accuracy_scores[iii])
                else:
                    scores.append('')

        if 'rationale-quality' in self.reward_name:
            scores = get_rq_scores(device=self.rq_device, mode=self.rq_mode,
                i2o_model=self.rq_i2o_model, i2o_tokenizer=self.rq_i2o_tokenizer,
                ir2o_model=self.rq_ir2o_model, ir2o_tokenizer=self.rq_ir2o_tokenizer,
                prompts=prompts, responses_gold=responses_gold, responses=responses,
                dataset_name=self.rq_dataset_name, dataset_choices=self.rq_dataset_choices)

        if 'plausibility' in self.reward_name:# == 'plausibility':
            scores_list = []
            batch_size = 40
            if self.reward_name=='plausibility' and len(rationales) == 0:
                rationales = [res.split('So the answer is')[0].strip() for res in responses]
            elif self.reward_name == 'plausibility-w-qn':
                rationales = []
                for i in range(len(prompts)):
                    ct_q = prompts[i]
                    if ct_q[-1]!='.' and ct_q[-1]!='?':
                        ct_q = ct_q + '.'
                    rationales.append(ct_q + ' ' + responses[i])
            elif self.reward_name=='coinflip-plausibility':
                rationales1 = [res.split('So the answer is')[0].strip() for res in responses]
                rationales = []
                for i in range(len(prompts)):
                    ct_q = prompts[i].replace(" Is the coin still heads up?", "")
                    if ct_q[-1]!='.':
                        ct_q = ct_q + '.'
                    rationales.append(ct_q + ' ' + rationales1[i])
                    if i<5:
                        print(i, "coinflip-plausibility:", rationales[i])
            steps = int(len(rationales)/batch_size) + 1
            for i in tqdm(range(steps)):
                try:
                    batch = rationales[i*batch_size:(i+1)*batch_size]
                except:
                    batch = rationales[i*batch_size:]
                if len(batch) == 0: 
                    continue
            
                tok = self.vera_tokenizer.batch_encode_plus(batch, return_tensors='pt', padding='longest',
                    truncation='longest_first', max_length=128)
                input_ids = tok.input_ids.to(self.vera_device)
                attention_mask = tok.attention_mask.to(self.vera_device)
                with torch.no_grad():
                    output = self.vera_model(input_ids=input_ids, attention_mask=attention_mask)
                    last_indices = attention_mask.sum(dim=1, keepdim=True) - 1 # (B, 1)
                    last_indices = last_indices.unsqueeze(-1).expand(-1, -1, self.vera_model.D) # (B, 1, D)
                    last_hidden_state = output.last_hidden_state.to(self.vera_device) # (B, L, D)
                    hidden = last_hidden_state.gather(dim=1, index=last_indices).squeeze(1) # (B, D)
                    logits = self.vera_linear(hidden).squeeze(-1) # (B)
                    logits_calibrated = logits / self.vera_t
                    scores = logits.sigmoid()
                    scores_calibrated = logits_calibrated.sigmoid()
                    scores_list.extend([ss for ss in scores_calibrated.cpu().tolist() if isinstance(ss, float)])
            scores = scores_list

        print("len of scores:", len(scores))
        predictions = [{'score': s} for s in scores]
        with open(perspective_file, 'w') as fo:
            for res in predictions:
                fo.write(json.dumps(res) + '\n')

        assert os.path.exists(perspective_file), 'missing perspective file'
        data = pd.DataFrame.from_dict({'prompt': prompts})
        results = collate(data, responses, load_jsonl(perspective_file),
            os.path.join(self.path, f'reward_{self.reward_name}_{epoch}.json'))
        rewards = [y['score'] for x in results for y in x]
        return rewards
      
