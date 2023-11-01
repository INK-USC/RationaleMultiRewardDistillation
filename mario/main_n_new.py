#########################################################
# base code taken from https://github.com/GXimingLu/Quark
#########################################################
import os
import csv
import torch
import json
import time
import logging
import random
import pickle
import argparse
import numpy as np
from typing import List
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSeq2SeqLM

from arguments import get_args
from policy_n import Policy
from data_pool_n import DataPool_N
from reward import Reward, RevReward
from reference import ReferenceDataPool
from utils.utils import ensure_dir, ceil_div, reduce_mean, reduce_sum
from utils.constants import T5_NUM_TOKEN
from flanT5_scores import FlanT5Processor

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)

class PromptDataset(Dataset):
    def __init__(self, path, dataset_name, check_repeats=0):
        self.data = pd.read_json(path, orient='records')
        self.data = self.data.T
        self.prompts = []
        self.references = []
        self.gold_labels = []

        for i in range(len(self.data)):
            d = self.data.iloc[i]
            ct_question = d['question']
            ct_rationale = d['rationale']
            if "So the answer is " in ct_rationale: # TEMPORARY FIX SHOULD CHANGE IN PREPRCESSING ITSELF
                continue
            if ct_rationale[-1]!='.': ct_rationale = ct_rationale + '.'

            ct_label = d['answer']
            if check_repeats == 1:
                if ct_question in self.prompts:
                    continue
            self.prompts.append(ct_question)
            self.references.append(ct_rationale + ' So the answer is ' + ct_label + '.')
            self.gold_labels.append('So the answer is ' + ct_label + '.')

            if i < 5:
                print("Prompt:", self.prompts[-1])
                print("Reference:", self.references[-1])
                print("Gold label sequence:", self.gold_labels[-1])
        print("Length of data:", len(self.prompts))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {'prompt': self.prompts[idx],
                'reference': self.references[idx],
                'gold_label': self.gold_labels[idx],
                }

class PromptCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        prompts = [sequence['prompt'] for sequence in sequences]
        encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return input_ids, attention_mask

class PromptCollator_WithGold(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        prompts = [sequence['prompt'] for sequence in sequences]
        responses_gold = [sequence['reference'] for sequence in sequences]
        gold_label_sequences = [sequence['gold_label'] for sequence in sequences]
        encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return prompts, input_ids, attention_mask, responses_gold, gold_label_sequences


class SequenceDataset(Dataset):
    def __init__(self, data_pool: DataPool_N):
        self.queries, self.responses, self.cat_tokens, self.tc_tokens, self.gold_label_sequences = data_pool.get_data()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {'query': self.queries[idx],
                'response': self.responses[idx],
                'cat_tokens': self.cat_tokens[idx],
                'tc_tokens': self.tc_tokens[idx],
                'gold_label_sequence': self.gold_label_sequences[idx],
                }

class SequenceCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        queries = [sequence['query'] for sequence in sequences]
        query_encodings_dict = self.tokenizer(queries, return_tensors="pt", padding=True)
        query_input_ids = query_encodings_dict['input_ids']
        query_mask = query_encodings_dict['attention_mask']

        responses = [sequence['response'] for sequence in sequences]
        response_encodings_dict = self.tokenizer(responses, return_tensors="pt", padding=True)
        response_input_ids = response_encodings_dict['input_ids']
        response_mask = response_encodings_dict['attention_mask']

        gold_label_sequences = [sequence['gold_label_sequence'] for sequence in sequences]
        gold_label_sequences_encodings_dict = self.tokenizer(gold_label_sequences, return_tensors="pt", padding=True)
        gold_label_sequences_input_ids = gold_label_sequences_encodings_dict['input_ids']
        gold_label_sequences_mask = gold_label_sequences_encodings_dict['attention_mask']

        tc_ids = [self.tokenizer.convert_tokens_to_ids(sequence['tc_tokens']) for sequence in sequences]

        if sequences[0]['cat_tokens']!='':
            cat_ids = []
            cat_mask = []
            for sequence in sequences:
                ct_cat_tokens = sequence['cat_tokens'].split(' ')
                ct_cat_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in ct_cat_tokens]
                cat_ids.extend(ct_cat_ids)
                num_control_codes = len(ct_cat_tokens)
                cat_mask.extend([1]*num_control_codes)
                # print("ct_cat_tokens:", ct_cat_tokens)
                # print("ct_cat_ids:", ct_cat_ids)
                # print("num_control_codes:", num_control_codes)
                # print("")
            cat_ids = response_input_ids.new(cat_ids).reshape(len(sequences), num_control_codes)
            cat_mask = response_mask.new(cat_mask).reshape(len(sequences), num_control_codes)

            response_input_ids_wo_tc = torch.cat(
                [cat_ids, response_input_ids],
                dim=1)
            response_input_ids_w_tc = torch.cat(
                [cat_ids, response_input_ids.new(tc_ids)[:, None], response_input_ids],
                dim=1)
            # print("response ids:", response_input_ids.shape)
            # print("response ids wo tc:", response_input_ids_wo_tc.shape)
            # print("response ids w tc:", response_input_ids_w_tc.shape)
            # print("")
            response_mask_wo_tc = torch.cat([cat_mask, response_mask], dim=1)
            response_mask_w_tc = torch.cat([cat_mask, response_mask.new([1] * len(query_mask))[:, None], response_mask], dim=1)

        else:
            response_input_ids_wo_tc = response_input_ids
            response_input_ids_w_tc = torch.cat(
                [response_input_ids.new(tc_ids)[:, None], response_input_ids],
                dim=1)
            response_mask_wo_tc = response_mask
            response_mask_w_tc = torch.cat([response_mask.new([1] * len(query_mask))[:, None], response_mask], dim=1)
            
        return query_input_ids, query_mask, response_input_ids_wo_tc, response_input_ids_w_tc, \
        response_mask_wo_tc, response_mask_w_tc, gold_label_sequences_input_ids, gold_label_sequences_mask

class FixedController:
    def __init__(self, coef):
        self.value = coef

    def update(self, current, n_steps, lower_bound):
        pass

class AdaptiveController:
    def __init__(self, init_coef, target, horizon):
        self.value = init_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps, lower_bound):
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        if lower_bound:
            mult = 1 + proportional_error * n_steps / self.horizon
        else:
            mult = 1 - proportional_error * n_steps / self.horizon
        self.value *= mult

class ConditionTrainer:
    def __init__(self,
                 params: argparse.Namespace,
                 policy: Policy,
                 ref_policy: Policy,
                 data_pool: DataPool_N,
                 score_model: Reward,
                 tree_tokens: List[str],
                 task_correctness_tokens: List[str],
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 expert_rollouts: ReferenceDataPool,
                 optimizer: Optimizer,
                 scheduler: LambdaLR):

        self.params = params
        self.policy = policy
        self.ref_policy = ref_policy
        self.data_pool = data_pool
        self.score_model = score_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.expert_rollouts = expert_rollouts
        self.writer = SummaryWriter()
        self.prompt_sampler = iter(self.train_dataloader)

        if self.params.adaptive_kl:
            self.kl_ctl = AdaptiveController(self.params.kl_coef, self.params.target_kl, self.params.horizon)
        else:
            self.kl_ctl = FixedController(self.params.kl_coef)
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

        if self.params.adaptive_entropy:
            self.entropy_ctl = AdaptiveController(self.params.entropy_coef, self.params.target_entropy,
                                                  self.params.horizon)
        else:
            self.entropy_ctl = FixedController(self.params.entropy_coef)

        self.tree_tokens = tree_tokens
        self.task_correctness_tokens = task_correctness_tokens

        self.best_cat = {}
        self.best_cat_id = {}
        self.num_control_codes = 0
        self.best_control_code = []

        if self.params.reward_as_product:
            rew = 'reward_product'
            self.num_control_codes = 1
            self.best_cat[rew] = self.tree_tokens[rew][0]
            self.best_cat_id[rew] = self.policy.tokenizer.convert_tokens_to_ids(self.best_cat[rew])
            self.best_control_code = [self.best_cat_id[rew]]
        else:
            for rew in self.score_model.keys():
                if self.params.train_option[rew] in ['quark', 'filter_quark']: # add control code only for this case
                    if rew == "accuracy":
                        self.best_cat[rew] = self.task_correctness_tokens[-1]
                        self.best_cat_id[rew] = self.policy.tokenizer.convert_tokens_to_ids(self.best_cat[rew])
                    else:
                        self.best_cat[rew] = self.tree_tokens[rew][0]
                        self.best_cat_id[rew] = self.policy.tokenizer.convert_tokens_to_ids(self.best_cat[rew])
                    self.num_control_codes += 1
                    self.best_control_code.append(self.best_cat_id[rew]) 
            if self.params.task_correctness == 'yes':
                self.best_tc = self.task_correctness_tokens[-1]
                self.best_tc_id = self.policy.tokenizer.convert_tokens_to_ids(self.best_tc)
                self.num_control_codes += 1
                self.best_control_code.append(self.best_tc_id)
        print("best control code:", self.best_control_code)

        self.sample_dataloader, self.sampler = None, None
        self.seq_collator = SequenceCollator(tokenizer=policy.tokenizer)

    # def add_control_code(self, input_ids, attention_mask):
    #     input_ids = torch.cat([input_ids.new([self.best_cat_id] * len(input_ids))[:, None], input_ids], dim=1)
    #     attention_mask = torch.cat([attention_mask.new([1] * len(attention_mask))[:, None], attention_mask], dim=1)
    #     return input_ids, attention_mask

    def get_control_code(self, input_ids):
        if self.num_control_codes > 0:
            return input_ids.new(self.best_control_code * len(input_ids)).reshape(-1, self.num_control_codes)
        else:
            return None
        # if self.params.task_correctness == 'no':
        #     return input_ids.new([self.best_cat_id] * len(input_ids))[:, None]
        # else:
        #     return input_ids.new([self.best_cat_id, self.best_tc_id] * len(input_ids)).reshape(-1, 2)

    def decode(self, query_input_ids, response_input_ids):
        query = [self.policy.tokenizer.decode(p, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                 for p in query_input_ids]
        response = [self.policy.tokenizer.decode(r, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    for r in response_input_ids]
        return query, response

    def sample(self, step):
            
        if step % self.params.sample_interval != 0:
            return
        print(f"[step {step}] Sampling ...")

        prompts, responses, responses_gold, gold_label_sequences = [], [], [], []
        # num_rollout_batch = self.params.num_policy_rollout // self.params.batch_size
        # print("num rollout batch:", num_rollout_batch)
        # for _ in tqdm(range(num_rollout_batch), desc='Sampling from current policy'):
        for roll in range(self.params.num_policy_rollout):
            for i, batch in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader),
                                           desc='Sampling from current policy')):
                #if i==1: break
                ct_queries, input_ids, attention_mask, response_gold, gold_label_sequence = batch

                if step < self.params.step_to_start_sampling:
                    break
                if step == 0 and self.params.sample_at_start == 0:
                    break
                elif step == 0 and self.params.sample_at_start == 1:
                    rollouts = self.ref_policy.sample(step=step, input_ids=input_ids, attention_mask=attention_mask,
                                                      max_len=self.params.decoding_len,
                                                      do_sample=self.params.do_sample, top_p=self.params.top_p,
                                                      num_beams=self.params.num_beams)
                else:
                    rollouts = self.policy.sample(step=step, input_ids=input_ids, attention_mask=attention_mask,
                                                  max_len=self.params.decoding_len,
                                                  control_code=self.get_control_code(input_ids),
                                                  do_sample=self.params.do_sample, top_p=self.params.top_p,
                                                  num_beams=self.params.num_beams,
                                                  training_in_stages=self.params.training_in_stages,
                                                  training_in_stages_mode=self.params.training_in_stages_mode,
                                                  stages_interval=self.params.stages_interval)

                prompt, response = rollouts['query/text'], rollouts['response/text']

                prompts.extend(ct_queries)#prompt)
                responses.extend(response)
                responses_gold.extend(response_gold)
                gold_label_sequences.extend(gold_label_sequence)

        # expert rollout
        gt_rollouts = self.expert_rollouts.sample()
        prompts.extend(gt_rollouts['query/text'])
        responses.extend(gt_rollouts['response/text'])
        responses_gold.extend(gt_rollouts['response/text'])
        gold_label_sequences.extend(gt_rollouts['gold_label_sequences'])
        

        if len(prompts) > 0:
            print("Initially sampled data len:", len(prompts))
            task_correctness = [0]*len(prompts)
            inds = []
            inds_correct = []
            for i in range(len(prompts)):
                if "So the answer is" in responses[i]: # checking validity of the sampled datapoint
                   inds.append(i)
                gold_label = responses_gold[i].split('So the answer is')[1].strip().strip('.').lower()
                split_text_pred = responses[i].split('So the answer is')
                if len(split_text_pred)>1:
                    predicted_label = split_text_pred[1].strip().strip('.').lower()
                else:
                    predicted_label = ''
                if ("So the answer is" in responses[i]) and (gold_label==predicted_label):
                    task_correctness[i] = 1
                    inds_correct.append(i) # tracking the correct datapoints

            if self.params.use_only_correct == 1:
                inds = inds_correct

            scores = {}
            for rew in self.score_model.keys():
                if rew == "accuracy":
                    scores[rew] = [task_correctness[i] for i in inds]
                else:
                    scores_rew = self.score_model[rew].get_reward(prompts, responses, responses_gold, f'step{step}', split='train')
                    scores[rew] = [scores_rew[i] for i in inds]
            pickle.dump(scores, open(os.path.join(self.params.reward_dir,'scores_'+str(step)+'.pkl'), 'wb'))
        
            self.data_pool.add(prompts=[prompts[i] for i in inds],
                responses=[responses[i] for i in inds],
                gold_label_sequences=[gold_label_sequences[i] for i in inds],
                scores=scores,
                task_correctness=[task_correctness[i] for i in inds],
                training_in_stages=self.params.training_in_stages,
                training_in_stages_mode=self.params.training_in_stages_mode,
                stages_interval=self.params.stages_interval,
                step=step)
            print("Finally added data len:", len(scores[rew]), sum(task_correctness))
            print("Printing some sampled data...")
            for i in range(5):
               print(i, ':', prompts[i], '\t', responses[i], '\t', gold_label_sequences[i]) 
            #print('\n\n')

        else:
            print("No data sampled or added at step:", step)

        sample_dataset = SequenceDataset(data_pool=self.data_pool)
        print("len of data:", len(sample_dataset), "\n")
        self.sample_dataloader = DataLoader(sample_dataset, batch_size=self.params.batch_size,
                                            shuffle=True, drop_last=True, collate_fn=self.seq_collator)
        self.sampler = iter(self.sample_dataloader)

    def step(self, step_num):
        step_started_at = time.time()

        if self.params.continue_from_checkpoint == 1:
            sample_dataset = SequenceDataset(data_pool=self.data_pool)
            #print("len of data:", len(sample_dataset), "\n")
            self.sample_dataloader = DataLoader(sample_dataset, batch_size=self.params.batch_size,
                                                shuffle=True, drop_last=True, collate_fn=self.seq_collator)
            self.sampler = iter(self.sample_dataloader)

        self.sample(step=step_num)
        #print("Sampling done!", step_num)

        try:
            batch = next(self.sampler)
            assert len(batch[0]) == self.params.batch_size, 'insufficient batch'
        except (StopIteration, AssertionError):
            self.sampler = iter(self.sample_dataloader)
            batch = next(self.sampler)

        ppo_loss, stats = self.loss(step_num, *batch)
        #print("Loss:", ppo_loss)
        ppo_loss.backward()
        if self.params.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.params.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        for metric in ['kl', 'entropy']:
            self.writer.add_scalar(f'Objective/{metric}', stats[f'objective/{metric}'], step_num)
        for metric in ['lm', 'kl', 'entropy', 'total']:
            self.writer.add_scalar(f'Loss/{metric}', stats[f'loss/{metric}'], step_num)
        self.writer.add_scalar(f'Params/lr', self.optimizer.param_groups[0]['lr'], step_num)
        self.writer.add_scalar(f'Params/kl_coef', self.kl_ctl.value, step_num)
        self.writer.add_scalar(f'Params/entropy_coef', self.entropy_ctl.value, step_num)

        self.kl_ctl.update(stats['objective/kl'], self.params.batch_size, True)
        self.entropy_ctl.update(stats['objective/entropy'], self.params.batch_size, False)

        step_time = time.time() - step_started_at
        eps_per_second = float(self.params.batch_size) / step_time
        #print(f"[step {step_num}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")
        
        self.save(step=step_num)
        self.eval_test(step=step_num, eval_type='val')
        self.eval_test(step=step_num, eval_type='test')
        #exit()

    def loss(self, step, query_input_ids, query_mask,
        response_input_ids_wo_tc, response_input_ids_w_tc, response_mask_wo_tc, response_mask_w_tc,
        gold_label_sequences_input_ids, gold_label_sequences_mask):
        #print("In loss function:", step)
        #print(query_input_ids)
        if self.params.task_correctness == 'no':
            response_input_ids = response_input_ids_wo_tc
            response_mask = response_mask_wo_tc
        else:
            response_input_ids = response_input_ids_w_tc
            response_mask = response_mask_w_tc
        outputs = self.policy.forward_pass(step, query_input_ids, query_mask, response_input_ids, response_mask,
                                            gold_label_sequences_input_ids, gold_label_sequences_mask,
                                           has_control_code=self.num_control_codes,
                                           training_in_stages=self.params.training_in_stages,
                                           training_in_stages_mode=self.params.training_in_stages_mode,
                                           stages_interval=self.params.stages_interval)
        lm_loss, logprobs, entropy, logits = outputs['response/lm_loss'], outputs['response/log_prob'], \
                                             outputs['response/entropy'], outputs['response/logits']
        glseq_loss = outputs['response/glseq_loss']
        logits = outputs['response/logits'][:, :, :T5_NUM_TOKEN] # I'm guessing because the remaining tokens are the added tree_tokens
        masks = response_mask[:, self.num_control_codes:].to(self.policy.device)
        gold_label_sequences_mask = gold_label_sequences_mask.to(self.policy.device)
        # go through after writing forward_pass!!!
        with torch.no_grad():
            ref_outputs = self.ref_policy.forward_pass(step, query_input_ids, query_mask,
                                                       response_input_ids[:, self.num_control_codes:],
                                                       response_mask[:, self.num_control_codes:],
                                                       gold_label_sequences_input_ids, gold_label_sequences_mask,
                                                       has_control_code=0)
            ref_logprobs, ref_logits = ref_outputs['response/log_prob'], ref_outputs['response/logits']
            ref_logits = ref_logits[:, :, :T5_NUM_TOKEN]

        kl = torch.sum(self.kl_loss(F.log_softmax(ref_logits, dim=-1), F.softmax(logits, dim=-1)), dim=-1)

        loss = reduce_mean(lm_loss + self.kl_ctl.value * kl - self.entropy_ctl.value * entropy, masks)
        loss = loss + self.params.task_loss_coef*reduce_mean(glseq_loss, gold_label_sequences_mask)
        #print("Checking glseq loss:", reduce_mean(glseq_loss, gold_label_sequences_mask))

        data = {'logprobs': logprobs, 'ref_logprobs': ref_logprobs, 'masks': masks,
                'logits': logits, 'ref_logits': ref_logits,
                'lm_loss': reduce_mean(lm_loss, masks), 'kl_loss': reduce_mean(kl, masks),
                'glseq_loss': reduce_mean(glseq_loss, gold_label_sequences_mask),
                'entropy': reduce_mean(entropy, masks), 'total_loss': loss}
        stats = self.record_step_stats(data)

        queries, responses = self.decode(query_input_ids, response_input_ids)
        self.print_samples(queries=queries, responses=responses, lm_loss=reduce_mean(lm_loss, masks, axis=1),
                           glseq_loss=reduce_mean(glseq_loss, gold_label_sequences_mask, axis=1),
                           logprobs=logprobs, ref_logprobs=ref_logprobs, masks=masks, step=step)

        if step%10==0:
            print(step, ":", loss)
        return loss, stats

    def record_step_stats(self, data):
        masks = data['masks']
        kl = torch.sum(self.kl_loss(F.log_softmax(data['ref_logits'], dim=-1), F.softmax(data['logits'], dim=-1)), dim=-1)
        mean_kl = torch.mean(reduce_sum(kl, masks, axis=1))
        mean_entropy = torch.mean(reduce_sum(-data['logprobs'], masks, axis=1))
        stats = {
            'objective/kl': mean_kl.item(),
            'objective/entropy': mean_entropy.item(),
        }
        stats.update({
            'loss/total': data['total_loss'].item(),
            'loss/kl': data['kl_loss'].item(),
            'loss/lm': data['lm_loss'].item(),
            'loss/entropy': data['entropy'].item(),
        })

        return stats

    def print_samples(self, queries, responses, lm_loss, glseq_loss, logprobs, ref_logprobs, masks, step):
        if step % self.params.log_interval != 0:
            return
            # Log samples
        for i in range(min(3, len(queries))):
            sample_kl = torch.sum((logprobs[i] - ref_logprobs[i]) * masks[i]).item()
            print(queries[i] + responses[i])
            print(f"  lm_loss = {lm_loss[i].item():+.2f}")
            print(f"  glseq_loss = {glseq_loss[i].item():+.2f}")
            print(f"  kl = {sample_kl:+.2f}")
            print(f"  total = {lm_loss[i].item() + self.params.kl_coef * sample_kl:+.2f}")

        # self.writer.add_text('Explanation', f'{queries[0]} ==> {responses[0]}', global_step=step)

    def save(self, step):
        if (step < 1000) or (step == self.params.checkpoint_num_cont_training) or (step % self.params.save_interval != 0):
            return
        torch.save({
            'policy_model': self.policy.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'data_pool': self.data_pool,
            'expert_rollouts': self.expert_rollouts,
        }, f'{self.params.model_dir}/ckp_{step}.pth')
        print(f"[step {step}] model checkpoint saved")

    def eval_test(self, step, eval_type='val'):
        if step % self.params.eval_interval != 0:
            return
        print(f"[step {step}] evaluating " + eval_type + " set ...")

        if eval_type == 'val':
            ct_dataloader = self.val_dataloader
        else:
            ct_dataloader = self.test_dataloader

        # top_p sampling, and sampling it 5 times
        '''sampling_multiple_scores = {}
        for rew in self.score_model.keys():
            sampling_multiple_scores[rew] = defaultdict(list)
        sampling_analyze_generations = {}
        for an_gen in ["num_sents", "num_words"]:
            sampling_analyze_generations[an_gen] = defaultdict(list)
        sampling_multiple_pred_labels = defaultdict(list)
        sampling_multiple_gold_labels = {}
        for sampling_multiple in range(5):
            prompts, responses, responses_gold, perplexities = [], [], [], []
            num_sents, num_words = [], []
            for i, (input_ids, attention_mask, response_gold, gold_label_sequence) in enumerate(tqdm(ct_dataloader)):
                with torch.no_grad():
                    if step == 0:
                        rollouts = self.ref_policy.sample(input_ids=input_ids, attention_mask=attention_mask,
                                                          max_len=self.params.decoding_len,
                                                          do_sample=self.params.do_sample, top_p=self.params.top_p,
                                                          num_beams=self.params.num_beams)
                    else:
                        rollouts = self.policy.sample(input_ids=input_ids, attention_mask=attention_mask,
                                                      max_len=self.params.decoding_len,
                                                      control_code=self.get_control_code(input_ids),
                                                      do_sample=self.params.do_sample, top_p=self.params.top_p,
                                                      num_beams=self.params.num_beams)
                    forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                                      'query_mask': rollouts['query/mask'],
                                      'response_input_ids': rollouts['response/input_ids'],
                                      'response_mask': rollouts['response/mask']}
                    #ref_logprobs = self.ref_policy.forward_pass(**forward_inputs)['response/log_prob']
                    #perplexity = -1. * reduce_sum(ref_logprobs, rollouts['response/mask'].float(), axis=1)
                    #perplexities.extend(perplexity.cpu().detach().numpy().tolist())

                    prompt, response = rollouts['query/text'], rollouts['response/text']
                    prompts.extend(prompt)
                    responses.extend(response)
                    responses_gold.extend(response_gold)
                    num_sents.extend([x.count('.') for x in response])
                    num_words.extend([len(x.split(' ')) for x in response])

            accuracy = self.get_accuracy(responses, responses_gold)
            pred_labels = self.get_pred_labels(responses)
            gold_labels = self.get_pred_labels(responses_gold)
            #ppl_score = np.mean(perplexities)
            scores = {}
            for rew in self.score_model.keys():
                scores[rew] = self.score_model[rew].get_reward(prompts, responses, f'step{step}_eval_' + eval_type + 'sampling')
                reward_score = np.mean([s for s in scores[rew] if isinstance(s, float)])
                print(eval_type + " sampling " + str(sampling_multiple) + " reward = ", reward_score)

            #print("perplexities:", perplexities)
            #print("scores:", scores)
            #print(eval_type + " sampling " + str(sampling_multiple) + " perplexity = ", ppl_score)
            print(eval_type + " sampling " + str(sampling_multiple) + " accuracy = ", accuracy)
            print(eval_type + " sampling " + str(sampling_multiple) + " num_sents = ", np.mean(num_sents))
            print(eval_type + " sampling " + str(sampling_multiple) + " num_words = ", np.mean(num_words))
            # self.writer.add_scalar(eval_type + ' sampling ' + str(sampling_multiple) + ' Evaluation/perplexity', ppl_score, step)
            # self.writer.add_scalar(eval_type + ' sampling ' + str(sampling_multiple) + ' Evaluation/reward', reward_score, step)

            # need to average scores and do majority voting for labels
            for i in range(len(prompts)):
                ct_prompt = prompts[i]
                ct_label = pred_labels[i]
                ct_gold_label = gold_labels[i]
                for rew in self.score_model.keys():
                    ct_score = scores[rew][i]
                    sampling_multiple_scores[rew][ct_prompt].append(ct_score)
                sampling_multiple_pred_labels[ct_prompt].append(ct_label)
                sampling_multiple_gold_labels[ct_prompt] = ct_gold_label
                sampling_analyze_generations["num_words"][ct_prompt].append(num_words[i])
                sampling_analyze_generations["num_sents"][ct_prompt].append(num_sents[i])

        # need to average scores and do majority voting for labels
        all_reward_scores = defaultdict(list)
        majority_accuracy = 0
        for ct_prompt in prompts:
            # for rew in self.score_model.keys():
            #     all_reward_scores[rew].append(np.mean([s for s in sampling_multiple_scores[rew][ct_prompt] if isinstance(s, float)]))
            tmp = sampling_multiple_pred_labels[ct_prompt]
            majority_pred_label = max(set(tmp), key = tmp.count)
            gold_label = sampling_multiple_gold_labels[ct_prompt]
            if majority_pred_label == gold_label:
                majority_accuracy += 1
            for rew in self.score_model.keys():
                tmp_rew_score = []
                for sub_idx in range(len(tmp)):
                    if tmp[sub_idx]==majority_pred_label: # considering only these scores for averaging
                        if isinstance(sampling_multiple_scores[rew][ct_prompt][sub_idx], float):
                            tmp_rew_score.append(sampling_multiple_scores[rew][ct_prompt][sub_idx])
                all_reward_scores[rew].append(np.mean(tmp_rew_score))
            for an_gen in ["num_sents", "num_words"]:
                tmp_nums = []
                for sub_idx in range(len(tmp)):
                    if tmp[sub_idx]==majority_pred_label: # considering only these scores for averaging
                        tmp_nums.append(sampling_analyze_generations[an_gen][ct_prompt][sub_idx])
                all_reward_scores[an_gen].append(np.mean(tmp_nums))

        averaged_reward_score = {}
        reward_scores_str = ''
        for rew in list(self.score_model.keys()) + ["num_sents", "num_words"]:
            averaged_reward_score[rew] = np.mean(all_reward_scores[rew])
            reward_scores_str += rew + ': ' + str(averaged_reward_score[rew]) + ' ' 
        accuracy = float(majority_accuracy)/len(prompts)*100
        with open(os.path.join(self.params.reward_dir, 'reward_scores_' + eval_type + '_sampling.txt'), 'a') as f:
            f.write('Step ' + str(step)+' - '+ reward_scores_str + ' Accuracy: ' + str(accuracy) + '\n')
        '''

        # greedy decoding
        prompts, responses, responses_gold, perplexities = [], [], [], []
        num_sents, num_words = [], []
        for i, (ct_queries, input_ids, attention_mask, response_gold, gold_label_sequence) in enumerate(tqdm(ct_dataloader)):
            with torch.no_grad():
                if step == 0:
                    rollouts = self.ref_policy.sample(step=step, input_ids=input_ids, attention_mask=attention_mask,
                                                      max_len=self.params.decoding_len, do_sample=False)#, top_p=self.params.top_p) # CHANGED
                else:
                    rollouts = self.policy.sample(step=step, input_ids=input_ids, attention_mask=attention_mask,
                                                  max_len=self.params.decoding_len,
                                                  control_code=self.get_control_code(input_ids), do_sample=False,
                                                  # do_sample=self.params.do_sample, top_p=self.params.top_p, # CHANGED
                                                  training_in_stages=self.params.training_in_stages,
                                                  training_in_stages_mode=self.params.training_in_stages_mode,
                                                  stages_interval=self.params.stages_interval)
                forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                                  'query_mask': rollouts['query/mask'],
                                  'response_input_ids': rollouts['response/input_ids'],
                                  'response_mask': rollouts['response/mask']}
                #ref_logprobs = self.ref_policy.forward_pass(**forward_inputs)['response/log_prob']
                #perplexity = -1. * reduce_sum(ref_logprobs, rollouts['response/mask'].float(), axis=1)
                #perplexities.extend(perplexity.cpu().detach().numpy().tolist())

                prompt, response = rollouts['query/text'], rollouts['response/text']
                prompts.extend(ct_queries)#prompt)
                responses.extend(response)
                responses_gold.extend(response_gold)
                num_sents.extend([x.count('.') for x in response])
                num_words.extend([len(x.split(' ')) for x in response])
        
        # printing some
        print("Printing some sampled data during eval:")
        for ps in range(5):
            print(ps, ':', prompts[ps], responses[ps])

        accuracy, correctness_list = self.get_accuracy(responses, responses_gold)

        # to save the individual datapoints
        to_save_dict = {'question': prompts,
                        'model_response': responses,
                        'correctness': correctness_list,
                        'num_sents': num_sents,
                        'num_words': num_words}

        #ppl_score = np.mean(perplexities)
        scores = {}
        reward_scores_str = ''
        for rew in self.score_model.keys():
            if rew == "accuracy":
                continue
            scores[rew] = self.score_model[rew].get_reward(prompts, responses, responses_gold, f'step{step}_eval_' + eval_type + 'greedy', split=eval_type)
            to_save_dict[rew] = scores[rew]
            reward_score = np.nanmean([s for s in scores[rew] if isinstance(s, float)])
            reward_scores_str += rew + ': ' + str(round(reward_score,2)) + ' ' 
            print(eval_type + " greedy reward = ", reward_score)
        reward_scores_str += 'num_sents' + ': ' + str(round(np.mean(num_sents),2)) + ' '
        reward_scores_str += 'num_words' + ': ' + str(round(np.mean(num_words),2)) + ' '
        # ppl_score, reward_score = np.mean(perplexities), np.mean([s for s in scores if isinstance(s, float)])
        #print(eval_type + " greedy perplexity = ", ppl_score)
        # print(eval_type + " greedy reward = ", reward_score)
        print(eval_type + " greedy accuracy = ", accuracy)
        print(eval_type + " greedy num_sents = ", np.mean(num_sents))
        print(eval_type + " greedy num_words = ", np.mean(num_words))
        # self.writer.add_scalar(eval_type + ' greedy Evaluation/perplexity', ppl_score, step)
        # self.writer.add_scalar(eval_type + ' greedy Evaluation/reward', reward_score, step)

        with open(os.path.join(self.params.reward_dir, 'reward_scores_' + eval_type + '_greedy.txt'), 'a') as f:
            f.write('Step ' + str(step)+' - '+ reward_scores_str + ' Accuracy: ' + str(round(accuracy,2)) + '\n')

        # saving the individual datapoints in csv
        with open(os.path.join(self.params.reward_dir,'eval_output' + eval_type + '_greedy_' + str(step) + '.jsonl'), 'w') as csv_f:
            json.dump(to_save_dict, csv_f)
        '''with open(os.path.join(self.params.reward_dir,'eval_output' + eval_type + '_greedy_' + str(step) + '.csv'), 'w') as csv_f:
            writer = csv.writer(csv_f)
            key_list = list(to_save_dict.keys())
            print("keys:", key_list, "\n\n")
            limit = len(to_save_dict[key_list[0]])
            print(to_save_dict)
            print("hi1")
            writer.writerow(key_list)
            print("hi1")
            for keykey in keylist:
                print(keykey)
                print(keykey, len(to_save_dict[keykey]), to_save_dict[keykey])
            for lt in range(limit):
                writer.writerow([to_save_dict[keykey][lt] for keykey in key_list])'''

    def get_pred_labels(self, pred_responses):
        pred_labels = []
        for i in range(len(pred_responses)):
            split_text_pred = pred_responses[i].split('So the answer is')
            if len(split_text_pred) > 1:
                pred_label = split_text_pred[1].strip().strip('.').lower()
            else:
                pred_label = ''
            pred_labels.append(pred_label)

        return pred_labels

    def get_accuracy(self, pred_responses, gold_responses):
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

def load_model(model_name_or_path):
    return load_supervised(model_name_or_path)

def load_supervised(model_path):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    arch = 't5-large' #'google/t5-large-lm-adapt'
    tokenizer = AutoTokenizer.from_pretrained(arch, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(arch)#.to(device)
    if model_path is not None:
        print(model_path)
        model_loaded = torch.load(model_path)
        try:
            states = {k.split('model.')[1]: v for k, v in model_loaded['state_dict'].items()}
            model.load_state_dict(states)
        except:
            model = model_loaded.to(torch.device("cpu"))
        print("model loaded from " + model_path)
    else:
        print("pretrained model loaded")
    return model, tokenizer

def main():
    args = get_args()

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
    #device = torch.device(f"cuda:{num_gpus-1}" if torch.cuda.is_available() else 'cpu')

    if args.continue_from_checkpoint == 0: # new training
        print('Not loading from a checkpoint..')
        time = datetime.now()
        date_time = time.strftime("%m-%d-%Y_%H-%M-%S")
        args.save_dir = os.path.join(args.output_dir, args.dataset_name, date_time + '_' + args.save_number)
        args.reward_dir = os.path.join(args.save_dir, 'reward')
        args.model_dir = os.path.join(args.save_dir, 'model')
        args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')
        for d in [args.output_dir, args.save_dir, args.reward_dir, args.model_dir, args.tensorboard_dir]:
            ensure_dir(d)
        print(f'Write to output directory: {args.save_dir}')

        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else: # training from a checkpoint
        print('Loading from checkpoint:', args.save_dir_cont_training, args.checkpoint_num_cont_training)
        args.save_dir = args.save_dir_cont_training
        args.reward_dir = os.path.join(args.save_dir, 'reward')
        args.model_dir = os.path.join(args.save_dir, 'model')
        args.tensorboard_dir = os.path.join(args.save_dir, 'tensorboard')

    args.n_extra_tokens_list = [int(x) for x in args.n_extra_tokens.split(',')]
    args.rewards_list = args.reward_name.split(',')
    print(args.rewards_list)
    args.reward_filter_scores_list = [float(x) for x in args.reward_filter_score.split(',')]
    train_option = args.train_option.split(',')
    print("train option:", args.train_option)
    assert len(args.rewards_list)==len(args.n_extra_tokens_list)
    assert len(args.rewards_list)==len(args.reward_filter_scores_list)
    assert len(args.rewards_list)==len(train_option)

    tree_tokens = {}
    reward = {}
    reward_filter_scores = {}
    args.train_option = {}
    flant5_model_flag = 0
    flant5_model = None
    for i in range(len(args.n_extra_tokens_list)):
        if args.rewards_list[i] == "accuracy":
            reward_filter_scores[args.rewards_list[i]] = args.reward_filter_scores_list[i]
            args.train_option[args.rewards_list[i]] = train_option[i]
            reward["accuracy"] = None
            # this is always calculated, don't need a reward calculator for it 
            # also don't need tree_tokens because task_correctness_tokens below takes care of it
            continue

        ct_tree_tokens = ['_TOKEN'+str(i)+'_{}'.format(str(idx).zfill(5)) for idx in range(args.n_extra_tokens_list[i])] #+ \
        #              ['_TOKEN'+str(i)+'_ZERO_COMMENTS']
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
                            reward_name=args.rewards_list[i], device=num_gpus - 1,
                            flant5_model=flant5_model)
        reward[args.rewards_list[i]] = ct_reward
        reward_filter_scores[args.rewards_list[i]] = args.reward_filter_scores_list[i]
        args.train_option[args.rewards_list[i]] = train_option[i]
    task_correctness_tokens = ['WRONG_LABEL', 'CORRECT_LABEL']

    if args.reward_as_product == 1:
        tree_tokens = {}
        tree_tokens['reward_product'] = ['_TOKEN'+str(i)+'_{}'.format(str(idx).zfill(5)) for idx in range(args.n_extra_tokens_list[0])]

    print(f'Initializing models ...')
    model, tokenizer = load_model(args.init_model)
    ref_model, ref_tokenizer = load_model(args.ref_model)
    ref_policy = Policy(model=ref_model, tokenizer=ref_tokenizer, temperature=args.temperature, device=device)
    # if len(tree_tokens) > 0:
    policy = Policy(model=model, tokenizer=tokenizer, temperature=args.temperature, device=device,
                    reward_cond=True, tree_tokens=tree_tokens, task_correctness_tokens=task_correctness_tokens)
    # else:
    #     policy = Policy(model=model, tokenizer=tokenizer, temperature=args.temperature, device=device)
    if args.continue_from_checkpoint == 1: # training from a checkpoint
        checkpoint_loaded = torch.load(os.path.join(args.model_dir,
            'ckp_' + str(args.checkpoint_num_cont_training) + '.pth'), map_location='cpu')
        policy.model.load_state_dict(checkpoint_loaded['policy_model'])

    if args.continue_from_checkpoint == 0: # new training
        data_pool = DataPool_N(tree_tokens=tree_tokens, n_extra_tokens_list=args.n_extra_tokens_list,
            rewards_list=args.rewards_list,
            reward_filter_scores=reward_filter_scores,
            train_option=args.train_option,
            task_correctness_tokens=task_correctness_tokens)
    else: # training from a checkpoint
        data_pool = checkpoint_loaded['data_pool']
    print(f'Initialization done!')

    prompt_collator_with_gold = PromptCollator_WithGold(tokenizer=policy.tokenizer)
    train_dataset_no_repeats = PromptDataset(args.dataset_train, args.dataset_name, check_repeats=1)
    train_dataloader_no_repeats = DataLoader(train_dataset_no_repeats, batch_size=150, shuffle=True,
        drop_last=True, collate_fn=prompt_collator_with_gold)
    print(f'Load train set (no repeats) with {len(train_dataset_no_repeats)} examples\n\n')

    train_dataset = PromptDataset(args.dataset_train, args.dataset_name)
    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, collate_fn=prompt_collator_with_gold)
    print(f'Load train set with {len(train_dataset)} examples\n\n')

    val_dataset = PromptDataset(args.dataset_val, args.dataset_name)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=prompt_collator_with_gold)
    print(f'Load val set with {len(val_dataset)} examples\n\n')

    test_dataset = PromptDataset(args.dataset_test, args.dataset_name)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=prompt_collator_with_gold)
    print(f'Load test set with {len(test_dataset)} examples\n\n')

    # reference datapool
    if args.continue_from_checkpoint == 0: # new training
        if args.expert_all_at_first == 1:
            args.num_expert_rollout = len(train_dataset)
        if args.expert_repeats == 1:
            expert_rollouts = ReferenceDataPool(dataset=train_dataset, num_expert_rollout=args.num_expert_rollout)
        else:
            expert_rollouts = ReferenceDataPool(dataset=train_dataset_no_repeats, num_expert_rollout=args.num_expert_rollout)
    else: # training from a checkpoint
        expert_rollouts = checkpoint_loaded['expert_rollouts']

    # set up optimizer and scheduler
    optimizer = Adam(policy.model.parameters(), lr=args.lr, eps=1e-8, betas=(0.9, 0.99))
    # args.total_steps = ceil_div(args.total_episodes, args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.total_steps)
    if args.continue_from_checkpoint == 1: # training from a checkpoint
        optimizer.load_state_dict(checkpoint_loaded['optimizer'])
        scheduler.load_state_dict(checkpoint_loaded['scheduler'])

    trainer = ConditionTrainer(params=args, policy=policy, ref_policy=ref_policy, data_pool=data_pool,
                               score_model=reward, tree_tokens=tree_tokens,
                               task_correctness_tokens=task_correctness_tokens,
                               train_dataloader=train_dataloader_no_repeats, val_dataloader=val_dataloader,
                               test_dataloader=test_dataloader, expert_rollouts=expert_rollouts,
                               optimizer=optimizer, scheduler=scheduler)

    if args.continue_from_checkpoint == 0:
        start_step = 0
    else:
        start_step = args.checkpoint_num_cont_training
    for step_num in range(start_step, args.total_steps+1):
        trainer.step(step_num)
        '''try:
            trainer.step(step_num)
        except:
            torch.cuda.empty_cache()
            continue'''
    print("training done!")

if __name__ == "__main__":
    main()
