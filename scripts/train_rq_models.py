# to train and eval models
import os
import torch
import json
import time
import logging
import random
import pickle
import argparse
import numpy as np
from typing import Union, List, Dict
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSeq2SeqLM

from base_model_arguments import get_args
from utils.utils import mask_pad, ensure_dir, ceil_div, reduce_mean, reduce_sum, logits_to_entropy
from utils.constants import T5_NUM_TOKEN



class PromptDataset(Dataset):
    def __init__(self, path, dataset_name, gen_mode):
        self.data = pd.read_json(path, orient='records')
        #if dataset_name!= "strategyqa":
        self.data = self.data.T
        self.prompts = []
        self.references = []
        self.gold_labels = []
        self.all_choices = []

        # choices
        if dataset_name=="strategyqa":
            all_choices_list = ["no", "yes"]
        elif dataset_name=="coinflip":
            all_choices_list = ["no", "yes"]
        elif dataset_name=="quarel":
            all_choices_list = ["(A)", "(B)"]
        elif dataset_name=="wg":
            all_choices_list = ["(A)", "(B)"]
        elif dataset_name=="qasc":
            all_choices_list = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)"]
        elif dataset_name=="numersense":
            all_choices_list = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)", "(K)", "(L)"]
        elif dataset_name=="penguins_in_a_table":
            all_choices_list = ["(A)", "(B)", "(C)", "(D)", "(E)"]
        elif dataset_name=="obqa":
            all_choices_list = ["(a)", "(b)", "(c)", "(d)"]
        elif dataset_name=="csqa":
            all_choices_list = ["(a)", "(b)", "(c)", "(d)", "(e)"]

        #if dataset_name=="strategyqa":
        #    classes = {0 : "No", 1 : "Yes"}

        for i in range(len(self.data)):
            d = self.data.iloc[i]
            ct_question = d['question']
            ct_rationale = d['rationale']
            if ct_rationale[-1]!='.': ct_rationale = ct_rationale + '.'
            
            #if dataset_name=="strategyqa":
            #    ct_label = classes[d['answer']]
            #else:
            ct_label = d['answer']

            if dataset_name in ["strategyqa", "coinflip"]:
                ct_question = ct_question + ": choice: no choice: yes"
            else:
                for choice_option in all_choices_list:
                    if choice_option in ct_question:
                        ct_question = ct_question.replace(choice_option, "choice: " + choice_option)

            if gen_mode == "i2o":
                ct_question = "explain " + ct_question
            elif gen_mode == "ir2o":
                ct_question = ct_question + " explanation: " + ct_rationale
                    


            self.prompts.append(ct_question)
            self.gold_labels.append(ct_label)
            self.references.append(ct_label)
            self.all_choices.append(all_choices_list)
            
            if i < 2:
                print("Prompt:", self.prompts[i])
                print("Reference:", self.references[i])
                print("Gold label sequence:", self.gold_labels[i])
                print("All choices:", self.all_choices[i])

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {'prompt': self.prompts[idx],
                'reference': self.references[idx],
                'gold_label': self.gold_labels[idx],
                'all_choices': self.all_choices[idx],
                }

# class PromptCollator(object):
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer

#     def __call__(self, sequences):
#         prompts = [sequence['prompt'] for sequence in sequences]
#         encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
#         input_ids = encodings_dict['input_ids']
#         attention_mask = encodings_dict['attention_mask']

#         return input_ids, attention_mask

class PromptCollator_WithGold(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        queries = [sequence['prompt'] for sequence in sequences]
        query_encodings_dict = self.tokenizer(queries, return_tensors="pt", padding=True)
        query_input_ids = query_encodings_dict['input_ids']
        query_attention_mask = query_encodings_dict['attention_mask']

        responses_gold = [sequence['reference'] for sequence in sequences]
        response_encodings_dict = self.tokenizer(responses_gold, return_tensors="pt", padding=True)
        response_input_ids = response_encodings_dict['input_ids']
        response_attention_mask = response_encodings_dict['attention_mask']

        gold_label_sequences = [sequence['gold_label'] for sequence in sequences]

        all_choices = []
        for sequence in sequences:
            ct_all_choices = []
            for choice in sequence['all_choices']:
                choice_encoded = self.tokenizer.encode(choice)
                choice_encoded += [-100]*(20-len(choice_encoded)) # padding
                ct_all_choices.append(choice_encoded)
            all_choices.append(ct_all_choices)

        all_choices = torch.Tensor(all_choices).long()


        return query_input_ids, query_attention_mask, \
                response_input_ids, response_attention_mask, \
                responses_gold, gold_label_sequences, all_choices

class LanguageModel:
    def __init__(self, model, tokenizer, temperature, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        print("Len of tokenizer:", len(self.tokenizer))

        self.model = self.model.to(self.device)
        self.model.parallelize()

        self.temperature = temperature

        self.sf=torch.nn.Softmax(dim=1)

    def sample(self,
               prompts: Union[str, List[str]] = None,
               input_ids: torch.Tensor = None,
               attention_mask: torch.Tensor = None,
               all_choices: torch.Tensor = None,
               all_choices_dict: Dict[int, str] = None,
               num_choices: int = None,
               # max_len: int = 128,
               # min_len: int = 10,
               # do_sample: bool = True,
               # top_k: int = None,
               # top_p: float = None,
               # num_beams: int = None,
               temperature: float = None) -> Dict[str, Union[torch.Tensor, List[str]]]:

        if temperature is None:
            temperature = self.temperature

        if prompts is not None:
            assert input_ids is None and attention_mask is None, 'repeated input'
            if isinstance(prompts, str):
                prompts = [prompts]

            encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = encodings_dict['input_ids'].to(self.device)
            attention_mask = encodings_dict['attention_mask'].to(self.device)

        else:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

        if num_choices is None:
            num_choices = len(all_choices_dict.keys())
            #print("num_choices:", num_choices)

        input_ids = input_ids.repeat(1,num_choices).reshape(-1,input_ids.shape[1])
        attention_mask = attention_mask.repeat(1,num_choices).reshape(-1,attention_mask.shape[1])
        output_labels = all_choices.view(-1, all_choices.size(-1)).to(self.device) # not needed?
        #print("sizes:", input_ids.size(), attention_mask.size(), output_labels.size())

        response_ids = self.model(input_ids=input_ids,
            attention_mask=attention_mask,
            labels=output_labels)

        log_probs = - F.cross_entropy(response_ids.logits.view(-1, response_ids.logits.size(-1)), output_labels.view(-1), ignore_index=-100, reduction='none') 
        log_probs = log_probs.view(-1, output_labels.size(-1)).sum(dim=-1)
        seq_lengths = (output_labels != -100).sum(dim=-1) * 1.0
        log_probs /= seq_lengths
        log_probs = log_probs.view(-1, num_choices)
        log_probs = self.sf(log_probs)
        _, predictions = log_probs.max(dim=1)

        response_text = [all_choices_dict[pred.item()] for pred in predictions]

        # response_ids = response_ids[:, 1:].contiguous()
        #output_mask = (response_ids != self.model.config.pad_token_id).int()

        # response_text = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        #                  for output in response_ids]

        if prompts is None:
            prompts = [self.tokenizer.decode(query, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                       for query in input_ids]

        #print("Sample in policy done", len(response_ids))
        return {
            'query/input_ids': input_ids,
            'query/text': prompts,
            'query/mask': attention_mask,
            'response/input_ids': response_ids,
            'response/text': response_text,
            #'response/mask': output_mask,
        }

    def forward_pass(self,
                     query_input_ids: torch.Tensor,
                     query_mask: torch.Tensor,
                     response_input_ids: torch.Tensor,
                     response_mask: torch.Tensor):

        query_input_ids = query_input_ids.to(self.device)
        query_mask = query_mask.to(self.device)
        response_input_ids = response_input_ids.to(self.device)
        response_mask = response_mask.to(self.device)

        outputs = self.model(
            input_ids=query_input_ids,
            attention_mask=query_mask,
            labels=mask_pad(response_input_ids, response_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,)

        logits = outputs.logits
        log_prob = F.log_softmax(logits, dim=-1)
        output_logprob = torch.gather(log_prob, 2, response_input_ids[:, :, None]).squeeze(2)
        lm_loss = -1. * output_logprob
        
        output_entropy = logits_to_entropy(logits)

        return {
            'response/log_prob': mask_pad(output_logprob, response_mask),
            'response/lm_loss': mask_pad(lm_loss, response_mask),
            'response/entropy': mask_pad(output_entropy, response_mask),
            'response/logits': logits,
        }


class Trainer:
    def __init__(self,
                 params: argparse.Namespace,
                 policy: LanguageModel,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: LambdaLR):

        self.params = params
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.sampler = iter(self.train_dataloader)

        # choices
        if self.params.dataset_name=="strategyqa":
            all_choices_dict = ["no", "yes"]
        elif self.params.dataset_name=="coinflip":
            all_choices_dict = ["no", "yes"]
        elif self.params.dataset_name=="quarel":
            all_choices_dict = ["(A)", "(B)"]
        elif self.params.dataset_name=="wg":
            all_choices_dict = ["(A)", "(B)"]
        elif self.params.dataset_name=="qasc":
            all_choices_dict = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)"]
        elif self.params.dataset_name=="numersense":
            all_choices_dict = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)", "(K)", "(L)"]
        elif self.params.dataset_name=="penguins_in_a_table":
            all_choices_dict = ["(A)", "(B)", "(C)", "(D)", "(E)"]
        elif self.params.dataset_name=="obqa":
            all_choices_dict = ["(a)", "(b)", "(c)", "(d)"]
        elif self.params.dataset_name=="csqa":
            all_choices_dict = ["(a)", "(b)", "(c)", "(d)", "(e)"]
        self.all_choices_dict = {}
        for iii in range(len(all_choices_dict)):
            self.all_choices_dict[iii] = all_choices_dict[iii]

    def decode(self, query_input_ids, response_input_ids):
        query = [self.policy.tokenizer.decode(p, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                 for p in query_input_ids]
        response = [self.policy.tokenizer.decode(r, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    for r in response_input_ids]
        return query, response

    def step(self, step_num):
        step_started_at = time.time()

        try:
            batch = next(self.sampler)
            assert len(batch[0]) == self.params.batch_size, 'insufficient batch'
        except (StopIteration, AssertionError):
            self.sampler = iter(self.train_dataloader)
            batch = next(self.sampler)

        loss = self.loss(step_num, *batch)
        loss.backward()
        if self.params.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.params.max_grad_norm)

        self.optimizer.step()
        if self.params.if_warmup:
            self.scheduler.step()
        self.optimizer.zero_grad()

        step_time = time.time() - step_started_at
        eps_per_second = float(self.params.batch_size) / step_time
        #print(f"[step {step_num}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")
        
        self.save(step=step_num)
        #if step_num>0 and step_num%7000==0:
        #    self.eval_test(step=step_num, eval_type='train')
        self.eval_test(step=step_num, eval_type='val')
        #self.eval_test(step=step_num, eval_type='test')

    def loss(self, step, query_input_ids, query_mask,
        response_input_ids, response_mask, 
        responses_gold_text, gold_label_sequences, all_choices):
        outputs = self.policy.forward_pass(query_input_ids, query_mask, response_input_ids, response_mask)
        lm_loss, logprobs, entropy, logits = outputs['response/lm_loss'], outputs['response/log_prob'], \
                                             outputs['response/entropy'], outputs['response/logits']
        logits = outputs['response/logits'][:, :, :T5_NUM_TOKEN] # I'm guessing because the remaining tokens are the added tree_tokens
        response_mask = response_mask.to(self.policy.device)
        loss = reduce_mean(lm_loss, response_mask)

        queries, responses = self.decode(query_input_ids, response_input_ids)
        self.print_samples(queries=queries, responses=responses, lm_loss=reduce_mean(lm_loss, response_mask, axis=1),
                           logprobs=logprobs, masks=response_mask, step=step)

        if step%20==0:
            print(step, ":", loss)
        return loss

    def print_samples(self, queries, responses, lm_loss, logprobs, masks, step):
        if step % self.params.log_interval != 0:
            return
            # Log samples
        for i in range(min(3, len(queries))):
            print(queries[i] + responses[i])
            print(f"  lm_loss = {lm_loss[i].item():+.2f}")

    def save(self, step):
        if step == 0 or step % self.params.save_interval != 0:
            return
        torch.save(self.policy.model, f'{self.params.model_dir}/ckp_{step}.pth')
        print(f"[step {step}] model checkpoint saved")

    # def get_pred_labels(self, pred_responses):
    #     pred_labels = []
    #     for i in range(len(pred_responses)):
    #         split_text_pred = pred_responses[i].split('The answer is')
    #         if len(split_text_pred) > 1:
    #             pred_label = split_text_pred[1].strip().strip('.').lower()
    #         else:
    #             pred_label = ''
    #         pred_labels.append(pred_label)

    #     return pred_labels

    def get_accuracy(self, pred_responses, gold_responses):
        correct = 0
        for i in range(len(pred_responses)):
            gold_label = gold_responses[i].lower()
            pred_label = pred_responses[i].lower()
            if i < 10:
                print("In get_accuracy:", pred_label, gold_label)

            if pred_label == gold_label:
                correct += 1

        accuracy = float(correct)/len(pred_responses)
        return accuracy*100

    def eval_test(self, step, eval_type='val'):
        if step % self.params.eval_interval != 0:
            return
        print(f"[step {step}] evaluating " + eval_type + " set ...")

        if eval_type == 'val':
            ct_dataloader = self.val_dataloader
        elif eval_type == 'test':
            ct_dataloader = self.test_dataloader
        elif eval_type == 'train':
            ct_dataloader = self.train_dataloader

        # greedy decoding
        prompts, responses, responses_gold, perplexities = [], [], [], []
        num_sents, num_words = [], []
        for i, (query_input_ids, query_attention_mask, dummy_response_input_ids, dummy_response_mask, response_gold, gold_label_sequence, all_choices) in enumerate(tqdm(ct_dataloader)):
            with torch.no_grad():
                rollouts = self.policy.sample(input_ids=query_input_ids, attention_mask=query_attention_mask,
                                              all_choices=all_choices,all_choices_dict=self.all_choices_dict)
                # forward_inputs = {'query_input_ids': rollouts['query/input_ids'],
                #                   'query_mask': rollouts['query/mask'],
                #                   'response_input_ids': rollouts['response/input_ids'],
                #                   'response_mask': rollouts['response/mask']}

                prompt, response = rollouts['query/text'], rollouts['response/text']
                prompts.extend(prompt)
                responses.extend(response)
                responses_gold.extend(response_gold)
                # num_sents.extend([x.count('.') for x in response])
                # num_words.extend([len(x.split(' ')) for x in response])

        for i in range(5):
            print(prompts[i], '--', responses[i], '--', responses_gold[i])
        accuracy = self.get_accuracy(responses, responses_gold)
        # reward_scores_str = ''
        # reward_scores_str += 'num_sents' + ': ' + str(np.mean(num_sents)) + ' '
        # reward_scores_str += 'num_words' + ': ' + str(np.mean(num_words)) + ' '

        print(eval_type + " accuracy = ", accuracy)
        # print(eval_type + " greedy num_sents = ", np.mean(num_sents))
        # print(eval_type + " greedy num_words = ", np.mean(num_words))

        with open(os.path.join(self.params.save_dir, 'op_scores_' + eval_type + '_greedy.txt'), 'a') as f:
            f.write('Step ' + str(step)+' - '+' Accuracy: ' + str(accuracy) + '\n')

def load_model(arch):
    tokenizer = AutoTokenizer.from_pretrained(arch, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(arch)#.to(device)
    return model, tokenizer

def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    num_gpus = torch.cuda.device_count()
    print(f'Detect {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    time = datetime.now()
    date_time = time.strftime("%m-%d-%Y_%H-%M-%S")
    args.save_dir = os.path.join(args.output_dir, args.dataset_name, args.gen_mode, date_time + '_' + args.save_number)
    args.model_dir = os.path.join(args.save_dir, 'model')
    for d in [args.output_dir, args.save_dir, args.model_dir]:
        ensure_dir(d)
    print(f'Write to output directory: {args.save_dir}')

    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print(f'Initializing models ...')
    model, tokenizer = load_model(args.model_type)

    policy = LanguageModel(model=deepcopy(model), tokenizer=deepcopy(tokenizer),
        temperature=args.temperature, device=device)
    print(f'Initialization done!')

    # dataset
    prompt_collator_with_gold = PromptCollator_WithGold(tokenizer=policy.tokenizer)
    train_dataset = PromptDataset(args.dataset_train, args.dataset_name, args.gen_mode)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=prompt_collator_with_gold)
    print(f'Load train set with {len(train_dataset)} examples\n\n')

    val_dataset = PromptDataset(args.dataset_val, args.dataset_name, args.gen_mode)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=prompt_collator_with_gold)
    print(f'Load val set with {len(val_dataset)} examples\n\n')

    test_dataset = PromptDataset(args.dataset_test, args.dataset_name, args.gen_mode)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=prompt_collator_with_gold)
    print(f'Load test set with {len(test_dataset)} examples\n\n')

    # set up optimizer and scheduler
    optimizer = Adam(policy.model.parameters(), lr=args.lr, eps=1e-8, betas=(0.9, 0.99))
    # number of steps
    args.num_train_steps_per_epoch = int(len(train_dataset)/args.batch_size)
    args.total_steps = int(args.num_epochs*args.num_train_steps_per_epoch)
    print("total_steps:", args.total_steps)
    scheduler = None
    if args.if_warmup:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.total_steps)

    trainer = Trainer(params=args, policy=policy, 
            train_dataloader=train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader,
            optimizer=optimizer, scheduler=scheduler)

    for step_num in range(args.total_steps):
        trainer.step(step_num)
    print("training done!")

if __name__ == "__main__":
    main()
