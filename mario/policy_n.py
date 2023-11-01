import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict
from utils.constants import NEGATIVE_INF
from utils.utils import logits_to_entropy, mask_pad
from utils.constants import T5_NUM_TOKEN

class Policy:
    def __init__(self, model, tokenizer, temperature, device, reward_cond=False, tree_tokens=None, task_correctness_tokens=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        print("Original tokenizer:", len(self.tokenizer))
        if reward_cond:
            tree_tokens_all = []
            for rew in tree_tokens.keys():
                tree_tokens_all.extend(tree_tokens[rew])
            print("Adding to tokenizer:", tree_tokens_all, task_correctness_tokens)      
            self.tokenizer.add_tokens(tree_tokens_all, special_tokens=True)
            self.tokenizer.add_tokens(task_correctness_tokens, special_tokens=True)

            weights = self.model.get_input_embeddings().weight.detach().numpy()
            mean_weights, std_weights = np.mean(weights, axis=0), np.std(weights, axis=0)
            new_inits = np.vstack([np.random.normal(loc=mean_weights, scale=std_weights) for _ in tree_tokens_all + task_correctness_tokens])

            self.model.resize_token_embeddings(len(self.tokenizer))
            with torch.no_grad():
                new_inits = torch.tensor(new_inits)
                self.model.get_input_embeddings().weight[-len(tree_tokens_all + task_correctness_tokens):, :] = new_inits
            print("New tokenizer:", len(self.tokenizer))

        self.model = self.model.to(self.device)
        print("policy model's device:", self.model.device)
        #self.model.parallelize()

        self.temperature = temperature

    def sample(self,
               step: int = None,
               prompts: Union[str, List[str]] = None,
               input_ids: torch.Tensor = None,
               attention_mask: torch.Tensor = None,
               control_code: torch.Tensor = None,
               max_len: int = 128,
               min_len: int = 10,
               do_sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               num_beams: int = None,
               temperature: float = None,
               training_in_stages: str = "no",
               training_in_stages_mode: str = "add_to_right",
               stages_interval: int = -1) -> Dict[str, Union[torch.Tensor, List[str]]]:

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

        if control_code is not None:
            control_code = control_code.to(self.device)
            num_control_codes = control_code.size()[1]
            if training_in_stages == "yes":
                assert stages_interval > 0
                # if you're at a multiple of stages interval, you shouldn't change the sampling till you've trained on it
                # adding to right
                if training_in_stages_mode == "add_to_right":
                    ct_interval = step//stages_interval + 1
                    if step//stages_interval==step/stages_interval: # i.e., a multiple
                        ct_interval = ct_interval - 1
                    if ct_interval < num_control_codes:
                        control_code = control_code[:, 0:ct_interval]
                # adding to left
                elif training_in_stages_mode == "add_to_left":
                    ct_interval = step//stages_interval + 1
                    if step//stages_interval==step/stages_interval: # i.e., a multiple
                        ct_interval = ct_interval - 1
                    if ct_interval < num_control_codes:
                        control_code = control_code[:, -1*ct_interval:]

                # adding to right in groups
                elif "add_to_right_groups_" in training_in_stages_mode:
                    tmp = training_in_stages_mode.replace("add_to_right_groups_", "")
                    stages_groups = [int(gg) for gg in tmp.split("-")]
                    assert sum(stages_groups) == num_control_codes # the groups must add up to the whole set of rewards
                    ct_interval = step//stages_interval + 1
                    if step//stages_interval==step/stages_interval: # i.e., a multiple
                        ct_interval = ct_interval - 1
                    if ct_interval > len(stages_groups):
                        pass
                    else:
                        ct_interval_new = sum(stages_groups[0:ct_interval])
                        control_code = control_code[:, 0:ct_interval_new]

                # adding to left in groups
                elif "add_to_left_groups_" in training_in_stages_mode:
                    tmp = training_in_stages_mode.replace("add_to_left_groups_", "")
                    stages_groups = [int(gg) for gg in tmp.split("-")]
                    stages_groups.reverse()
                    assert sum(stages_groups) == num_control_codes # the groups must add up to the whole set of rewards
                    ct_interval = step//stages_interval + 1
                    if step//stages_interval==step/stages_interval: # i.e., a multiple
                        ct_interval = ct_interval - 1
                    if ct_interval > len(stages_groups):
                        pass
                    else:
                        ct_interval_new = sum(stages_groups[0:ct_interval])
                        control_code = control_code[:, -1*ct_interval_new:]

                if True: #(step-1)%stages_interval == 0:
                    print("control_code:", control_code.shape)
                    print("control_code:", control_code[0, :5])
            #print(input_ids.shape, control_code.shape)
            bos_ids = self.model._prepare_decoder_input_ids_for_generation(batch_size=input_ids.shape[0])
            decoder_input_ids = torch.cat([bos_ids, control_code], dim=1)

            response_ids = self.model.generate(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               max_length=max_len,
                                               min_length=min_len,
                                               do_sample=do_sample,
                                               top_k=top_k,
                                               top_p=top_p,
                                               num_beams=num_beams,
                                               temperature=temperature,
                                               decoder_input_ids=decoder_input_ids)
            # since we don't know how many control codes there are
            num_to_ignore = decoder_input_ids.shape[-1]
            response_ids = response_ids[:, num_to_ignore:].contiguous()

        else:
            response_ids = self.model.generate(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               max_length=max_len,
                                               min_length=min_len,
                                               do_sample=do_sample,
                                               top_k=top_k,
                                               top_p=top_p,
                                               num_beams=num_beams,
                                               temperature=temperature)
            response_ids = response_ids[:, 1:].contiguous()
        output_mask = (response_ids != self.model.config.pad_token_id).int()

        response_text = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                         for output in response_ids]

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
            'response/mask': output_mask,
        }

    def forward_pass(self,
                     step: int,
                     query_input_ids: torch.Tensor,
                     query_mask: torch.Tensor,
                     response_input_ids: torch.Tensor,
                     response_mask: torch.Tensor,
                     gold_label_sequences_input_ids: torch.Tensor,
                     gold_label_sequences_mask: torch.Tensor,
                     has_control_code: int = 0,
                     training_in_stages: str = "no",
                     training_in_stages_mode: str = "add_to_right",
                     stages_interval: int = -1):

        query_input_ids = query_input_ids.to(self.device)
        query_mask = query_mask.to(self.device)
        response_input_ids = response_input_ids.to(self.device)
        response_mask = response_mask.to(self.device)
        gold_label_sequences_input_ids = gold_label_sequences_input_ids.to(self.device)
        gold_label_sequences_mask = gold_label_sequences_mask.to(self.device)

        if has_control_code > 0:
            if training_in_stages == "no":
                pass
            else:
                assert stages_interval > 0
                # adding to right
                if training_in_stages_mode == "add_to_right":
                    ct_interval = step//stages_interval + 1
                    if ct_interval > has_control_code:
                        pass
                    else:
                        response_input_ids = torch.cat([response_input_ids[:, 0:ct_interval],
                                                response_input_ids[:, has_control_code:]], dim=1)
                        response_mask = torch.cat([response_mask[:, 0:ct_interval],
                                                response_mask[:, has_control_code:]], dim=1)
                        has_control_code = ct_interval
                # adding to left
                elif training_in_stages_mode == "add_to_left":
                    ct_interval = step//stages_interval + 1
                    if ct_interval > has_control_code:
                        pass
                    else:
                        response_input_ids = response_input_ids[:, (has_control_code-ct_interval):]
                        response_mask = response_mask[:, (has_control_code-ct_interval):]
                        has_control_code = ct_interval

                # adding to right in groups
                elif "add_to_right_groups_" in training_in_stages_mode:
                    tmp = training_in_stages_mode.replace("add_to_right_groups_", "")
                    stages_groups = [int(gg) for gg in tmp.split("-")]
                    assert sum(stages_groups) == has_control_code # the groups must add up to the whole set of rewards
                    ct_interval = step//stages_interval + 1
                    if ct_interval > len(stages_groups):
                        pass
                    else:
                        ct_interval_new = sum(stages_groups[0:ct_interval])
                        response_input_ids = torch.cat([response_input_ids[:, 0:ct_interval_new],
                                                response_input_ids[:, has_control_code:]], dim=1)
                        response_mask = torch.cat([response_mask[:, 0:ct_interval_new],
                                                response_mask[:, has_control_code:]], dim=1)
                        has_control_code = ct_interval_new

                # adding to left in groups
                elif "add_to_left_groups_" in training_in_stages_mode:
                    tmp = training_in_stages_mode.replace("add_to_left_groups_", "")
                    stages_groups = [int(gg) for gg in tmp.split("-")]
                    stages_groups.reverse()
                    assert sum(stages_groups) == has_control_code # the groups must add up to the whole set of rewards
                    ct_interval = step//stages_interval + 1
                    if ct_interval > len(stages_groups):
                        pass
                    else:
                        ct_interval_new = sum(stages_groups[0:ct_interval])
                        response_input_ids = response_input_ids[:, (has_control_code-ct_interval_new):]
                        response_mask = response_mask[:, (has_control_code-ct_interval_new):]
                        has_control_code = ct_interval_new


                # printing for verification purposes
                if (step-1)%stages_interval == 0:
                    print("num_control_codes:", has_control_code)
                    print("response_input_ids:", response_input_ids.shape)
                    print("response_mask:", response_mask.shape)
                    print("response_input_ids:", response_input_ids[:,:5])

        outputs = self.model(
            input_ids=query_input_ids,
            attention_mask=query_mask,
            labels=mask_pad(response_input_ids, response_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,)

        if has_control_code != 0:
            logits = outputs.logits[:, has_control_code:]
            response_input_ids = response_input_ids[:, has_control_code:]
            response_mask = response_mask[:, has_control_code:]
        else:
            logits = outputs.logits

        if training_in_stages == "yes" and has_control_code > 0:
            if step%stages_interval == 0:
                print("num_control_codes:", has_control_code)
                print("response_input_ids:", response_input_ids.shape)
                print("response_input_ids:", response_input_ids[:,:5])
                print("response_mask:", response_mask.shape)
                print("logits:", logits.shape)
        #print("LOSS:", outputs.loss)
        #print("LABELS:", mask_pad(response_input_ids, response_mask, -100))
        #print("LOGITS:", logits,"\n")
        log_prob = F.log_softmax(logits, dim=-1)
        output_logprob = torch.gather(log_prob, 2, response_input_ids[:, :, None]).squeeze(2)
        lm_loss = -1. * output_logprob
        
        output_entropy = logits_to_entropy(logits)
        '''except:
            print("LOGITS ARE NAN HERE!!!")
            print("Num of nan:", torch.sum(torch.isnan(logits)))
            print("Logits shape:", logits.shape)
            output_entropy = torch.zeros(lm_loss.shape)'''

        # Doing the task output (alone) cross-entropy loss
        # first get the index of the eos token
        predicted_ids = torch.argmax(logits[:, :, :T5_NUM_TOKEN], dim=2)
        #print("predicted_ids:", predicted_ids)
        #print("eos token id:", self.tokenizer.eos_token_id)
        finding_eos_token = predicted_ids==self.tokenizer.eos_token_id
        eos_token_index = []
        for ii in range(len(finding_eos_token)):
            try:
                ct_eos = finding_eos_token[ii].tolist().index(True)
                assert ct_eos + 1 >= gold_label_sequence_length
            except:
                ct_eos = len(finding_eos_token[ii].tolist())-2
            eos_token_index.append(ct_eos)
        #eos_token_index = (predicted_ids==self.tokenizer.eos_token_id).nonzero()[:,1]
        # now get just the part of the sequence that corresponds to the task output
        #print("logits:", logits.shape) 
        l = []
        gold_label_sequence_length = gold_label_sequences_input_ids.size()[-1]
        req_logits = torch.zeros(logits.size()[0], gold_label_sequence_length, T5_NUM_TOKEN).to(self.device)
        #print("req_logits init:", req_logits.shape)
        #print("eos_token_index:", eos_token_index)
        for ii,xx in enumerate(eos_token_index):
            ct_eos_index = xx#.item()
            req_logits[ii] = logits[ii, (ct_eos_index - gold_label_sequence_length + 1):(ct_eos_index + 1), :T5_NUM_TOKEN]
        # now calculate loss
        log_prob_glseq = F.log_softmax(req_logits, dim=-1)
        output_logprob_glseq = torch.gather(log_prob_glseq, 2, gold_label_sequences_input_ids[:,:,None]).squeeze(2)
        glseq_loss = -1. * output_logprob_glseq

        #print("req_logits:", req_logits,"\n")
        #print("glseq logits:", gold_label_sequences_input_ids, "\n")

        return {
            'response/log_prob': mask_pad(output_logprob, response_mask),
            'response/lm_loss': mask_pad(lm_loss, response_mask),
            'response/glseq_loss': mask_pad(glseq_loss, gold_label_sequences_mask),
            'response/entropy': mask_pad(output_entropy, response_mask),
            'response/logits': logits,
        }
