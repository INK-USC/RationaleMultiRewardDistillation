import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import copy
from tqdm import tqdm

def load_supervised_rq(model_path, reward_model_size, device):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    arch = 't5-'+'base'#reward_model_size #'google/t5-large-lm-adapt'
    tokenizer = AutoTokenizer.from_pretrained(arch, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(arch)#.to(device)
    model.to("cuda:0") # preliminary!!
    if model_path is not None:
        #states = {k.split('model.')[1]: v for k, v in torch.load(model_path)['state_dict'].items()}
        #model.load_state_dict(states)
        model_loaded = torch.load(model_path)
        model_state_dict = model_loaded.state_dict()
        model.load_state_dict(model_state_dict)
        #model = model.to(torch.device("cpu"))
        print("model loaded from " + model_path)
    else:
        print("pretrained model loaded")
    model.to(device)
    print("model kept in device:", device)
    model.eval()
    return model, tokenizer

def get_rq_scores(device, mode,
    i2o_model, i2o_tokenizer, ir2o_model, ir2o_tokenizer,
    prompts, responses_gold, responses,
    dataset_name, dataset_choices):
	# first get pred rationales and labels
	# then get gold rationales and labels

    batch_size = 20
	
    pred_rationales = []
    pred_labels = []
    gold_label_inds = []
    all_output_labels = []
    i2o_inputs = []
    ir2o_inputs = []
    for i in range(len(prompts)):
        ct_question = prompts[i]
        ct_response = responses[i]
        ct_gold = responses_gold[i]
        tmp = ct_response.split("So the answer is ")
        ct_pred_rationale = tmp[0]
        if len(tmp) > 1:
            ct_pred_label = tmp[1].strip().strip('.')
        else:
            ct_pred_label = ''
        ct_gold_label = ct_gold.split("So the answer is ")[1].strip().strip('.')
        ct_gold_label_idx = dataset_choices.index(ct_gold_label)

        if dataset_name in ["strategyqa", "coinflip"]:
            ct_question = ct_question + ": choice: no choice: yes"
        else:
            for choice_option in dataset_choices:
                if choice_option in ct_question:
                    ct_question = ct_question.replace(choice_option, "choice: " + choice_option)

        ct_i2o_input = "explain " + ct_question
        ct_ir2o_input = ct_question + " explanation: " + ct_pred_rationale

        pred_rationales.append(ct_pred_rationale)
        pred_labels.append(ct_pred_label)
        gold_label_inds.append(ct_gold_label_idx)
        all_output_labels.append(copy.deepcopy(dataset_choices))
        i2o_inputs.append(ct_i2o_input)
        ir2o_inputs.append(ct_ir2o_input)

    # running the i2o and ir2o models
    sf=torch.nn.Softmax(dim=1)
    steps = int(len(prompts)/batch_size) + 1
    discrete_rq_scores = []
    continuous_rq_scores = []
    for i in tqdm(range(steps)):
        start_idx = i*batch_size
        try:
            end_idx = (i+1)*batch_size
            batch_prompts = prompts[start_idx:end_idx]
        except:
            end_idx = len(prompts) - 1
            batch_prompts = prompts[start_idx:end_idx]

        if len(batch_prompts) == 0:
            continue

        # i2o inputs
        encodings_dict_i2o = i2o_tokenizer(i2o_inputs[start_idx:end_idx], return_tensors="pt", padding=True)
        input_ids_i2o = encodings_dict_i2o['input_ids'].to(device)
        attention_mask_i2o = encodings_dict_i2o['attention_mask'].to(device)
        input_ids_i2o = input_ids_i2o.repeat(1,len(dataset_choices)).reshape(-1,input_ids_i2o.shape[1])
        attention_mask_i2o = attention_mask_i2o.repeat(1,len(dataset_choices)).reshape(-1, attention_mask_i2o.shape[1])

        # ir2o inputs
        encodings_dict_ir2o = ir2o_tokenizer(ir2o_inputs[start_idx:end_idx], return_tensors="pt", padding=True)
        input_ids_ir2o = encodings_dict_ir2o['input_ids'].to(device)
        attention_mask_ir2o = encodings_dict_ir2o['attention_mask'].to(device)
        input_ids_ir2o = input_ids_ir2o.repeat(1,len(dataset_choices)).reshape(-1,input_ids_ir2o.shape[1])
        attention_mask_ir2o = attention_mask_ir2o.repeat(1,len(dataset_choices)).reshape(-1, attention_mask_ir2o.shape[1])

        # output labels
        batch_output_labels = all_output_labels[start_idx:end_idx]
        batch_output_label_ids_i2o = []
        batch_output_label_ids_ir2o = []
        for ii in range(len(batch_output_labels)):
            tmp_output_labels_i2o = []
            tmp_output_labels_ir2o = []
            for ll in batch_output_labels[ii]:
                ll_ids_i2o = i2o_tokenizer.encode(ll)
                ll_ids_i2o += [-100]*(10-len(ll_ids_i2o))
                tmp_output_labels_i2o.append(ll_ids_i2o)

                ll_ids_ir2o = ir2o_tokenizer.encode(ll)
                ll_ids_ir2o += [-100]*(10-len(ll_ids_ir2o))
                tmp_output_labels_ir2o.append(ll_ids_ir2o)

            batch_output_label_ids_i2o.append(tmp_output_labels_i2o)
            batch_output_label_ids_ir2o.append(tmp_output_labels_ir2o)

        batch_output_label_ids_i2o = torch.Tensor(batch_output_label_ids_i2o).long().to(device)
        batch_output_label_ids_ir2o = torch.Tensor(batch_output_label_ids_ir2o).long().to(device)
        if 0: #i < 5:
            print("batch_output_label_ids_i2o: ", batch_output_label_ids_i2o)
            print("batch_output_label_ids_ir2o: ", batch_output_label_ids_ir2o)
        batch_output_label_ids_i2o = batch_output_label_ids_i2o.view(-1, batch_output_label_ids_i2o.size(-1))
        batch_output_label_ids_ir2o = batch_output_label_ids_ir2o.view(-1, batch_output_label_ids_ir2o.size(-1))


        # running i2o
        
        '''
        print("device of input etc.:", device)
        print("ip:", input_ids_i2o.device)
        print("att:", attention_mask_i2o.device)
        print("label ids:", batch_output_label_ids_i2o.device)
        print("i2o model:", i2o_model.device)
        print("ir2o model:", ir2o_model.device)
        '''
        with torch.no_grad():
            response_ids_i2o = i2o_model(input_ids=input_ids_i2o,
                attention_mask=attention_mask_i2o,
                labels=batch_output_label_ids_i2o)
            #print(response_ids_i2o.device)
            # running ir2o
            response_ids_ir2o = ir2o_model(input_ids=input_ids_ir2o,
                attention_mask=attention_mask_ir2o,
                labels=batch_output_label_ids_ir2o)

        # i2o probabs
        log_probs_i2o = - F.cross_entropy(response_ids_i2o.logits.view(-1, response_ids_i2o.logits.size(-1)),
            batch_output_label_ids_i2o.view(-1), ignore_index=-100, reduction='none') 
        log_probs_i2o = log_probs_i2o.view(-1, batch_output_label_ids_i2o.size(-1)).sum(dim=-1)
        seq_lengths_i2o = (batch_output_label_ids_i2o != -100).sum(dim=-1) * 1.0
        log_probs_i2o /= seq_lengths_i2o
        log_probs_i2o = log_probs_i2o.view(-1, len(dataset_choices))
        log_probs_i2o = sf(log_probs_i2o)
        _, predictions_i2o = log_probs_i2o.max(dim=1)

        # ir2o probabs
        log_probs_ir2o = - F.cross_entropy(response_ids_ir2o.logits.view(-1, response_ids_ir2o.logits.size(-1)),
            batch_output_label_ids_ir2o.view(-1), ignore_index=-100, reduction='none') 
        log_probs_ir2o = log_probs_ir2o.view(-1, batch_output_label_ids_ir2o.size(-1)).sum(dim=-1)
        seq_lengths_ir2o = (batch_output_label_ids_ir2o != -100).sum(dim=-1) * 1.0
        log_probs_ir2o /= seq_lengths_ir2o
        log_probs_ir2o = log_probs_ir2o.view(-1, len(dataset_choices))
        log_probs_ir2o = sf(log_probs_ir2o)
        _, predictions_ir2o = log_probs_ir2o.max(dim=1)

        # getting the continuous and discrete stuff for RQ
        batch_gold_label_inds = gold_label_inds[start_idx:end_idx]
        for ii in range(len(batch_gold_label_inds)):
            ct_gold_label_idx = batch_gold_label_inds[ii]
            ct_i2o_pred = predictions_i2o[ii].item()
            ct_ir2o_pred = predictions_ir2o[ii].item()

            # discrete
            if ct_ir2o_pred != ct_gold_label_idx:
                ct_discrete = -1
            if (ct_ir2o_pred == ct_gold_label_idx) and (ct_i2o_pred == ct_gold_label_idx):
                ct_discrete = 0
            if (ct_ir2o_pred == ct_gold_label_idx) and (ct_i2o_pred != ct_gold_label_idx):
                ct_discrete = 1
            # continuous
            ct_continuous = log_probs_ir2o[ii][ct_gold_label_idx].item() - log_probs_i2o[ii][ct_gold_label_idx].item()

            # appending
            discrete_rq_scores.append(float(ct_discrete))
            continuous_rq_scores.append(float(ct_continuous))

    if mode == "discrete":
        assert len(discrete_rq_scores)==len(prompts)
        return discrete_rq_scores
    elif mode == "continuous":
        assert len(continuous_rq_scores)==len(prompts)
        return continuous_rq_scores

def main():
    device = torch.device('cuda:0')
    i2o_model, i2o_tokenizer = load_supervised_rq('save/quarel/i2o/04-13-2023_10-27-16_1/model/ckp_5000.pth', 'large', device)
    ir2o_model, ir2o_tokenizer = load_supervised_rq('save/quarel/ir2o/04-18-2023_16-04-48_1/model/ckp_3000.pth', 'large', device)
    dataset_name = 'quarel'
    dataset_choices = ['(A)', '(B)']
    mode = 'discrete'

    prompts = ['blah blah', 'nnnnnnnnn', 'kksd sndjsnd sdjsndjnd']
    responses = ['The answer is (A)', 'The answer is (B)', 'The answer is (A)']
    responses_gold = ['The answer is (A)', 'The answer is (B)', 'The answer is (A)']
    scores = get_rq_scores(device=device, mode=mode, i2o_model=i2o_model, i2o_tokenizer=i2o_tokenizer, ir2o_model=ir2o_model, ir2o_tokenizer=ir2o_tokenizer, prompts=prompts, responses_gold=responses_gold, responses=responses, dataset_name=dataset_name, dataset_choices=dataset_choices)

if __name__=="__main__":
    main()
