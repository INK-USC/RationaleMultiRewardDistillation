import torch
import argparse

CKPT_PATH=''

def get_args():
    parser = argparse.ArgumentParser(description='MaRio')

    # loading from checkpoints
    # not that the same arguments have to be given
    parser.add_argument(
        '--continue_from_checkpoint', type=int, default=0, help='whether to continue training from some checkpoint')
    parser.add_argument(
        '--save_dir_cont_training', type=str, default='', help='path to save dir with trained checkpoints')
    parser.add_argument(
        '--checkpoint_num_cont_training', type=int, default=-1,
        help='which checkpoint number to continue training from')

    # dataset
    parser.add_argument(
        '--output-dir', type=str, default='../save_dir')
    parser.add_argument(
        '--save_number', type=str, default='1')
    parser.add_argument(
        '--dataset-name', type=str, default='strategyqa', help='name of dataset')
    parser.add_argument(
        '--dataset-train', type=str, default='data/strategyqa/raw/strategyqa_processed_train.json',
        help='JSONL file containing train data. Each row must contain a prompt at `row["question"]`.')
    parser.add_argument(
        '--dataset-val', type=str, default='data/strategyqa/raw/strategyqa_processed_dev.json',
        help='JSONL file containing val data. Each row must contain a prompt at `row["question"]`.')
    parser.add_argument(
        '--dataset-test', type=str, default='data/strategyqa/raw/strategyqa_processed_test.json',
        help='JSONL file containing test data. Each row must contain a prompt at `row["question"]`.')

    # reward
    parser.add_argument('--reward_as_product', type=int, default=0, help='whether to product the rewards')
    parser.add_argument(
        '--n_extra_tokens', type=str, default='5', help='number of reward categorization')
    parser.add_argument(
        '--task_correctness', type=str, default='yes', help='yes/no if to have a task correctness tag')
    parser.add_argument(
        '--use_only_correct', type=int, default=0, help='0/1 if to use only correct train datapoints')
    parser.add_argument(
        '--sample-interval', type=int, default=500, help='step interval to sample from current policy')
    parser.add_argument(
        '--horizon', type=float, default=2500, help='horizon value in adaptive controller')
    # task loss term
    parser.add_argument(
        '--task_loss_coef', type=float, default=1.0, help='coefficient for task loss')
    # KL term
    parser.add_argument(
        '--kl_coef', type=float, default=0.05, help='coefficient for KL term in reward')
    parser.add_argument(
        '--adaptive_kl', action='store_true', default=False, help='whether to use adaptive KL controller')
    parser.add_argument(
        '--target_kl', type=float, default=3, help='target value in adaptive KL controller')
    # entropy term
    parser.add_argument(
        '--entropy_coef', type=float, default=0.05, help='coefficient for entropy term in reward')
    parser.add_argument(
        '--adaptive_entropy', action='store_true', default=False, help='whether to use adaptive entropy controller')
    parser.add_argument(
        '--target_entropy', type=float, default=40, help='target value in adaptive entropy controller')
    # need to change
    parser.add_argument(
        '--reward_name', type=str, default='', help='reward to use')
    parser.add_argument(
        '--flan_t5_size', type=str, default='xxl', help='size of flan-t5 when reward being used is flan-t5-factuality')
    parser.add_argument(
        '--reward_filter_score', type=str, default='0.4', help='list of filter scores for rewards')
    parser.add_argument(
        '--reward_batch_size', type=int, default=4, help='batch size to compute reward')
    # overall option
    parser.add_argument(
        '--train_option', type=str, default='quark',
        help='list of training options for each reward: quark, filter, quark_no_tag, filter_quark')

    # policy
    parser.add_argument(
        '--init-model', type=str, default=None, help='language model used for policy.')
    parser.add_argument(
        '--ref-model', type=str, default=None, help='language model used for reference policy.')
    parser.add_argument(
        '--temperature', type=float, default=1.0, help='temperature for sampling policy.')
    parser.add_argument(
        '--num_expert_rollout', type=int, default=800, help='number of rollouts that use expert policy')
    parser.add_argument(
        '--expert_repeats', type=int, default=1, help='whether expert should use all the train data or just one of each question')
    parser.add_argument(
        '--expert_all_at_first', type=int, default=0, help='whether expert should give all the train data at the start, or phase it out over the steps')
    parser.add_argument(
        '--num_policy_rollout', type=int, default=2, help='number of rollouts that use current policy')

    # training
    parser.add_argument(
        '--total-steps', type=int, default=10000, help='total number of episodes')
    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size')
    parser.add_argument(
        '--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument(
        '--num_warmup_steps', type=int, default=1000, help='number of warmup steps in lr scheduler')
    parser.add_argument(
        '--clip_grad', action='store_true', default=True, help='whether to clip gradient')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5, help='maximum norm of gradients ')
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=2, help="gradient accumulation step.")
    parser.add_argument(
        '--save_interval', type=int, default=1000, help="Num steps after which to save.")

    # training in stages
    parser.add_argument(
        '--training_in_stages', type=str, default="no", help="whether to train rewards in stages")
    parser.add_argument(
        '--training_in_stages_mode', type=str, default="add_to_right", help="how to add rewards over the stages")
    parser.add_argument(
        '--stages_interval', type=int, default=-1,
        help="interval at which to add new rewards, must be > 0 if training_in_stages=yes")

    # generation
    parser.add_argument(
        '--prefix_len', type=int, default=32, help='number of tokens in each prompt.')
    parser.add_argument(
        '--decoding_len', type=int, default=256, help='number of tokens to generate for each prompt.')
    parser.add_argument(
        '--top_p', type=float, default=0.9, help='hyperparameter for nucleus sampling')
    parser.add_argument(
        '--num_beams', type=int, default=None, help='hyperparameter for beam search')
    parser.add_argument(
        '--do_sample', action='store_false', default=True, help='whether to use sampling to decode')
    parser.add_argument(
        '--sample_at_start', type=int, default=1, help='whether to use sample for train data at step 0 or not')
    parser.add_argument(
        '--step_to_start_sampling', type=int, default=0, help='when to start sampling train data')

    # other
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=1000, help='step interval to print out logs')
    parser.add_argument(
        '--save-interval', type=int, default=1000, help='step interval to save model checkpoints')
    parser.add_argument(
        '--eval-interval', type=int, default=1000, help='step interval to do evaluation')
    parser.add_argument(
        '--cuda-deterministic', action='store_false', default=True,
        help="sets flags for determinism when using CUDA (potentially slow!)")

    # only for evaluation
    parser.add_argument(
        '--save_dir_for_eval', type=str, default="", help='path to save dir')
    parser.add_argument(
        '--ckp_num_for_eval', type=str, default="all", help='checkpoints to evaluate')
    parser.add_argument(
        '--reward_names_other', type=str, default="", help='other rewards to evaluate')
    parser.add_argument(
        '--actual_rewards_to_calculate', type=str, default="",
        help='actual set of rewards to calculate, defaults to reward_name')
    # if evaluating rationales from other sources
    parser.add_argument(
        '--outputs_other_sources_file', type=str, default="", help='other outputs to evaluate')
    parser.add_argument(
        '--outputs_other_sources_gold_file', type=str, default="",
        help="gold file for other outputs, optional, needed if the outputs file doesn't have gold labels")
    parser.add_argument(
        '--outputs_other_sources_type', type=str, default="json", help='type of file of other outputs to evaluate')
    parser.add_argument(
        '--outputs_other_name', type=str, default="", help='identifier name of other outputs\' source')
    
    # for random evaluation
    parser.add_argument(
        '--random_tags_for_eval', type=str, default="no", help="should random tags be used for eval")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args
