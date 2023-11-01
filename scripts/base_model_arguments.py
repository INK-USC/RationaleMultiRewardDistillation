import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='training base I2RO model')

    # dataset
    parser.add_argument(
        '--output-dir', type=str, default='../save')
    parser.add_argument(
        '--save_number', type=str, default='1')
    parser.add_argument(
        '--gen-mode', type=str, default='i2ro', help='name of gen_mode out of i2o, i2ro')
    parser.add_argument(
        '--use_demonstrations', type=int, default=0, help='0/1 whether to use demonstrations')
    parser.add_argument(
        '--dataset-name', type=str, default='strategyqa', help='name of dataset')
    parser.add_argument(
        '--dataset-train', type=str, default='',
        help='JSONL file containing train data. Each row must contain a prompt at `row["question"]`.')
    parser.add_argument(
        '--dataset-val', type=str, default='',
        help='JSONL file containing val data. Each row must contain a prompt at `row["question"]`.')
    parser.add_argument(
        '--dataset-test', type=str, default='',
        help='JSONL file containing test data. Each row must contain a prompt at `row["question"]`.')

    # training
    parser.add_argument(
        '--model-type', type=str, default='t5-large', help='model architecture to train')
    parser.add_argument(
        '--num-epochs', type=int, default=25, help='total number of epochs')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument(
        '--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument(
        '--if_warmup', type=int, default=1, help='0/1 whether to warmup')
    parser.add_argument(
        '--num_warmup_steps', type=int, default=1000, help='number of warmup steps in lr scheduler')
    parser.add_argument(
        '--clip_grad', action='store_true', default=True, help='whether to clip gradient')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5, help='maximum norm of gradients ')
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=1, help="gradient accumulation step.")

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
    parser.add_argument(
            '--temperature', type=float, default=1.0, help='temperature for sampling policy.')

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

    # only for evaluation
    parser.add_argument(
        '--save_dir_for_eval', type=str, default="", help='path to save dir')
    parser.add_argument(
        '--ckp_num_for_eval', type=str, default="all", help='checkpoints to evaluate')
    parser.add_argument(
        '--reward_names_other', type=str, default="", help='other rewards to evaluate')


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args
