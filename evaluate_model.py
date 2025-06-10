import os
import numpy as np
import torch
import eval
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script parameters")

    # device
    parser.add_argument('--device', type=str, default='cuda:0')

    # Define evaluation parameters
    parser.add_argument('--experiment_name', type=str, default='test_fs',
                        help='Name of the experiment')
    parser.add_argument('--exp_base', type=str, default='./experiments',
                        help='Name of directory with experiments')
    parser.add_argument('--text_section', type=str, default='range', help='section of text to use for eval -'
                                                                          'range, habitat, species_description, overview_summary,'
                                                                          'or blank string for uniform random selection')
    # Define train overrides
    parser.add_argument('--dataset', type=str, default='eval_flexible')
    parser.add_argument('--ckp_name', type=str, default='model.pt')
    parser.add_argument('--data_probs', type=str, default='loc+text+image:1.0')
    parser.add_argument('--tokens_per_type', type=str, default='loc:50;text:1;image:1')
    return parser.parse_args()

def main(args):

    # Setting up evaluation parameters
    eval_params = {
        'device': args['device'],
        'experiment_name': args['experiment_name'],
        'exp_base': args['exp_base'],
        'ckp_name': args['ckp_name'],
        'text_section': args['text_section']
    }

    train_overrides = {
        'dataset': args['dataset'],
        'data_probs': args['data_probs'],
        'tokens_per_type': args['tokens_per_type'],
    }

    for eval_type in ['snt', 'iucn']:
        eval_params['eval_type'] = eval_type
        if 'text' in args['data_probs']:
            save_name = os.path.join(eval_params['exp_base'], eval_params['experiment_name'], f'results_{eval_params["ckp_name"]}_{eval_type}_token_amounts_{train_overrides["tokens_per_type"]}_token_probs_{train_overrides["data_probs"]}_{eval_params["text_section"]}_text.npy')
        else:
            save_name = os.path.join(eval_params['exp_base'], eval_params['experiment_name'], f'results_{eval_params["ckp_name"]}_{eval_type}_token_amounts_{train_overrides["tokens_per_type"]}_token_probs_{train_overrides["data_probs"]}.npy')
        if eval_type == 'iucn' and False:
            eval_params['device'] = torch.device('cpu')  # if needed for memory reasons
        cur_results = eval.launch_eval_run(eval_params, train_overrides)
        np.save(save_name, cur_results)

if __name__ == '__main__':
    args = parse_args()
    main(vars(args))