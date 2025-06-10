import os
import numpy as np
import torch
from train import Trainer
import argparse
import setup
import shutil
import utils
import datasets
import models
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def launch_training_run(train_params):
    # setup:
    params = setup.get_default_params_train(train_params)
    params['save_path'] = os.path.join(params['save_base'], params['experiment_name'])
    if params['timestamp']:
        params['save_path'] = params['save_path'] + '_' + utils.get_time_stamp()
    try:
        os.makedirs(params['save_path'], exist_ok=False)
    except:
        shutil.rmtree(params['save_path'])
        os.makedirs(params['save_path'], exist_ok=False)
    if (params['transformer_input_enc'] == 'sin_cos_env') and (params['dataset'] == 'transformer'):
        params['species_dim'] = params['species_dim'] - 20

    # data:
    train_dataset = datasets.get_train_data(params)
    params['input_dim'] = train_dataset.input_dim
    params['num_classes'] = train_dataset.num_classes + (1 if params['loss'] == 'pdf_count' or params['loss'] == 'an_pdf' else 0)
    params['class_to_taxa'] = train_dataset.class_to_taxa
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=8,
        collate_fn=getattr(train_dataset, 'collate_fn', None))

    # model:
    model = models.get_model(params)
    model = model.to(params['device'])

    # train:
    trainer = Trainer(model, train_loader, params)

    for epoch in range(0, params['num_epochs']):
        print(f'epoch {epoch + 1}')
        trainer.train_one_epoch()

        # save the current model
        trainer.save_model()
        if epoch > 0 and epoch % params['save_frequency'] == 0:
            trainer.save_model(postfix=f'_{epoch}')

    return

def main(train_params):

    # Launch the training run with the converted train_params
    launch_training_run(train_params)

    print(f"Training finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument('--device', type=str, default='cuda:0',  # choices=['cuda', 'cpu']
                        help="Device to use for computation.")
    parser.add_argument('--hard_cap_num_per_class', type=int, default=10,
                        help="Maximum number of examples per class to use for training.")
    parser.add_argument('--input_enc', type=str, default='sin_cos', choices=['sin_cos', 'env', 'sin_cos_env'],
                        help="Type of inputs to use for training.")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs.")
    parser.add_argument('--experiment_name', type=str, default='test_fs',
                        help="Name of the experiment.")
    parser.add_argument('--obs_file', type=str, default='geo_prior_train.csv',
                        help="Path to the observations file.")
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--data_probs', type=str,
                        default='loc:0.1;text:0.1;image:0.1;loc+text:0.1;loc+image:0.1;text+image:0.1;loc+text+image:0.4')
    parser.add_argument('--tokens_per_type', type=str, default='loc:20;text:1;image:1')

    args = parser.parse_args()
    args = vars(args)

    # Infer use_xxx_inputs flags
    args['use_text_inputs'] = ('text' in args['data_probs']) or ('text' in args['tokens_per_type'])
    args['use_env_inputs'] = ('env' in args['data_probs']) or ('env' in args['tokens_per_type'])
    args['use_image_inputs'] = ('image' in args['data_probs']) or ('image' in args['tokens_per_type'])

    # Set random seed
    set_seed(args['seed'])

    # Call the main function
    main(args)