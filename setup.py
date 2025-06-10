import copy
import torch

def apply_overrides(params, overrides):
    params = copy.deepcopy(params)
    for param_name in overrides:
        params[param_name] = overrides[param_name]
    return params

def get_default_params_train(overrides=None):
    if overrides is None:
        overrides = {}
    params = {}

    # --- misc ---
    params['device'] = 'cuda'  # cuda, cpu
    params['save_base'] = './experiments/'  # where to save outputs
    params['save_frequency'] = 5  # epochs between saves
    params['experiment_name'] = 'test_fs'  # name of this run
    params['timestamp'] = False  # append timestamp to save dir

    # --- data settings ---
    params['species_set'] = 'all'  # all, snt_birds
    params['hard_cap_seed'] = 9472  # random seed for hard cap
    params['hard_cap_num_per_class'] = 1000  # max examples per class
    params['aux_species_seed'] = 8099  # seed for aux species sampling
    params['num_aux_species'] = 0  # number of auxiliary species
    params['input_time'] = False  # include time as feature
    params['input_time_dim'] = 0  # dimension for time encoding
    params['dataset'] = 'flexible'  # dataset name
    params['zero_shot'] = True  # zero-shot inference
    params['subset_cap_name'] = None  # cap on subset by name
    params['subset_cap_num_per_class'] = -1  # cap on subset size
    params['seed'] = 1000  # global random seed
    params['add_location_noise'] = False  # jitter location inputs
    params['variable_context_length'] = False  # vary context length at train
    params['eval_dataset'] = 'eval_flexible'  # evaluation dataset
    params['use_text_inputs'] = True  # include text tokens
    params['use_image_inputs'] = True  # include image tokens
    params['class_token_transformation'] = 'identity'  # how to transform class token

    # --- data files ---
    params['obs_file'] = 'geo_prior_train.csv'  # observations CSV
    params['taxa_file'] = 'geo_prior_train_meta.json'  # metadata JSON

    # --- model architecture ---
    params['model'] = 'MultiInputModel_2'
    params['num_filts'] = 256  # embedding dimension
    params['input_enc'] = 'sin_cos'  # coordinate encoder
    params['input_dim'] = 4  # input dimension
    params['depth'] = 4  # network depth
    params['noise_time'] = False  # add noise to time

    # species encoding
    params['species_dim'] = 256  # species embedding dim
    params['species_enc_depth'] = 4  # depth of species encoder
    params['species_filts'] = 512  # filters in species encoder

    # --- transformer settings ---
    params['transformer_input_enc'] = 'sinr'  # positional encoding type
    params['transformer_dropout'] = 0.1  # dropout in transformer
    params['num_heads'] = 2  # attention heads
    params['ema_factor'] = 1.0  # EMA for weights
    params['use_register'] = True  # register intermediate states

    # --- pretrained SINR ---
    params['use_pretrained_sinr'] = True
    params['pretrained_loc'] = (
            '/disk/scratch_fast/chris_2/sinr/experiments/' +
            'baseline_sinr_sin_cos_1000_cap_20_epochs_no_eval_species/model.pt'
    )
    params['use_pretrained_env_sinr'] = True
    params['pretrained_env_loc'] = (
            '/disk/scratch_fast/chris_2/sinr/experiments/' +
            'baseline_sinr_sin_cos_1000_cap_20_epochs_no_eval_species/model.pt'
    )
    params['use_pretrained_image_sinr'] = False
    params['pretrained_image_loc'] = (
            '/disk/scratch_fast/chris_2/sinr/experiments/' +
            'flex_1-5_image_only/model.pt'
    )
    params['freeze_sinr'] = False  # freeze SINR encoder

    # --- data sampling and dropout ---
    params['data_probs'] = (
        'loc:0.1;text:0.1;image:0.1;'
        'loc+text:0.1;loc+image:0.1;'
        'text+image:0.1;loc+text+image:0.4'
    )
    params['tokens_per_type'] = 'loc:20;text:1;image:1'
    params['primary_env'] = False

    # --- loss ---
    params['loss'] = 'an_full_given_classes'
    params['pos_weight'] = 2048  # positive class weight

    # --- optimization ---
    params['batch_size'] = 64
    params['lr'] = 0.0005
    params['lr_decay'] = 0.98
    params['num_epochs'] = 20

    # --- logging & saving ---
    params['log_frequency'] = 512  # steps between logs

    # apply any user overrides
    params = apply_overrides(params, overrides)
    return params


def get_default_params_eval(overrides={}):

    params = {}

    '''
    misc
    '''
    params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['seed'] = 2022
    params['exp_base'] = './experiments'
    params['ckp_name'] = 'model.pt'
    params['eval_type'] = 'snt' # snt, iucn
    params['experiment_name'] = 'fs_test'
    params['input_dim'] = 4
    params['input_time'] = False
    params['input_time_dim'] = 0
    params['num_samples'] = 0
    params['text_section'] = ''
    params['extract_pos'] = False
    params['batch_size'] = 64

    params = apply_overrides(params, overrides)

    return params
