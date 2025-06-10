import collections
import sys

import numpy as np
import random
import torch
import time
import os
import json
import setup
import utils
import models
import datasets
import torch.nn as nn
from tqdm import tqdm

class EvaluatorSNT:
    def __init__(self, train_params, eval_params):
        self.train_params = train_params
        self.eval_params = eval_params
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        D = np.load(os.path.join(paths['snt'], 'snt_res_5.npy'), allow_pickle=True)
        D = D.item()
        self.loc_indices_per_species = D['loc_indices_per_species']
        self.labels_per_species = D['labels_per_species']
        self.taxa = D['taxa']
        self.obs_locs = D['obs_locs']
        self.obs_locs_idx = D['obs_locs_idx']
        self.pos_eval_data_loc = os.path.join(paths['data'], 'positive_eval_data.npz')
        self.background_eval_data_loc = os.path.join(paths['data'], '10000_background_negs.npz')

    @torch.no_grad()
    def run_evaluation(self, model, enc, extra_input=None):
        results = {}

        # set seeds:
        np.random.seed(self.eval_params['seed'])
        random.seed(self.eval_params['seed'])

        # evaluate the geo model for each taxon
        results['per_species_average_precision_all'] = np.zeros((len(self.taxa)), dtype=np.float32)

        # get eval locations and apply input encoding
        obs_locs = torch.from_numpy(self.obs_locs).to(self.eval_params['device'])
        loc_feat = torch.cat([enc.encode(obs_locs), extra_input.expand(obs_locs.shape[0], -1)], dim=1) if extra_input is not None else enc.encode(obs_locs)

        # get classes to eval
        classes_of_interest = torch.zeros(len(self.taxa), dtype=torch.int64)
        for tt_id, tt in enumerate(self.taxa):
            class_of_interest = np.where(np.array(self.train_params['class_to_taxa']) == tt)[0]
            if len(class_of_interest) != 0:
                classes_of_interest[tt_id] = torch.from_numpy(class_of_interest)

        with torch.no_grad():
            dummy_context_mask = None
            dummy_context_sequence = None

            # generate model predictions for classes of interest at eval locations
            loc_emb = model(x=loc_feat, context_sequence=dummy_context_sequence, context_mask=dummy_context_mask,
                            class_ids=classes_of_interest, return_feats=True)

            classes_of_interest = classes_of_interest.to(self.eval_params["device"])

            wt = model.get_eval_embeddings(classes_of_interest)

            pred_mtx = torch.matmul(loc_emb, torch.transpose(wt, 0, 1))

        split_rng = np.random.default_rng(self.eval_params['split_seed'])
        for tt_id, tt in tqdm(enumerate(self.taxa)):
            class_of_interest = np.where(np.array(self.train_params['class_to_taxa']) == tt)[0]
            if len(class_of_interest) == 0 and not (self.train_params['zero_shot'] or self.eval_params['num_samples'] > 0):
                # taxa of interest is not in the model
                results['per_species_average_precision_all'][tt_id] = np.nan
            else:
                # generate ground truth labels for current taxa
                cur_loc_indices = np.array(self.loc_indices_per_species[tt_id])
                cur_labels = np.array(self.labels_per_species[tt_id])
                # apply per-species split:
                assert self.eval_params['split'] in ['all', 'val', 'test']
                if self.eval_params['split'] != 'all':
                    num_val = np.floor(len(cur_labels) * self.eval_params['val_frac']).astype(int)
                    idx_rand = split_rng.permutation(len(cur_labels))
                    if self.eval_params['split'] == 'val':
                        idx_sel = idx_rand[:num_val]
                    elif self.eval_params['split'] == 'test':
                        idx_sel = idx_rand[num_val:]
                    cur_loc_indices = cur_loc_indices[idx_sel]
                    cur_labels = cur_labels[idx_sel]
                cur_labels = (torch.from_numpy(cur_labels).to(self.eval_params['device']) > 0).float()

                with torch.no_grad():
                    logits = pred_mtx[:, tt_id]
                    preds = torch.sigmoid(logits)

                    results['per_species_average_precision_all'][tt_id] = utils.average_precision_score_fasterer(
                        cur_labels,
                        preds[cur_loc_indices]).item()
                continue

        valid_taxa = ~np.isnan(results['per_species_average_precision_all'])

        # store results
        per_species_average_precision_valid = results['per_species_average_precision_all'][valid_taxa]
        results['mean_average_precision'] = per_species_average_precision_valid.mean()
        results['num_eval_species_w_valid_ap'] = valid_taxa.sum()
        results['num_eval_species_total'] = len(self.taxa)

        return results

    def report(self, results):
        for field in ['mean_average_precision', 'num_eval_species_w_valid_ap', 'num_eval_species_total']:
            print(f'{field}: {results[field]}')

class EvaluatorIUCN:

    def __init__(self, train_params, eval_params):
        self.train_params = train_params
        self.eval_params = eval_params
        with open('paths.json', 'r') as f:
            paths = json.load(f)
        with open(os.path.join(paths['iucn'], 'iucn_res_5.json'), 'r') as f:
            self.data = json.load(f)
        self.obs_locs = np.array(self.data['locs'], dtype=np.float32)
        self.taxa = [int(tt) for tt in self.data['taxa_presence'].keys()]
        self.pos_eval_data_loc = os.path.join(paths['data'], 'positive_eval_data.npz')
        self.background_eval_data_loc = os.path.join(paths['data'], '10000_background_negs.npz')

    @torch.no_grad()
    def run_evaluation(self, model, enc, extra_input=None):
        results = {}
        results['per_species_average_precision_all'] = np.zeros(len(self.taxa), dtype=np.float32)
        # get eval locations and apply input encoding
        obs_locs = torch.from_numpy(self.obs_locs).to(self.eval_params['device'])
        loc_feat = torch.cat([enc.encode(obs_locs), extra_input.expand(obs_locs.shape[0], -1)], dim=1) if extra_input is not None else enc.encode(obs_locs)

        # get classes to eval
        # classes_of_interest = torch.zeros(len(self.taxa), dtype=torch.int64)
        classes_of_interest = np.zeros(len(self.taxa))
        array_class_to_taxa = np.array(self.train_params['class_to_taxa'])
        for tt_id, tt in enumerate(self.taxa):
            class_of_interest = np.where(array_class_to_taxa == tt)[0]
            if len(class_of_interest) != 0:
                classes_of_interest[tt_id] = class_of_interest
        classes_of_interest = torch.from_numpy(classes_of_interest).to(dtype=torch.long, device=self.eval_params['device'])

        with torch.no_grad():
            dummy_context_mask = None
            dummy_context_sequence = None
            # generate model predictions for classes of interest at eval locations
            loc_emb = model(x=loc_feat, context_sequence=dummy_context_sequence, context_mask=dummy_context_mask,
                            class_ids=classes_of_interest, return_feats=True)
            wt = model.get_eval_embeddings(classes_of_interest)
            print("Creating IUCN prediction matrix")
            pred_mtx = torch.matmul(loc_emb, torch.transpose(wt, 0, 1))

        for tt_id, tt in tqdm(enumerate(self.taxa)):
            class_of_interest = np.where(array_class_to_taxa == tt)[0]
            if len(class_of_interest) == 0 and not (self.train_params['zero_shot'] or self.eval_params['num_samples'] > 0):
                # taxa of interest is not in the model
                results['per_species_average_precision_all'][tt_id] = np.nan
            else:
                gt = torch.zeros(obs_locs.shape[0], dtype=torch.float32, device=self.eval_params['device'])
                gt[self.data['taxa_presence'][str(tt)]] = 1.0
                with torch.no_grad():
                    logits = pred_mtx[:, tt_id]
                    preds = torch.sigmoid(logits)
                    results['per_species_average_precision_all'][tt_id] = utils.average_precision_score_fasterer(gt, preds).item()
                    continue

                gt = torch.zeros(obs_locs.shape[0], dtype=torch.float32, device=self.eval_params['device'])
                gt[self.data['taxa_presence'][str(tt)]] = 1.0
                # average precision score:
                results['per_species_average_precision_all'][tt_id] = utils.average_precision_score_fasterer(gt, pred).item()

        valid_taxa = ~np.isnan(results['per_species_average_precision_all'])

        # store results
        per_species_average_precision_valid = results['per_species_average_precision_all'][valid_taxa]
        results['mean_average_precision'] = per_species_average_precision_valid.mean()
        results['num_eval_species_w_valid_ap'] = valid_taxa.sum()
        results['num_eval_species_total'] = len(self.taxa)
        return results

    def report(self, results):
        for field in ['mean_average_precision', 'num_eval_species_w_valid_ap', 'num_eval_species_total']:
            print(f'{field}: {results[field]}')


def launch_eval_run(overrides, train_overrides=None):

    eval_params = setup.get_default_params_eval(overrides)

    # set up model:
    eval_params['model_path'] = os.path.join(eval_params['exp_base'], eval_params['experiment_name'], eval_params['ckp_name'])
    train_params = torch.load(eval_params['model_path'], map_location='cpu', weights_only=False)
    default_params = setup.get_default_params_train()
    for key in default_params:
        if key not in train_params['params']:
            train_params['params'][key] = default_params[key]
    if train_overrides != None:
        for key, value in train_overrides.items():
            print(f'updating train param {key}')
            train_params['params'][key] = value

    model = models.get_model(train_params['params'], inference_only=True)
    model.load_state_dict(train_params['state_dict'], strict=False)
    model = model.to(eval_params['device'])
    model.eval()

    # create input encoder:
    if "env" in train_params['params']['input_enc']:
        raster = datasets.load_env().to(eval_params['device'])
    else:
        raster = None
    enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster, input_dim=train_params['params']['input_dim'])
    if train_params['params']['input_time']:
        time_enc = utils.TimeEncoder(input_enc='conical') if train_params['params']['input_time'] else None
        extra_input = torch.cat([time_enc.encode(torch.tensor([[0.0, 1.0]]))], dim=1).to(eval_params['device'])
    else:
        extra_input = None

    train_dataset = datasets.get_train_data(train_params['params'])

    if eval_params['text_section'] != '':
        train_dataset.select_text_section(eval_params['text_section'])
        print(f'Using {eval_params["text_section"]} text for evaluation')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_params['params']['batch_size'],
        shuffle=True,
        num_workers=8,
        collate_fn=getattr(train_dataset, 'collate_fn', None))

    # if len(train_params['params']['class_to_taxa']) != train_dataset.class_to_taxa:

    # Create new embedding layers for the expanded classes
    num_new_classes = len(train_dataset.class_to_taxa)
    embedding_dim = model.ema_embeddings.embedding_dim
    new_ema_embeddings = nn.Embedding(num_embeddings=num_new_classes, embedding_dim=embedding_dim).to(eval_params["device"])
    new_eval_embeddings = nn.Embedding(num_embeddings=num_new_classes, embedding_dim=embedding_dim).to(eval_params["device"])
    nn.init.xavier_uniform_(new_ema_embeddings.weight)
    nn.init.xavier_uniform_(new_eval_embeddings.weight)

    # Convert lists to numpy arrays for indexing
    class_to_taxa_np = np.array(train_params['params']['class_to_taxa'])
    class_to_taxa_expanded_np = np.array(train_dataset.class_to_taxa)

    # Find common taxa and their indices
    common_taxa, original_indices, expanded_indices = np.intersect1d(
        class_to_taxa_np, class_to_taxa_expanded_np, return_indices=True)

    # Update new embeddings for the common taxa
    new_ema_embeddings.weight.data[expanded_indices] = model.ema_embeddings.weight.data[original_indices]
    new_eval_embeddings.weight.data[expanded_indices] = model.eval_embeddings.weight.data[original_indices]

    # Replace old embeddings with new embeddings
    model.ema_embeddings = new_ema_embeddings
    model.eval_embeddings = new_eval_embeddings

    # Print to verify
    print("Updating EMA Embeddings: ", model.ema_embeddings.weight.size())
    print("Updating Eval Embeddings: ", model.eval_embeddings.weight.size())

    train_params['params']['class_to_taxa'] = train_dataset.class_to_taxa

    for _, batch in tqdm(enumerate(train_loader)):
        loc_feat, _, class_id, context_feats, context_mask, text_emb, text_mask, image_emb, image_mask, env_emb, env_mask = batch
        loc_feat = loc_feat.to(eval_params['device'])
        class_id = class_id.to(eval_params['device'])
        context_feats = context_feats.to(eval_params['device'])
        context_mask = context_mask.to(eval_params['device'])
        text_emb = text_emb.to(eval_params['device'])
        text_mask = text_mask.to(eval_params['device'])
        image_emb = image_emb.to(eval_params['device'])
        image_mask = image_mask.to(eval_params['device'])
        env_emb = env_emb.to(eval_params['device'])
        env_mask = env_mask.to(eval_params['device'])
        probs = model.forward(loc_feat, context_feats, context_mask, class_id, text_emb=text_emb, image_emb=image_emb,
                         env_emb=env_emb, return_feats=False, return_class_embeddings=False, class_of_interest=None, use_eval_embeddings=True)
    print('eval embeddings generated!')

    print('\n' + eval_params['eval_type'])
    t = time.time()
    if eval_params['eval_type'] == 'snt':
        eval_params['split'] = 'test' # val, test, all
        eval_params['val_frac'] = 0.50
        eval_params['split_seed'] = 7499
        evaluator = EvaluatorSNT(train_params['params'], eval_params)
        results = evaluator.run_evaluation(model, enc, extra_input=extra_input)
        evaluator.report(results)
    elif eval_params['eval_type'] == 'iucn':
        evaluator = EvaluatorIUCN(train_params['params'], eval_params)
        results = evaluator.run_evaluation(model, enc, extra_input=extra_input)
        evaluator.report(results)
    else:
        raise NotImplementedError('Eval type not implemented.')
    print(f'evaluation completed in {np.around((time.time()-t)/60, 1)} min')
    return results

def generate_eval_embeddings(overrides, taxa_of_interest, num_context, train_overrides=None):

    eval_params = setup.get_default_params_eval(overrides)
    
    # set up model:
    eval_params['model_path'] = os.path.join(eval_params['exp_base'], eval_params['experiment_name'], eval_params['ckp_name'])
    eval_params['device'] = 'cpu'
    train_params = torch.load(eval_params['model_path'], map_location='cpu')
    train_params['params']['device'] = 'cpu'
    default_params = setup.get_default_params_train()
    for key in default_params:
        if key not in train_params['params']:
            train_params['params'][key] = default_params[key]

    # create input encoder:
    if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
        raster = datasets.load_env().to(eval_params['device'])
    else:
        raster = None
    enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster, input_dim=train_params['params']['input_dim'])
    if train_params['params']['input_time']:
        time_enc = utils.TimeEncoder(input_enc='conical') if train_params['params']['input_time'] else None
        extra_input = torch.cat([time_enc.encode(torch.tensor([[0.0, 1.0]]))], dim=1).to(eval_params['device'])
    else:
        extra_input = None

    if train_overrides != None:
        for key, value in train_overrides.items():
            print(f'updating train param {key}')
            train_params['params'][key] = value

    train_dataset = datasets.get_train_data(train_params['params'])

    model = models.get_model(train_params['params'], inference_only=True)
    model.load_state_dict(train_params['state_dict'], strict=False)
    model = model.to(eval_params['device'])
    model.eval()

    # Create new embedding layers for the expanded classes
    num_new_classes = len(train_dataset.class_to_taxa)
    embedding_dim = model.ema_embeddings.embedding_dim
    new_ema_embeddings = nn.Embedding(num_embeddings=num_new_classes, embedding_dim=embedding_dim).to(eval_params["device"])
    new_eval_embeddings = nn.Embedding(num_embeddings=num_new_classes, embedding_dim=embedding_dim).to(eval_params["device"])
    nn.init.xavier_uniform_(new_ema_embeddings.weight)
    nn.init.xavier_uniform_(new_eval_embeddings.weight)

    # Convert lists to numpy arrays for indexing
    class_to_taxa_np = np.array(train_params['params']['class_to_taxa'])
    class_to_taxa_expanded_np = np.array(train_dataset.class_to_taxa)

    # Find common taxa and their indices
    common_taxa, original_indices, expanded_indices = np.intersect1d(
        class_to_taxa_np, class_to_taxa_expanded_np, return_indices=True)

    # Update new embeddings for the common taxa
    new_ema_embeddings.weight.data[expanded_indices] = model.ema_embeddings.weight.data[original_indices]
    new_eval_embeddings.weight.data[expanded_indices] = model.eval_embeddings.weight.data[original_indices]

    # Replace old embeddings with new embeddings
    model.ema_embeddings = new_ema_embeddings
    model.eval_embeddings = new_eval_embeddings

    # Print to verify
    print("Updated EMA Embeddings: ", model.ema_embeddings.weight.size())
    print("Updated Eval Embeddings: ", model.eval_embeddings.weight.size())

    train_params['params']['class_to_taxa'] = train_dataset.class_to_taxa

    class_of_interest = train_dataset.class_to_taxa.index(taxa_of_interest)

    # Find the index of class_of_interest in the labels tensor
    loc_index_of_interest = (train_dataset.labels == class_of_interest).nonzero(as_tuple=True)[0].item()

    loc_of_interest = train_dataset.loc_feats[loc_index_of_interest]

    all_class_context_feats = train_dataset.per_class_loc_feats[class_of_interest]
    all_class_context_locs = train_dataset.per_class_locs[class_of_interest]

    context_feats_of_interest = all_class_context_feats[:num_context,:]
    context_locs_of_interest = all_class_context_locs[:num_context,:]

    context_mask = (context_locs_of_interest == -10).all(dim=-1).to(eval_params['device']).unsqueeze(0)

    probs = model.forward(
        x=loc_of_interest.to(train_params['params']['device']),
        context_sequence=context_feats_of_interest.to(train_params['params']['device']),
        context_mask=context_mask,
        class_ids=class_of_interest,
        return_feats=False,
        return_class_embeddings=False,
        class_of_interest=None,
        use_eval_embeddings=True
    )

    print(f'eval embedding generated for class {class_of_interest}, taxa {taxa_of_interest}')

    return model, context_locs_of_interest, train_params, class_of_interest

def generate_eval_embedding_from_given_points(context_points, overrides, taxa_of_interest, train_overrides=None, text_emb=None, image_emb=None):

    eval_params = setup.get_default_params_eval(overrides)

    # set up model:
    eval_params['model_path'] = os.path.join(eval_params['exp_base'], eval_params['experiment_name'], eval_params['ckp_name'])
    train_params = torch.load(eval_params['model_path'], map_location='cpu')
    default_params = setup.get_default_params_train()
    for key in default_params:
        if key not in train_params['params']:
            train_params['params'][key] = default_params[key]

    # create input encoder:
    if train_params['params']['input_enc'] in ['env', 'sin_cos_env']:
        raster = datasets.load_env().to(eval_params['device'])
    else:
        raster = None
    enc = utils.CoordEncoder(train_params['params']['input_enc'], raster=raster, input_dim=train_params['params']['input_dim'])
    if train_params['params']['input_time']:
        time_enc = utils.TimeEncoder(input_enc='conical') if train_params['params']['input_time'] else None
        extra_input = torch.cat([time_enc.encode(torch.tensor([[0.0, 1.0]]))], dim=1).to(eval_params['device'])
    else:
        extra_input = None

    if train_overrides != None:
        for key, value in train_overrides.items():
            print(f'updating train param {key}')
            train_params['params'][key] = value

    # create context point encoder
    transformer_input_enc = train_params['params']['transformer_input_enc']
    if transformer_input_enc in ['env', 'sin_cos_env']:
        transformer_raster = datasets.load_env().to(eval_params['device'])
    else:
        transformer_raster = None
    token_dim = train_params['params']['species_dim']

    if transformer_input_enc == 'sinr':
        transformer_enc = enc
    else:
        transformer_enc = utils.CoordEncoder(transformer_input_enc, transformer_raster, input_dim=token_dim)

    # load model
    model = models.get_model(train_params['params'], inference_only=True)
    model.load_state_dict(train_params['state_dict'], strict=False)
    model = model.to(eval_params['device'])
    model.eval()

    # # Create new embedding layers for the expanded classes
    embedding_dim = model.ema_embeddings.embedding_dim
    new_eval_embeddings = nn.Embedding(num_embeddings=model.eval_embeddings.weight.size()[0], embedding_dim=embedding_dim).to(eval_params["device"])

    # Update new embeddings for the common taxa
    new_eval_embeddings.weight.data = model.eval_embeddings.weight.data

    # Replace old embeddings with new embeddings
    model.eval_embeddings = new_eval_embeddings

    class_of_interest = 0

    just_loc = torch.from_numpy(np.array([[0.0,0.0]]).astype(np.float32))

    loc_of_interest = enc.encode(just_loc, normalize=False)

    context_points = torch.from_numpy(np.array(context_points).astype(np.float32))

    all_class_context_feats = transformer_enc.encode(context_points, normalize=False)
    all_class_context_locs = context_points

    context_feats_of_interest = all_class_context_feats
    context_locs_of_interest = all_class_context_locs

    context_mask = torch.from_numpy(np.full((1, context_feats_of_interest.shape[0]), False))

    probs = model.forward(x=loc_of_interest.to(eval_params['device']),
                          context_sequence=context_feats_of_interest.to(eval_params['device']),
                          context_mask=context_mask,
                          class_ids=class_of_interest,
                          text_emb=text_emb,
                          image_emb=image_emb,
                          env_emb=None,
                          return_feats=False,
                          return_class_embeddings=False,
                          class_of_interest=None,
                          use_eval_embeddings=True)

    print(f'eval embedding generated for class {class_of_interest}')

    return model, context_locs_of_interest, train_params, class_of_interest
