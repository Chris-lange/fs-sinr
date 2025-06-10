import torch
import utils


def get_loss_function(params):
    return an_full_given_classes_flexible

def neg_log(x):
    return -torch.log(x + 1e-5)

def an_full_given_classes_flexible(batch, model, params, loc_to_feats, neg_type='hard'):
    loc_feat, _, class_id, context_feats, context_mask, text_emb, text_mask, image_emb, image_mask, env_emb, env_mask = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    context_feats = context_feats.to(params['device'])
    context_mask = context_mask.to(params['device'])
    text_emb = text_emb.to(params['device'])
    text_mask = text_mask.to(params['device'])
    image_emb = image_emb.to(params['device'])
    image_mask = image_mask.to(params['device'])
    env_emb = env_emb.to(params['device'])
    env_mask = env_mask.to(params['device'])
    # if env_emb is not None:
    #     env_emb = env_emb.to(params['device'])

    batch_size = loc_feat.shape[0]
    inds = torch.arange(batch_size).to(params['device'])

    # Create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    if params['input_time_dim'] > 0:
        rand_feat = torch.cat([loc_to_feats(rand_loc, normalize=False),
                               loc_feat[:, -(params['input_time_dim'] + (1 if params['noise_time'] else 0)):]], dim=1)
    else:
        rand_feat = loc_to_feats(rand_loc, normalize=False)

    # Get embeddings and probabilities for actual and random locations
    loc_prob = model(loc_feat, context_feats, context_mask, class_id, text_emb=text_emb, image_emb=image_emb, env_emb=env_emb)
    rand_prob = model(rand_feat, context_feats, context_mask, class_id, text_emb=text_emb, image_emb=image_emb, env_emb=env_emb)

    # Create a mask for intra-class comparison
    same_class_mask = torch.eq(class_id[:, None], class_id[None, :])

    # Modify mask: keep self-comparisons unmasked
    same_class_mask.fill_diagonal_(0)

    # Data loss
    loss_pos = neg_log(1.0 - loc_prob)  # assume negative
    # next line hopefully prevents loss from correctly predicting positives in the right place for the current class
    loss_pos = loss_pos * (~same_class_mask).float()  # Only keep losses where class IDs differ
    loss_bg = neg_log(1.0 - rand_prob)  # assume negative

    loss_pos[inds, inds] = params['pos_weight'] * neg_log(loc_prob[inds, inds])

    # Total loss
    loss = loss_pos.mean() + loss_bg.mean()

    return loss