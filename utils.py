import torch
import torch.utils.data
import torch.nn.functional as F
import os
import numpy as np


def norm_batch(x):
    bs = x.shape[0]
    Isize = x.shape[-1]
    min_value = x.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    max_value = x.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x


def Dice_loss(y_true, y_pred, smooth=1):
    alpha = 0.5
    beta = 0.5
    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3))
    fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky_class), tversky_class


def get_dice_ji(predict, target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))


def get_input_dict(imgs, original_sz, img_sz):
    batched_input = []
    for i, img in enumerate(imgs):
        input_size = tuple([int(x) for x in img_sz[i].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[i].squeeze().tolist()])
        singel_input = {
            'image': img,
            'original_size': original_size,
            'image_size': input_size,
            'point_coords': None,
            'point_labels': None,
        }
        batched_input.append(singel_input)
    return batched_input



def get_sam_model_output(sam, batched_input, owlv2_mask, it=1):
    input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)
    image_embeddings = sam.image_encoder(input_images)
    sam_mask = owlv2_mask.clone()
    sparse_embeddings, dense_embeddings = sam.prompt_encoder(points=None, boxes=None, masks=None)
    sam_mask, score = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    sam_mask = (sam_mask - sam_mask.min()) / (sam_mask.max() - sam_mask.min())
    return sam_mask, score

def get_points(gt, Npos, Nneg):
    gt_small = F.interpolate(gt, (64,64), mode='nearest').squeeze().reshape(-1)
    pos_index = torch.where(gt_small == 1)
    shuffle_inx_pos =  torch.randperm(len(pos_index[0]))
    neg_index = torch.where(gt_small == 0)
    shuffle_inx_neg = torch.randperm(len(neg_index[0]))
    return pos_index[0][shuffle_inx_pos[:Npos]], neg_index[0][shuffle_inx_neg[:Nneg]]

def get_similarity_maps(sam, imgs, gts, th, pos, neg):
    image_embeds = sam.image_encoder(imgs.cuda())
    image_embeds = image_embeds.squeeze().reshape(256, -1).permute(1,0)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    sim = image_embeds @ image_embeds.T
    p_i, n_i = get_points(gts, pos, neg)
    pred = sim.reshape(-1, 64, 64).unsqueeze(dim=1)
    if neg>0:
        pred_pos = sim[p_i].sum(dim=0).reshape(64, 64).unsqueeze(dim=2)
        pred_neg = sim[n_i].sum(dim=0).reshape(64, 64).unsqueeze(dim=2)
        tmp = torch.cat((pred_neg, pred_pos), dim=2)
        pred = torch.argmax(tmp, dim=2).float()
    else:
        pred = sim[p_i].sum(dim=0).reshape(64, 64)
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        pred[pred>th] = 1
        pred[pred<=th] = 0
    return pred, p_i, n_i

def zero_sam_with_masks(sam, masks, batched_input):
    masks = masks.unsqueeze(dim=0).unsqueeze(dim=0)
    masks = F.interpolate(masks, (256,256), mode='nearest')
    input_images = torch.stack([sam.preprocess(x["image"].cuda()) for x in batched_input], dim=0)
    image_embeddings = sam.image_encoder(input_images)
    sparse_embeddings, dense_embeddings = sam.prompt_encoder(points=None, boxes=None, masks=masks)
    pred, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    pred[pred>0.5] = 1
    pred[pred<=0.5] = 0
    return pred.squeeze()

def zero_sam_with_points(sam, batched_input, p_i, n_i):
    pos = p_i.shape[0]
    neg = n_i.shape[0]
    pos_labels = torch.ones(pos)
    neg_labels = torch.zeros(neg)
    pos_points = []
    neg_points = []
    for i in range(pos):
        pos_points.append([p_i[i]%64 ,p_i[i]//64])
    for i in range(neg):
        neg_points.append([n_i[i]%64 ,n_i[i]//64])
    pos_points = torch.tensor(pos_points) * 16
    neg_points = torch.tensor(neg_points) * 16
    labels = torch.cat((neg_labels,pos_labels)).cuda().unsqueeze(dim=1)
    points = torch.cat((neg_points,pos_points)).cuda().unsqueeze(dim=1)
    input_images = torch.stack([sam.preprocess(x["image"].cuda()) for x in batched_input], dim=0)
    image_embeddings = sam.image_encoder(input_images)
    sparse_embeddings, dense_embeddings = sam.prompt_encoder(points=(points,labels), boxes=None, masks=None)
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    if neg>0:
        bs_half = low_res_masks.shape[0] // 2
        out = torch.cat((low_res_masks[bs_half:].sum(dim=0), low_res_masks[:bs_half].sum(dim=0)), dim=0)
        pred = torch.argmax(out, dim=0).float()
    else:
        pred = low_res_masks.sum(dim=0).squeeze()
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0
    return pred


