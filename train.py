import torch.optim as optim
import torch.utils.data
import torch
from tqdm import tqdm, trange
import os
import numpy as np
from dataset.glas import get_glas_dataset
from dataset.MoNuBrain import get_monu_dataset
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
from utils import *

def get_sam_model_output(sam, batched_input):
    input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)
    image_embeddings = sam.image_encoder(input_images)
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


def preprocess_logits(logits, size=256, th=0.01, is_resize=True):
    logits = F.sigmoid(logits)
    logits = (logits - logits.min()) / (logits.max() - logits.min())
    if is_resize:
        logits =  F.interpolate(logits, (size, size), mode='bilinear', align_corners=True)
    logits[logits>th] = 1
    logits[logits<=th] = 0
    return logits


def loss_step(optimizer, J, pred, criterion):
    dice_loss, _ = Dice_loss(pred, J)
    bce_loss = criterion(pred, J)
    loss = args['w0']*dice_loss + bce_loss
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def get_inputs(imgs, gts, sam, original_sz, img_sz):
    '''
    TODO: 
    1. self/cross options
    2. predefined points
    '''
    bs = imgs.shape[0]
    batched_input = get_input_dict(imgs.cuda(), original_sz, img_sz)
    gts = F.interpolate(gts.cuda().unsqueeze(dim=1), (256, 256), mode='nearest')
    # gts_large = sam.postprocess_masks(gts.unsqueeze(dim=1),
    #                                     input_size=img_sz[0],
    #                                     original_size=original_sz[0].squeeze().long().tolist())
    gts[gts>0.5] = 1
    gts[gts<=0.5] = 0
    J = torch.zeros(bs, 1, 256, 256).cuda()
    for inx in range(bs):
        curr_j, _, _  = get_similarity_maps(sam.eval(),
                                            imgs[inx:inx+1],
                                            gts[inx:inx+1],
                                            th=0.5,
                                            pos=args['pos'],
                                            neg=args['neg'])
        J[inx] = F.interpolate(curr_j[None, None], (256, 256), mode='bilinear', align_corners=True)
    return batched_input, gts, J


def step(ds, sam, optimizer):
    loss_list = []
    criterion = torch.nn.BCELoss()
    pbar = tqdm(ds)
    for ix, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
        batched_input, _, J = get_inputs(imgs, gts, sam.eval(), original_sz, img_sz)
        pred, _ = get_sam_model_output(sam.train(), batched_input)
        loss = loss_step(optimizer, J, pred, criterion)
        loss_list.append(loss)
        pbar.set_description(
            '(train | {}) epoch {epoch} ::'
            'loss: {loss:.4f}'.format(
                'Medical',
                epoch=ix,
                loss=np.mean(loss_list),
            ))


@torch.no_grad()
def eval_ds(ds, sam, args):
    loss_list, j_list, pred_list  = [], [], []
    pbar = tqdm(ds)
    for ix, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
        batched_input, gts, J = get_inputs(imgs, gts, sam.eval(), original_sz, img_sz)
        pred, _ = get_sam_model_output(sam.train(), batched_input)
        pred = (pred >= 0.5).float()
        dice_loss = 1 - Dice_loss(pred, J)[0]
        dice_j = 1 - Dice_loss(J, gts)[0]
        dice_pred = 1 - Dice_loss(pred, gts)[0]
        loss_list.append(dice_loss.item())
        j_list.append(dice_j.item())
        pred_list.append(dice_pred.item())
        pbar.set_description(
            '(train | {}) epoch {epoch} ::'
            'loss: {loss:.4f}, dice j: {dice_j:.4f}, dice pred: {dice_pred:.4f}'.format(
                args['task'],
                epoch=ix,
                loss=np.mean(loss_list),
                dice_j=np.mean(j_list),
                dice_pred=np.mean(pred_list),
            ))
        

def training(args=None, sam_args=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    transform = ResizeLongestSide(1024)
    if args['task'] == 'monu':
        trainset, testset = get_monu_dataset(args, sam_trans=transform)
    elif args['task'] == 'glas':
        trainset, testset = get_glas_dataset(args, sam_trans=transform)
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=device)
    opt_model = sam.mask_decoder #mask_decoder, prompt_encoder, image_encoder
    for param in sam.parameters():
        param.requires_grad = False
    for param in opt_model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(opt_model.parameters(),
                        lr=float(args['learning_rate']),
                        weight_decay=float(args['WD']))
    ds = torch.utils.data.DataLoader(trainset,
                                     batch_size=args['Batch_size'],
                                     shuffle=True,
                                     num_workers=int(args['nW']),
                                     drop_last=True)
    ds_test = torch.utils.data.DataLoader(testset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=int(args['nW_eval']),
                                          drop_last=False)
    for _ in range(args['epoches']):
        step(ds, sam.train(), optimizer)
        # eval_ds(ds, sam.eval(), args)
        eval_ds(ds_test, sam.eval(), args)
           

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr', '--learning_rate', default=1e-5, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', type=int, default=4, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', default=70, help='number of epoches', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-WD', '--WD', default=1e-5, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='glas', help='evaluation iteration', required=False)
    parser.add_argument('-datadir', '--datadir', default='data/Warwick/', help='evaluation iteration', required=False)
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=512, help='image size', required=False)
    parser.add_argument('-vit', '--vit', default='vit_b', help='image size', required=False)
    parser.add_argument('-pos', '--pos', type=int, default=5, help='image size', required=False)
    parser.add_argument('-neg', '--neg', type=int, default=5, help='image size', required=False)
    parser.add_argument('-w0', '--w0', type=float, default=1, help='image size', required=False)
    args = vars(parser.parse_args())
    os.makedirs('vis', exist_ok=True)
    os.makedirs(os.path.join('vis', args['task'], args['vit']), exist_ok=True)
    sam_args = {
        'sam_checkpoint': "cp/sam_" + args['vit'] + ".pth",
        'model_type': args['vit'],
        'generator_args': {
            'points_per_side': 8,
            'pred_iou_thresh': 0.95,
            'stability_score_thresh': 0.7,
            'crop_n_layers': 0,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 0,
            'point_grids': None,
            'box_nms_thresh': 0.7,
        },
        'gpu_id': 0,
    }
    training(args=args, sam_args=sam_args)
            
