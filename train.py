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


def preprocess_logits(logits, size=256, th=0.01, is_resize=True):
    logits = F.sigmoid(logits)
    logits = (logits - logits.min()) / (logits.max() - logits.min())
    if is_resize:
        logits =  F.interpolate(logits, (size, size), mode='bilinear', align_corners=True)
    logits[logits>th] = 1
    logits[logits<=th] = 0
    return logits


def gen_step(optimizer, out_mask, logits, score, criterion):
    loss = 0
    dice_loss, _ = Dice_loss(logits, out_mask)
    bce_loss = criterion(out_mask, logits)
    loss = loss + 0.00*(1 - score) + dice_loss + bce_loss
    loss.backward()
    optimizer.step()
    return loss.item(), dice_loss.item(), bce_loss.item()


def train_single_image(imgs, gts, original_sz, img_sz, sam, optimizer, args, file, ix):
    loss_list = []
    image = imgs.to(sam.device)
    batched_input = get_input_dict(image, original_sz, img_sz)
    pbar = trange(int(args['epoches']))
    criterion = torch.nn.BCELoss()
    gts_large = sam.postprocess_masks(gts.unsqueeze(dim=0),
                                        input_size=img_sz[0],
                                        original_size=original_sz.squeeze().long().tolist())
    gts_large[gts_large>0.5] = 1
    gts_large[gts_large<=0.5] = 0
    with torch.no_grad():
        zero_pred, p_i, n_i  = get_similarity_maps(sam.eval(), imgs, gts.unsqueeze(dim=0).cuda(), th=0.5, pos=args['pos'], neg=args['neg'])
        points_pred = zero_sam_with_points(sam, batched_input, p_i, n_i)
        masks_pred = zero_sam_with_masks(sam, zero_pred, batched_input)
    zero_pred =  F.interpolate(zero_pred.unsqueeze(dim=0).unsqueeze(dim=0),
                              (256, 256),
                               mode='bilinear', align_corners=True).cuda()
    zero_pred[zero_pred>0.5]=1
    zero_pred[zero_pred<=0.5]=0
    zero_pred_large = sam.postprocess_masks(zero_pred,
                                            input_size=img_sz[0],
                                            original_size=original_sz.squeeze().long().tolist())
    zero_pred_large[zero_pred_large>0.5] = 1
    zero_pred_large[zero_pred_large<=0.5] = 0
    points_pred_large = sam.postprocess_masks(points_pred.unsqueeze(dim=0).unsqueeze(dim=0),
                                            input_size=img_sz[0],
                                            original_size=original_sz.squeeze().long().tolist())
    points_pred_large[points_pred_large>0.5] = 1
    points_pred_large[points_pred_large<=0.5] = 0
    masks_pred_large = sam.postprocess_masks(masks_pred.unsqueeze(dim=0).unsqueeze(dim=0),
                                            input_size=img_sz[0],
                                            original_size=original_sz.squeeze().long().tolist())
    masks_pred_large[masks_pred_large>0.5] = 1
    masks_pred_large[masks_pred_large<=0.5] = 0
    
    best = 0.0
    res2 = 0.0
    best_mask = None
    best_i = None
    for i in pbar:
        optimizer.zero_grad()
        out, score = get_sam_model_output(sam.train(), batched_input, zero_pred, it=args['it'])
        loss, _, _ = gen_step(optimizer, out, zero_pred.detach(), score, criterion)
        with torch.no_grad():
            pred = out
            pred[pred>0.5] = 1
            pred[pred<=0.5] = 0
            pred = sam.postprocess_masks(pred,
                                         input_size=img_sz[0],
                                         original_size=original_sz.squeeze().long().tolist())
            dice_gt, ji_gt = get_dice_ji(pred.cpu().numpy(), gts_large.cpu().numpy())
            _, ji = get_dice_ji(pred.cpu().numpy(), zero_pred_large.cpu().numpy())
        if ji >= best:
            best = ji
            res2 = (dice_gt, ji_gt)
            best_mask = pred.clone()
            best_i = i
        loss_list.append(loss)
        pbar.set_description(
            '(train | {}) epoch {epoch} ::'
            ' res {res:.4f}'.format(
                'Medical',
                epoch=i,
                res=res2[1]
            ))
    if best_mask is None:
        best_mask = pred
        best_i = args['epoches']
    res1 = get_dice_ji(zero_pred_large.cpu().numpy(), gts_large.cpu().numpy())
    file.write(str(ix) + ',' + 
               str(best_i) + ',' +
               str(best)[:6] + ',' +
               str(res1[1])[:6] + ',' +
               str(res2[1])[:6] + '\n')
    file.flush()
    return res1, res2


def zero_sam(args=None, sam_args=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    transform = ResizeLongestSide(1024)
    if args['task'] == 'monu':
        trainset, testset = get_monu_dataset(args, sam_trans=transform)
    elif args['task'] == 'glas':
        trainset, testset = get_glas_dataset(args, sam_trans=transform)
    ds = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=int(args['nW_eval']), drop_last=False)
    pbar = tqdm(ds)
    file = open(os.path.join('vis', args['task'], args['vit'], 'results.csv'), 'w')
    file.write('file,best_ix,iou_loss,iou_owl,iou_gt\n')
    file.flush()
    dice1_list, dice2_list, iou1_list, iou2_list = [], [], [], []
    for ix, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
        sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
        sam.to(device=device)
        opt_model = sam.image_encoder #mask_decoder, prompt_encoder, image_encoder
        for param in sam.parameters():
            param.requires_grad = False
        for param in opt_model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(opt_model.parameters(),
                            lr=float(args['learning_rate']),
                            weight_decay=float(args['WD']))
        res1, res2 = train_single_image(imgs, gts, original_sz, img_sz, sam.train(), optimizer, args, file, ix)
        dice1_list.append(res1[0])
        dice2_list.append(res2[0])
        iou1_list.append(res1[1])
        iou2_list.append(res2[1])
        pbar.set_description(
            '(train | {}) epoch {epoch} ::'
            'iou1: {iou1:.4f} :: iou2 {iou2:.4f}'.format(
                'Medical',
                epoch=ix,
                iou1=np.mean(iou1_list),
                iou2=np.mean(iou2_list)
            ))
    return np.mean(dice1_list), np.mean(iou1_list), np.mean(dice2_list), np.mean(iou2_list)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr', '--learning_rate', default=1e-5, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=1, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', default=2, help='number of epoches', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-WD', '--WD', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='glas', help='evaluation iteration', required=False)
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=512, help='image size', required=False)
    parser.add_argument('-it', '--it', default=1, help='image size', required=False)
    parser.add_argument('-vit', '--vit', default='vit_b', help='image size', required=False)
    parser.add_argument('-pos', '--pos', default=1, help='image size', required=False)
    parser.add_argument('-neg', '--neg', default=1, help='image size', required=False)
    args = vars(parser.parse_args())
    os.makedirs('vis', exist_ok=True)
    os.makedirs(os.path.join('vis', args['task'], args['vit']), exist_ok=True)
    sam_args = {
        'sam_checkpoint': "/home/tal/MedicalSam/cp/sam_" + args['vit'] + ".pth",
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
    name = args['task'] + '_' + args['vit'] + '_' +str(args['epoches']) 
    f = open('results_' + name + '.csv', 'w')
    f.write('dataset,vit,pos,neg,dice1,iou1,dice2,iou2\n')
    f.flush()
    for points in [(1,1), (2,2), (3,3), (4,4), (5,5), (1,0), (2,0), (3,0), (4,0), (5,0)]:
        args['pos'] = points[0]
        args['neg'] = points[1]
        dice1,iou1,dice2,iou2 = zero_sam(args=args, sam_args=sam_args)
        f.write(args['task'] + ',' +
                args['vit'] + ',' +
                str(args['pos']) + ',' +
                str(args['neg']) + ',' +
                str(dice1) + ',' + 
                str(iou1) + ',' +
                str(dice2) + ',' + 
                str(iou2) + '\n'
                )
        f.flush()
