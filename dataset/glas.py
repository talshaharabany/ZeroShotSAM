import torch
# from torch.utils.data.sampler import WeightedRandomSampler
from dataset.tfs import get_glas_transform
# from tfs import get_glas_transform
import cv2
import os
import glob

def cv2_loader(path, is_mask):
    if is_mask:
        img = cv2.imread(path, 0)
        img[img > 0] = 1
    else:
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img

def filter_path():
    f = open('vis/glas/vit_l/results.csv', 'r')
    buffer = f.readlines()
    good = []
    for item in buffer[1:]:
        if float(item.split(',')[2]) > 0.0:
            good.append(item.split(',')[0])
    imgs_path = glob.glob('vis/glas/images/*.jpg')
    mask_path = glob.glob('vis/glas/vit_l/*_pred.jpg')
    imgs_path.sort()
    mask_path.sort()
    out_imgs, out_masks = [], []
    for img_path, mask_path in zip(imgs_path, mask_path):
        ix = img_path.split('/')[-1].split('_')[0]
        if ix in good:
            out_imgs.append(img_path)
            out_masks.append(mask_path)
    return out_imgs, out_masks



class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=False, loader=cv2_loader,
                 split=None, val=False, sam_trans=None, loop=1, is_finetune=False):
        self.root = root
        self.is_finetune = is_finetune
        self.imgs_path = []
        self.mask_path = []  
        if is_finetune:
            self.root = ''
            self.imgs_path, self.mask_path = filter_path()
        else:
            files = os.listdir(root)
            files.sort()
            for file in files:
                if file.split('.')[-1] == 'csv':
                    continue
                if len(file.split('_')) == 2:
                    if file.split('_')[0] == 'train' and train:
                        self.imgs_path.append(file)
                    elif (file.split('_')[0] == 'testA' or file.split('_')[0] == 'testB') and not train:
                        self.imgs_path.append(file)
                else:
                    if file.split('_')[0] == 'train' and train:
                        self.mask_path.append(file)
                    elif (file.split('_')[0] == 'testA' or file.split('_')[0] == 'testB') and not train:
                        self.mask_path.append(file)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.val = val
        self.split = split
        self.data_size = len(self.imgs_path)
        self.sam_trans = sam_trans
        self.loop = loop
        # print('num of data:{}'.format(self.data_size))

    def __getitem__(self, index):
        index = index % self.data_size
        file_path = self.imgs_path[index]
        if self.is_finetune:
            mask_path = self.mask_path[index]
        else:
            mask_path = file_path.split('.')[0] + '_anno.bmp'
        img = self.loader(os.path.join(self.root, file_path), is_mask=False)
        mask = self.loader(os.path.join(self.root, mask_path), is_mask=True)
        img, mask = self.transform(img, mask)
        original_size = tuple(img.shape[1:3])
        img, mask = self.sam_trans.apply_image_torch(img), self.sam_trans.apply_image_torch(mask)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        image_size = tuple(img.shape[1:3])
        return self.sam_trans.preprocess(img), self.sam_trans.preprocess(mask), torch.Tensor(
            original_size), torch.Tensor(image_size)

    def __len__(self):
        return self.data_size * self.loop


def get_glas_dataset(args, sam_trans, aug=True, is_finetune=False, loop=10):
    datadir = '/home/tal/Medical/Warwick/'
    transform_train, transform_test = get_glas_transform()
    if aug:
        ds_train = ImageLoader(datadir, train=True, is_finetune=is_finetune, transform=transform_train, sam_trans=sam_trans, loop=loop)
    else:
        ds_train = ImageLoader(datadir, train=True, is_finetune=is_finetune, transform=transform_test, sam_trans=sam_trans, loop=1)
    ds_test = ImageLoader(datadir, train=False, transform=transform_test, sam_trans=sam_trans)
    return ds_train, ds_test


if __name__ == "__main__":
    from tqdm import tqdm
    import argparse
    import os
    from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
    from segment_anything.utils.transforms import ResizeLongestSide

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-pSize', '--pSize', default=8, help='learning_rate', required=False)
    parser.add_argument('-th_inter', '--th_inter', default=0, help='is load check point?', required=False)
    parser.add_argument('-K_fold', '--K_fold', default=False, help='is load check point?', required=False)
    parser.add_argument('-K', '--K', default=1, help='is load check point?', required=False)
    parser.add_argument('-vit', '--vit', default='vit_l', help='is load check point?', required=False)
    args = vars(parser.parse_args())

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
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=torch.device('cuda', sam_args['gpu_id']))
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    trainset, testset = get_glas_dataset(args, sam_trans=transform, is_finetune=True, loop=3)
    ds = torch.utils.data.DataLoader(trainset,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=True,
                                     drop_last=True)
    pbar = tqdm(ds)
    for i, (img, mask, _, _) in enumerate(pbar):
        a = img.squeeze().permute(1, 2, 0).cpu().numpy()
        b = mask.squeeze().cpu().numpy()
        a = (a - a.min()) / (a.max() - a.min())
        cv2.imwrite('kaki.jpg', 255*a)
        cv2.imwrite('kaki_mask.jpg', 255*b)


