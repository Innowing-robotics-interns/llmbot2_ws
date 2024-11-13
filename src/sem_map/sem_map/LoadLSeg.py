Lang_Seg_Path = '/home/fyp/lang-seg'

import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import clip
from modules.lseg_module_zs import LSegModuleZS
from additional_utils.models import LSeg_MultiEvalModule
from fewshot_data.common.logger import Logger, AverageMeter
from fewshot_data.common.vis import Visualizer
from fewshot_data.common.evaluation import Evaluator
from fewshot_data.common import utils
from fewshot_data.data.dataset import FSSDataset
from test_lseg_zs import Options

import sys
sys.argv = [
    'ipykernel_launcher.py',  # Script name
    '--backbone', 'clip_vitl16_384',
    '--module', 'clipseg_DPT_test_v2',
    '--dataset', 'fss',
    '--widehead', 
    '--no-scaleinv', 
    '--arch_option', '0',
    '--ignore_index', '255',
    '--fold', '0',
    '--nshot', '0',
    '--weights', Lang_Seg_Path+'/checkpoints/demo_e200.ckpt',
    '--datapath', 'data/Datasets_HSN'
]
args = Options().parse()

# Change working directory
os.chdir(Lang_Seg_Path)

from modules.models.lseg_blocks_zs import forward_vit
class FeatureLSeg(LSegModuleZS):
    def __init__(self, *args, **kwargs):
        super(FeatureLSeg, self).__init__(*args, **kwargs)
    
    def forward(self, x):

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.net.pretrained, x)

        layer_1_rn = self.net.scratch.layer1_rn(layer_1)
        layer_2_rn = self.net.scratch.layer2_rn(layer_2)
        layer_3_rn = self.net.scratch.layer3_rn(layer_3)
        layer_4_rn = self.net.scratch.layer4_rn(layer_4)

        path_4 = self.net.scratch.refinenet4(layer_4_rn)
        path_3 = self.net.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.net.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.net.scratch.refinenet1(path_2, layer_1_rn)

        self.net.logit_scale = self.net.logit_scale.to(x.device)

        image_features = self.net.scratch.head1(path_1)
        imshape = image_features.shape
        image_features = image_features.permute(0,2,3,1).reshape(-1, self.net.out_c)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        pixel_encoding = self.net.logit_scale * image_features.half() 
        pixel_encoding = pixel_encoding.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)
        pixel_encoding = self.net.scratch.output_conv(pixel_encoding)
        pixel_encoding = pixel_encoding.squeeze(0).permute(1, 2, 0)
        pixel_encoding = pixel_encoding / pixel_encoding.norm(dim=-1, keepdim=True)
            
        return pixel_encoding
    
    def encode_text(self, text, device=torch.device(type='cuda',index=0)):
        text = clip.tokenize(text).to(device)
        text_feature = self.net.clip_pretrained.encode_text(text.to(device))
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        return text_feature

print("Loading model...")
model_feat = FeatureLSeg.load_from_checkpoint(
    checkpoint_path=args.weights,
    data_path=args.datapath,
    dataset=args.dataset,
    backbone=args.backbone,
    aux=args.aux,
    num_features=256,
    aux_weight=0,
    se_loss=False,
    se_weight=0,
    base_lr=0,
    batch_size=1,
    max_epochs=0,
    ignore_index=args.ignore_index,
    dropout=0.0,
    scale_inv=args.scale_inv,
    augment=False,
    no_batchnorm=False,
    widehead=args.widehead,
    widehead_hr=args.widehead_hr,
    map_locatin="cpu",
    arch_option=args.arch_option,
    use_pretrained=args.use_pretrained,
    strict=args.strict,
    logpath=Lang_Seg_Path+'/fewshot/logpath_4T/',
    fold=args.fold,
    block_depth=0,
    nshot=args.nshot,
    finetune_mode=False,
    activation=args.activation,
)
print("Model loaded.")

model = model_feat.eval().cuda()

