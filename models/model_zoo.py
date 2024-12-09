import torch.nn as nn
import torch
import timm
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from .baevit import RSNA_BAEViT
from timm.models.registry import register_model


@register_model
def rsna_baevit(pretrained=False, num_classes=1000, drop_path_rate=0.2, layer_lr_decay=1.0):
    return RSNA_BAEViT(
        img_size=512,
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[16, 16, 32, 16],
        drop_path_rate=drop_path_rate,
        layer_lr_decay=layer_lr_decay,
    )


class TimmRegressor(nn.Module):
    def __init__(self, model_name, feature_dim, is_sigmoid=False, img_size=224, config=None):
        super().__init__()
        self.model_name = model_name
        self.is_sigmoid = is_sigmoid
        self.feature_dim = feature_dim
        self.img_size = img_size

        if 'tiny_vit' in model_name:
            self.model = timm.create_model(model_name, pretrained=False, num_classes=0,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                layer_lr_decay=config.MI.LAYER_LR_DECAY)
        else:
            self.model = timm.create_model(model_name, pretrained=False, num_classes=0)

        self.linear = torch.nn.Linear(feature_dim, 1)
        if is_sigmoid:
            self.sigmoid = torch.nn.Sigmoid()
        else:
            self.sigmoid = None

    def load_pretrained_model(self, pretrained_path):
        ckpt = torch.load(pretrained_path, map_location='cpu')
        try:
            load_pretrained_model(self.model, ckpt['model'])
        except KeyError:
            load_pretrained_model(self.model, ckpt)
        del ckpt


    def forward(self, x):
        if self.sigmoid:
            return self.sigmoid(self.linear(self.model(x)))
        else:
            return self.linear(self.model(x))

def load_pretrained_model(model, ckpt_state):
    new_ckpt_state = {}
    for key in ckpt_state:
        if key.startswith('model.'):
            new_key = key.split('model.', 1)[1]
            new_ckpt_state[new_key] = ckpt_state[key]
        else:
            new_ckpt_state[key] = ckpt_state[key]
    ckpt_state = new_ckpt_state

    model_state = model.state_dict()

    ## Drop head weights
    mnames = ['head.weight', 'head.bias']
    if mnames[-1] in model_state:
        if ckpt_state[mnames[-1]].shape != model_state[mnames[-1]].shape:
            for mname in mnames:
                ckpt_state.pop(mname)
    ## Drop attn mask
    for key in model_state:
        if ('attn_mask' in key or 'relative_position_index' in key) and key in ckpt_state:
            ckpt_state.pop(key)
    ## Interpolate relative pos bias table
    all_key_matched = True
    for key in model_state:
        if 'relative_position_bias_table' in key and key in ckpt_state: ## Swin
            # table: (L, num_heads)
            l_target, nh_target = model_state[key].size()
            l_source, nh_source = ckpt_state[key].size()
            if l_target != l_source:
                print(f'Interpolate the position bias table from {l_source} to {l_target}...')
                assert nh_source == nh_target, f"Different number of heads detected! {nh_source} VS {nh_target}..."
                sl_source = int(l_source ** 0.5)
                sl_target = int(l_target ** 0.5)
                value = F.interpolate(ckpt_state[key].permute(1,0).view(1, nh_source, sl_source, sl_source), size=(sl_target, sl_target), mode='bicubic') # (1, nh, l2, l2)

                value = value.reshape(nh_source, l_target).permute(1, 0)
                ckpt_state[key] = value
        elif 'attention_biases' in key and key in ckpt_state:
            nh_target, l_target = model_state[key].size()
            nh_source, l_source = ckpt_state[key].size()
            if l_target != l_source:
                print(f'Interpolate the position bias table from {l_source} to {l_target}...')
                assert nh_source == nh_target, f"Different number of heads detected! {nh_source} VS {nh_target}..."
                sl_source = int(l_source ** 0.5)
                sl_target = int(l_target ** 0.5)
                value = F.interpolate(ckpt_state[key].view(1, nh_source, sl_source, sl_source), size=(sl_target, sl_target), mode='bicubic') # (1, nh, l2, l2)

                value = value.reshape(nh_source, l_target)
                ckpt_state[key] = value
        elif 'local_conv.c' in key and key in ckpt_state:
            _, _, s1, s2 = model_state[key].size()
            _, _, t1, t2 = ckpt_state[key].size()
            if (s1 != t1) or (s2 != t2):
                print(f'Interpolate the local convolution size ({key}) from ({s1},{s2}) to ({t1},{t2})...')
                value = F.interpolate(ckpt_state[key], size=(s1, s2), mode='bicubic')
                ckpt_state[key] = value
        elif key == 'pos_embed': # DeiT
            # pos_embed (1, n, dim)
            _, l_target, dim_target = model_state[key].size()
            _, l_source, dim_source = ckpt_state[key].size()
            if l_target != l_source:
                print(f'Interpolate the position bias table from {l_source} to {l_target}...')
                assert dim_target == dim_source, f"Different number of heads detected! {dim_source} VS {dim_target}..."
                sl_source = int((l_source-1) ** 0.5)
                sl_target = int((l_target-1) ** 0.5)
                ## pos embed for cls token is at the first row
                value = F.interpolate(ckpt_state[key][0, 1:, :].permute(1, 0).view(1, dim_source, sl_source, sl_source), size=(sl_target, sl_target), mode='bicubic')
                value = value.reshape(dim_source, sl_target*sl_target).permute(1, 0)
                value = torch.cat([ckpt_state[key][0, 0, :].unsqueeze(0), value])
                ckpt_state[key] = value.unsqueeze(0)
        if key in ckpt_state:
            model_state[key] = ckpt_state[key]
        else:
            all_key_matched = False
            print(f"Key not found in the checkpoint: {key}")
    if all_key_matched:
        print(f"All keys successfully matched!")

    model.load_state_dict(model_state, strict=True)