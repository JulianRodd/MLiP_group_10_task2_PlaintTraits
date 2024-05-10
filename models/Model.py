from torch import nn
import timm 
import torch
import timm.models.vision_transformer
from timm.models.layers import trunc_normal_
from functools import partial




class Model(nn.Module):
    def __init__(self, config, model_name=None):
        super().__init__()
        self.backbone = self.get_backbone(config, model_name)
        
    def get_backbone(self, config, model_name): 
        if model_name is not None: 
            return timm.create_model(
                model_name,
                num_classes=config.N_TARGETS,
                pretrained=config.PRETRAINED)
        else: 
            return timm.create_model(
                config.MODEL,
                num_classes=config.N_TARGETS,
                pretrained=config.PRETRAINED)

        
    def forward(self, inputs):
        return self.backbone(inputs)    


class CLEFVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(timm.models.vision_transformer.VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    
class MultiRegressionVisionTransformer(CLEFVisionTransformer):
    def __init__(self, global_pool=False, **kwargs):
        super(CLEFVisionTransformer, self).__init__(**kwargs)
        
        self.head = nn.Linear(self.embed_dim, 6)
    
    ### debugging purpose only ###
    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
#         if self.attn_pool is not None:
#             x = self.attn_pool(x)
#         elif self.global_pool == 'avg':
#             x = x[:, self.num_prefix_tokens:].mean(dim=1)
#         elif self.global_pool:
#             x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)
    ###############################


def vit_base_patch16(**kwargs):
    model = MultiRegressionVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = MultiRegressionVisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = MultiRegressionVisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

models_vit = {
    "vit_base_patch16": vit_base_patch16,
    "vit_large_patch16": vit_large_patch16,
    "vit_huge_patch14": vit_huge_patch14,
}


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def get_model_clef(config, model_name, model_path):

    model = models_vit[model_name](
        num_classes=config.N_TARGETS,
        drop_path_rate=config.DROPOUT,
        global_pool=config.GLOBAL_POOL,
        )

    checkpoint = torch.load(model_path, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % model_path)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)

    return model.to(config.DEVICE)