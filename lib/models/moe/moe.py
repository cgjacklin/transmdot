import torch
import torch.nn as nn
from torchscale.architecture.config import EncoderConfig
from lib.models.moe.modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config
from .utils import combine_tokens, recover_tokens
from transformers import XLMRobertaTokenizer
from lib.models.layers.patch_embed import PatchEmbed
from torchvision.transforms import Resize
#         from torchscale.component.embedding import (
#     PositionalEmbedding,
#     TextEmbedding,
#     VisionEmbedding,
# )

class MultiwayTransformer(BEiT3Wrapper):
    def __init__(self, args, norm_layer=nn.LayerNorm,  **kwargs):
        super(MultiwayTransformer, self).__init__(args=args, **kwargs)
        self.embed_dim = args.encoder_embed_dim



    def forward_features(self,z, x, nl_token_ids, nl_token_masks,):

        # torchResize = Resize([x.shape[2], x.shape[3]])
        # z = torchResize(z)
        # x = torch.cat([z, x], dim=1)   # [B, 3, 224, 224] -> [B, 6, 224, 224]


        # tokenizer = XLMRobertaTokenizer("/home/visiondata/chenguanlin/transmdot/pretrained_models/beit3.spm")

        # nlp_token = tokenizer.tokenize(n[0])
        # seq_length = 40



        # print("visual_tokens", x.shape)              # 暂时还没加上模板
        # print("textual_tokens", nl_token_ids.shape)
        nl_token_ids = nl_token_ids.transpose(0, 1)
        nl_token_masks = nl_token_masks.transpose(0, 1)
        output = self.beit3(template_tokens=z,
                            textual_tokens=nl_token_ids,
                            visual_tokens=x,
                            text_padding_position=nl_token_masks,
                            )



        return output




    def forward(self, z, x, nl_token_ids,nl_token_masks,
                return_last_attn=False):
        
        x = self.forward_features(z, x, nl_token_ids, nl_token_masks)

        return x






# def _create_moe(args, pretrained=False, **kwargs):
#     if kwargs.get('features_only', None):
#         raise RuntimeError('features_only not implemented for Vision Transformer models.')

#     model = MultiwayTransformer(args, **kwargs)

#     if pretrained:
#         if 'npz' in pretrained:
#             model.load_pretrained(pretrained, prefix='')
#         else:
#             checkpoint = torch.load(pretrained, map_location="cpu")
#             missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
#             print('Load pretrained model from: ' + pretrained)

#     return model


# 文本-搜索模态
# def _create_moe(args, pretrained=False, **kwargs):
#     if kwargs.get('features_only', None):
#         raise RuntimeError('features_only not implemented for Vision Transformer models.')

#     model = MultiwayTransformer(args, **kwargs)

#     if pretrained:
#         if 'npz' in pretrained:
#             model.load_pretrained(pretrained, prefix='')
#         else:
#             checkpoint = torch.load(pretrained, map_location="cpu")

#             checkpoint = checkpoint["model"]

#             # interpolate position embedding
#             for pos_embed_key in ("vision_pos_embed", "pos_embed", "beit3.encoder.embed_positions.A.weight"):    # (A:(724,768),B:(1024,768))
#                 if pos_embed_key in checkpoint:
#                     pos_embed_checkpoint = checkpoint[pos_embed_key]
#                     embedding_size = pos_embed_checkpoint.shape[-1]
#                     if pos_embed_key == "beit3.encoder.embed_positions.A.weight":
#                         # being consistent with Fairseq, which starts from 2 for position embedding
#                         torchscale_model = True

#                         # 原来的
#                         num_patches = model.beit3.vision_embed.num_patches
#                         num_extra_tokens = model.beit3.vision_embed.num_position_embeddings() + 2 - num_patches

#                         # 模板+搜索+语言
#                         # num_patches = model.beit3.vision_embed.num_patches + model.beit3.template_embed.num_patches
#                         # num_extra_tokens = model.beit3.vision_embed.num_position_embeddings() + model.beit3.template_embed.num_position_embeddings() + 2 - num_patches

#                     else:
#                         torchscale_model = False
#                         num_patches = model.patch_embed.num_patches
#                         num_extra_tokens = getattr(model, pos_embed_key).shape[-2] - num_patches
#                     # height (== width) for the checkpoint position embedding
#                     orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)    # 14x14
#                     # height (== width) for the new position embedding
#                     new_size = int(num_patches ** 0.5)
#                     # class_token and dist_token are kept unchanged
#                     if orig_size != new_size:
#                         print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
#                         if torchscale_model:
#                             extra_tokens = pos_embed_checkpoint[:num_extra_tokens].unsqueeze(0)
#                             # only the position tokens are interpolated
#                             pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
#                         else:
#                             extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
#                             # only the position tokens are interpolated
#                             pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
#                         pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
#                         pos_tokens = torch.nn.functional.interpolate(
#                             pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
#                         pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
#                         new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
#                         if torchscale_model:
#                             new_pos_embed = new_pos_embed.squeeze(0)
#                         checkpoint[pos_embed_key] = new_pos_embed    # 需要724*768



#             missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
#             print('Load pretrained model from: ' + pretrained)

#     return model




def _create_moe(args, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = MultiwayTransformer(args, **kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")

            checkpoint = checkpoint["model"]

            # interpolate position embedding
            for pos_embed_key in ("vision_pos_embed", "pos_embed", "beit3.encoder.embed_positions.A.weight"):    # (A:(724,768),B:(1024,768))
                if pos_embed_key in checkpoint:
                    pos_embed_checkpoint = checkpoint[pos_embed_key]
                    embedding_size = pos_embed_checkpoint.shape[-1]
                    if pos_embed_key == "beit3.encoder.embed_positions.A.weight":
                        # being consistent with Fairseq, which starts from 2 for position embedding
                        torchscale_model = True

                        # 原来的
                        # num_patches = model.beit3.vision_embed.num_patches
                        # num_extra_tokens = model.beit3.vision_embed.num_position_embeddings() + 2 - num_patches

                        # 模板+搜索+语言
                        num_search_patches = model.beit3.vision_embed.num_patches      # 576
                        num_temp_paches = model.beit3.template_embed.num_patches          # 144
                        num_extra_tokens = 3

                    else:
                        torchscale_model = False
                        num_patches = model.patch_embed.num_patches
                        num_extra_tokens = getattr(model, pos_embed_key).shape[-2] - num_patches
                    # height (== width) for the checkpoint position embedding
                    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)    # 14x14
                    extra_tokens = pos_embed_checkpoint[:num_extra_tokens].unsqueeze(0)
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[num_extra_tokens:]    # [196,768]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)


                    # 模板
                    new_size_temp = int(num_temp_paches ** 0.5)
                    pos_tokens_temp = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size_temp, new_size_temp), mode='bicubic', align_corners=False)
                    print("Template Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size_temp, new_size_temp))
                    pos_tokens_temp = pos_tokens_temp.permute(0, 2, 3, 1).flatten(1, 2)

                    # 搜索
                    new_size_search = int(num_search_patches ** 0.5)
                    pos_tokens_search = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size_search, new_size_search), mode='bicubic', align_corners=False)
                    print("Search region Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size_search, new_size_search))
                    pos_tokens_search = pos_tokens_search.permute(0, 2, 3, 1).flatten(1, 2)


                    new_pos_embed = torch.cat((extra_tokens, pos_tokens_temp, extra_tokens[:,-1,:].unsqueeze(0), pos_tokens_search), dim=1)
                    if torchscale_model:
                        new_pos_embed = new_pos_embed.squeeze(0)
                    checkpoint[pos_embed_key] = new_pos_embed

                    # # height (== width) for the new position embedding
                    # new_size = int(num_patches ** 0.5)
                    # # class_token and dist_token are kept unchanged
                    # if orig_size != new_size:
                    #     print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                    #     if torchscale_model:
                    #         extra_tokens = pos_embed_checkpoint[:num_extra_tokens].unsqueeze(0)
                    #         # only the position tokens are interpolated
                    #         pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
                    #     else:
                    #         extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    #         # only the position tokens are interpolated
                    #         pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    #     pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    #     pos_tokens = torch.nn.functional.interpolate(
                    #         pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    #     pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    #     new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    #     if torchscale_model:
                    #         new_pos_embed = new_pos_embed.squeeze(0)
                    #     checkpoint[pos_embed_key] = new_pos_embed



            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model













# def _get_base_config(
#         img_size=224, patch_size=16, drop_path_rate=0, 
#         checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
# ):
#     return EncoderConfig(
#         img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
#         layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
#         drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12, 
#         encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12, 
#         checkpoint_activations=checkpoint_activations, 
#     )



def moe_base_patch16_224(pretrained=False, **kwargs):
    args = _get_base_config(img_size=384, **kwargs)
    # args = _get_base_config(img_size=224, **kwargs)

    model = _create_moe(args, pretrained=pretrained ,**kwargs)
    return model



