# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
import torch.nn as nn

# from torchscale.architecture.encoder import Encoder
from lib.models.moe.encoder import Encoder
# from torchscale.component.embedding import (
#     PositionalEmbedding,
#     TextEmbedding,
#     VisionEmbedding,
# )
from lib.models.moe.embedding import (
    PositionalEmbedding,
    TextEmbedding,
    VisionEmbedding,
)
from torchscale.component.multiway_network import MutliwayEmbedding


class BEiT3(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        assert args.multiway
        assert args.vocab_size > 0
        assert not args.share_encoder_input_output_embed
        self.text_embed = TextEmbedding(args.vocab_size, args.encoder_embed_dim)
        self.vision_embed = VisionEmbedding(
            args.img_size,                       # 搜索区域尺寸大小
            args.patch_size,
            args.in_chans,
            args.encoder_embed_dim,
            contain_mask_token=True,
            prepend_cls_token=True,
        )
        self.template_embed = VisionEmbedding(
            192,                                  # 模板尺寸大小
            args.patch_size,
            args.in_chans,
            args.encoder_embed_dim,
            contain_mask_token=True,
            prepend_cls_token=True,
        )
        # being consistent with Fairseq, which starts from 2 for position embedding
        # embed_positions = MutliwayEmbedding(
        #     modules=[
        #         PositionalEmbedding(self.vision_embed.num_position_embeddings() + 2, args.encoder_embed_dim),
        #         PositionalEmbedding(args.max_source_positions, args.encoder_embed_dim),
        #     ],
        #     dim=1,
        # )

        # 多模态position embedding
        embed_positions = MutliwayEmbedding(
            modules=[
                # PositionalEmbedding(self.template_embed.num_position_embeddings() + 2, args.encoder_embed_dim),   # 模板
                PositionalEmbedding(self.vision_embed.num_position_embeddings() + self.template_embed.num_position_embeddings() + 2, args.encoder_embed_dim),    # 搜索区域
                PositionalEmbedding(args.max_source_positions, args.encoder_embed_dim),    # 文本
            ],
            dim=1,
        )

        self.encoder = Encoder(
            args,
            embed_tokens=None,
            embed_positions=embed_positions,
            output_projection=None,
            is_encoder_decoder=False,
        )

    def forward(
        self,
        template_tokens=None,
        textual_tokens=None,
        visual_tokens=None,
        text_padding_position=None,
        attn_mask=None,
        vision_masked_position=None,
        incremental_state=None,
        positions=None,
    ):
        assert textual_tokens is not None or visual_tokens is not None

        if textual_tokens is None:
            x = self.vision_embed(visual_tokens, vision_masked_position)
            encoder_padding_mask = None
            multiway_split_position = -1
        elif visual_tokens is None:
            x = self.text_embed(textual_tokens)
            encoder_padding_mask = text_padding_position
            multiway_split_position = 0
        elif template_tokens is None and visual_tokens is not None and textual_tokens is not None:   # 仅使用language模态
            x1 = self.vision_embed(visual_tokens, vision_masked_position)              # [1， 577， 768]
            multiway_split_position = x1.size(1)
            x2 = self.text_embed(textual_tokens)                       # [1, 40, 768]
            x = torch.cat([x1, x2], dim=1)            # [1, 617, 768]

            if text_padding_position is not None:
                encoder_padding_mask = torch.cat(
                    [
                        torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                        text_padding_position,
                    ],
                    dim=1,
                )
            else:
                encoder_padding_mask = None
        
        else:    # 使用language和template模态

            # x1 = self.vision_embed(visual_tokens, vision_masked_position)          
            # multiway_split_position = x1.size(1)
            # x2 = self.text_embed(textual_tokens)
            # x = torch.cat([x1, x2], dim=1)

            # if text_padding_position is not None:
            #     encoder_padding_mask = torch.cat(
            #         [
            #             torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
            #             text_padding_position,
            #         ],
            #         dim=1,
            #     )
            # else:
            #     encoder_padding_mask = None


            ######################################
            x1 = self.vision_embed(visual_tokens, vision_masked_position)     # 搜索区域
            x2 = self.text_embed(textual_tokens)            # NLP文本
            x3 = self.template_embed(template_tokens)    # 模板
            multiway_split_position = x1.size(1) + x3.size(1)
            x = torch.cat([x3, x1, x2], dim=1)     # 模板+搜索区域+NLP文本

            if text_padding_position is not None:
                encoder_padding_mask = torch.cat(
                    [
                        torch.zeros(x3.shape[:-1]).to(x3.device).bool(),
                        torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                        text_padding_position,
                    ],
                    dim=1,
                )
            else:
                encoder_padding_mask = None

        encoder_out = self.encoder(
            src_tokens=None,
            encoder_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,           # None
            token_embeddings=x,             # [1,762,768]
            multiway_split_position=multiway_split_position,
            incremental_state=incremental_state,  # None
            positions=positions,   # None
        )
        encoder_out["multiway_split_position"] = multiway_split_position

        return encoder_out
