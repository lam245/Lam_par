# import torch.nn as nn
# import torch
# from clip import clip
# from models.vit import *
# from config import argument_parser
# parser = argument_parser()
# args = parser.parse_args()
# class TransformerClassifier(nn.Module):
#     def __init__(self, clip_model, attr_num, attributes, dim=768, pretrain_path='/data/jinjiandong/jx_vit_base_p16_224-80ecf9dd.pth'):
#         super().__init__()
#         super().__init__()
#         self.attr_num = attr_num
#         self.word_embed = nn.Linear(clip_model.visual.output_dim, dim)
#         vit = vit_base()
#         vit.load_param(pretrain_path)
#         self.norm = vit.norm
#         self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
#         self.dim = dim
#         self.text = clip.tokenize(attributes).to("cuda")
#         self.bn = nn.BatchNorm1d(self.attr_num)
#         fusion_len = self.attr_num + 257 + args.vis_prompt
#         if not args.use_mm_former :
#             print('Without MM-former, Using MLP Instead')
#             self.linear_layer = nn.Linear(fusion_len, self.attr_num)
#         else:
#             self.blocks = vit.blocks[-args.mm_layers:]
#     def forward(self,imgs,clip_model):
#         b_s=imgs.shape[0]
#         clip_image_features,all_class,attenmap=clip_model.visual(imgs.type(clip_model.dtype))
#         text_features = clip_model.encode_text(self.text).to("cuda").float()
#         if args.use_div:
#             final_similarity,logits_per_image = clip_model.forward_aggregate(all_class,text_features)
#         else : 
#             final_similarity = None
#         textual_features = self.word_embed(text_features).expand(b_s, self.attr_num, self.dim)
#         x = torch.cat([textual_features,clip_image_features], dim=1)
        
#         if args.use_mm_former:
#             for blk in self.blocks:
#                 x = blk(x)
#         else :# using linear layer fusion
#             x = x.permute(0, 2, 1)
#             x= self.linear_layer(x)
#             x = x.permute(0, 2, 1)
            
#         x = self.norm(x)
#         logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
#         bn_logits = self.bn(logits)
        
        
#         return bn_logits,final_similarity

import torch.nn as nn
import torch
from clip import clip
from models_v2_rope import rope_mixed_deit_base_patch16_LS
from config import argument_parser

parser = argument_parser()
args = parser.parse_args()

class TransformerClassifier(nn.Module):
    def __init__(self, clip_model, attr_num, attributes, dim=768, pretrain_path='/data/jinjiandong/jx_vit_base_p16_224-80ecf9dd.pth'):
        super().__init__()
        self.attr_num = attr_num
        self.word_embed = nn.Linear(clip_model.visual.output_dim, dim)
        
        # Replace vit_base with rope_mixed_deit_base_patch16_LS
        vit = rope_mixed_deit_base_patch16_LS()
        vit.load_state_dict(torch.load(pretrain_path, map_location='cpu'), strict=False)
        
        self.norm = vit.norm
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.dim = dim
        self.text = clip.tokenize(attributes).to("cuda")
        self.bn = nn.BatchNorm1d(self.attr_num)
        
        fusion_len = self.attr_num + 257 + args.vis_prompt
        if not args.use_mm_former:
            print('Without MM-former, Using MLP Instead')
            self.linear_layer = nn.Linear(fusion_len, self.attr_num)
        else:
            self.blocks = vit.blocks[-args.mm_layers:]

    def forward(self, imgs, clip_model):
        b_s = imgs.shape[0]
        clip_image_features, all_class, attenmap = clip_model.visual(imgs.type(clip_model.dtype))
        text_features = clip_model.encode_text(self.text).to("cuda").float()
        
        if args.use_div:
            final_similarity, logits_per_image = clip_model.forward_aggregate(all_class, text_features)
        else:
            final_similarity = None
            
        textual_features = self.word_embed(text_features).expand(b_s, self.attr_num, self.dim)
        x = torch.cat([textual_features, clip_image_features], dim=1)
        
        if args.use_mm_former:
            # For RoPE ViT, we need to compute the position embeddings
            if hasattr(self.blocks[0], 'rope_mixed') and self.blocks[0].rope_mixed:
                # Using mixed RoPE
                t_x, t_y = self.blocks[0].init_t_xy(end_x=x.shape[1], end_y=1)  # Adjust dimensions as needed
                freqs_cis = self.blocks[0].compute_cis(self.blocks[0].freqs, t_x.to(x.device), t_y.to(x.device))
                for i, blk in enumerate(self.blocks):
                    x = blk(x, freqs_cis=freqs_cis[i])
            else:
                # Using axial RoPE
                freqs_cis = self.blocks[0].compute_cis(end_x=x.shape[1], end_y=1).to(x.device)
                for blk in self.blocks:
                    x = blk(x, freqs_cis=freqs_cis)
        else:
            # Using linear layer fusion
            x = x.permute(0, 2, 1)
            x = self.linear_layer(x)
            x = x.permute(0, 2, 1)
            
        x = self.norm(x)
        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        bn_logits = self.bn(logits)
        
        return bn_logits, final_similarity