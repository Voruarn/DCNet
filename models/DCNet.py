from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.vision_transformer import PatchEmbed, Block
from models.ViT import PatchEmbed, Block, ConvBlock
from util.pos_embed import get_2d_sincos_pos_embed
from models.ResNet import resnet18, resnet34, resnet50


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    


class DCNet(nn.Module):
    """ DCNet
    """
    def __init__(self, aux_encoder='resnet18', num_classes=2, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=24, num_heads=16,
                 decoder_embed_dim=[64, 128, 256, 512], 
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # DCNet encoder specifics
        # aux_encoder=resnet18, resent34, resnet50
        self.depth=depth
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.encoder = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
      
        aux_enc_dims=[64,128,256,512]
        if aux_encoder=='resnet50':
            aux_enc_dims=[256,512,1024,2048]

        self.aux_encoder=eval(aux_encoder)()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # decoder specifics
        self.decoder_embed1 = ConvModule(embed_dim, decoder_embed_dim[1])
        self.decoder_embed2 = ConvModule(embed_dim, decoder_embed_dim[2])
        self.decoder_embed3 = ConvModule(embed_dim, decoder_embed_dim[3])
        self.decoder_embed4 = ConvModule(embed_dim, decoder_embed_dim[3])

        self.ae_side1=ConvModule(aux_enc_dims[0], decoder_embed_dim[1])
        self.ae_side2=ConvModule(aux_enc_dims[1], decoder_embed_dim[2])
        self.ae_side3=ConvModule(aux_enc_dims[2], decoder_embed_dim[3])
        self.ae_side4=ConvModule(aux_enc_dims[3], decoder_embed_dim[3])


        self.dec_block4=ConvBlock(decoder_embed_dim[3], decoder_embed_dim[3], norm_layer=norm_layer)
        self.dec_block3=nn.Sequential(ConvModule(decoder_embed_dim[3],decoder_embed_dim[2]), 
                                    ConvBlock(decoder_embed_dim[2], decoder_embed_dim[2], norm_layer=norm_layer))
        self.dec_block2=nn.Sequential(ConvModule(decoder_embed_dim[2],decoder_embed_dim[1]), 
                                    ConvBlock(decoder_embed_dim[1], decoder_embed_dim[1], norm_layer=norm_layer))
        self.dec_block1=nn.Sequential(ConvModule(decoder_embed_dim[1],decoder_embed_dim[0]), 
                                    ConvBlock(decoder_embed_dim[0], decoder_embed_dim[0], norm_layer=norm_layer))
                           
        
        self.fpn41 = nn.Sequential(
                nn.ConvTranspose2d(decoder_embed_dim[3], decoder_embed_dim[1], kernel_size=2, stride=2),
                nn.BatchNorm2d(decoder_embed_dim[1]),
                nn.GELU(),
                nn.ConvTranspose2d(decoder_embed_dim[1], decoder_embed_dim[1], kernel_size=2, stride=2),
            ) # H/4 * W/4
        self.fpn42 = nn.Sequential(
            nn.ConvTranspose2d(decoder_embed_dim[3], decoder_embed_dim[2], kernel_size=2, stride=2),
        )
        self.fpn43 = nn.Identity()
        self.fpn44 = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.fpn22 = nn.Sequential(
            nn.ConvTranspose2d(decoder_embed_dim[2], decoder_embed_dim[2], kernel_size=2, stride=2),
        )
        self.fpn11 = nn.Sequential(
                nn.ConvTranspose2d(decoder_embed_dim[1], decoder_embed_dim[1], kernel_size=2, stride=2),
                nn.BatchNorm2d(decoder_embed_dim[1]),
                nn.GELU(),
                nn.ConvTranspose2d(decoder_embed_dim[1], decoder_embed_dim[1], kernel_size=2, stride=2),
            ) # H/4 * W/4

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dside1 = nn.Conv2d(decoder_embed_dim[0], num_classes, 3, padding=1)
        self.dside2 = nn.Conv2d(decoder_embed_dim[1], num_classes, 3, padding=1)
        self.dside3 = nn.Conv2d(decoder_embed_dim[2], num_classes, 3, padding=1)
        self.dside4 = nn.Conv2d(decoder_embed_dim[3], num_classes, 3, padding=1)
        
        
        self.initialize_weights()
        # initialise weights

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        # embed patches
        B,C,H,W=x.shape
        # print('1x.shape:',x.shape)
        x, (H, W) = self.patch_embed(x)

        # print('2x.shape:',x.shape)
        """
        1x.shape: torch.Size([2, 3, 256, 256])
        2x.shape: torch.Size([2, 256, 768])
        """
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        outs=[]

        stage_pace=self.depth//4 # 12//4=3; 24//4=6
        # apply Transformer blocks
        for i, blk in enumerate(self.encoder):
            x = blk(x)
            if (i+1)%stage_pace==0: # 2, 5, 8, 11
                # remove cls token and save the intermediate feature for the decoder
                # print('x.shape:', x.shape) # x.shape: torch.Size([2, 257, 768])
                re_x=self.norm(x[:, 1:, :]) 
                # print('re_x.shape:', re_x.shape)  # re_x.shape: torch.Size([2, 256, 768])
                re_x = re_x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                # x1.shape:torch.Size([2, 768, 16, 16])
                outs.append(re_x)

        return outs

    def forward_decoder(self, x, aux_outs, H, W):
        # aux_encoder_outputs
        a1,a2,a3,a4=aux_outs
        a1=self.ae_side1(a1)
        a2=self.ae_side2(a2)
        a3=self.ae_side3(a3)
        a4=self.ae_side4(a4)

        # embed tokens
        c1,c2,c3,c4=x
        c1=self.decoder_embed1(c1)
        c1=self.fpn11(c1)

        c2=self.decoder_embed2(c2)
        c2=self.fpn22(c2)
        c3=self.decoder_embed3(c3)
        c4=self.decoder_embed4(c4)
        
        c41=self.fpn41(c4)
        c42=self.fpn42(c4)
        c43=self.fpn43(c4)
        c44=self.fpn44(c4)

        up4=a4 + self.dec_block4(c44)

        up3=a3 + c43 + self.upsample2(up4)
        up3=self.dec_block3(up3)

        up2=a2 + c42 + self.upsample2(up3)
        up2=self.dec_block2(up2)

        up1=a1 + c41 + self.upsample2(up2)
        up1=self.dec_block1(up1)

        d1=self.dside1(up1)
        d2=self.dside2(up2)
        d3=self.dside3(up3)
        d4=self.dside4(up4)

        S1 = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True)
        S2 = F.interpolate(d2, size=(H, W), mode='bilinear', align_corners=True)
        S3 = F.interpolate(d3, size=(H, W), mode='bilinear', align_corners=True)
        S4 = F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=True)
    
        return S1,S2,S3,S4


    def forward(self, imgs):
        B,C,H,W=imgs.shape
        latent = self.forward_encoder(imgs)
        aux_outs=self.aux_encoder(imgs)
        pred = self.forward_decoder(latent, aux_outs, H, W)  # [N, L, p*p*3]
        return pred


def dcnet_vit_tiny_patch16(img_size=256, **kwargs):
    model = DCNet(img_size=img_size,
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=[64, 64, 64, 64], 
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def dcnet_vit_small_patch16(img_size=256, **kwargs):
    model = DCNet(img_size=img_size,
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=[64, 128, 256, 320], 
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def dcnet_vit_base_patch16_dec512d8b(img_size=256, **kwargs):
    model = DCNet(img_size=img_size,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=[64, 128, 256, 512], 
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


dcnet_vit_tiny = dcnet_vit_tiny_patch16
dcnet_vit_small = dcnet_vit_small_patch16
dcnet_vit_base = dcnet_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
