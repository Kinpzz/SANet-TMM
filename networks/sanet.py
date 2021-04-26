import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from networks import ResNet101, dpn92, Decoder, ASPP, AMTreeGRU

class mutan_head(nn.Module):
    def __init__(self, vis_dim, lang_dim, hid_dim):
        super(mutan_head, self).__init__()
        self.vis_conv = nn.Conv2d(vis_dim, hid_dim, kernel_size=1, bias=True)
        self.lang_conv = nn.Conv2d(lang_dim, hid_dim, kernel_size=1, bias=True)

    def forward(self, vis_feat, lang_feat):
        # vis_feat : [B,C,H,W]
        # lang_feat: [B,C,H,W]
        vis_trans = self.vis_conv(vis_feat)
        vis_trans = torch.tanh(vis_trans)
        lang_trans = self.lang_conv(lang_feat)
        lang_trans = torch.tanh(lang_trans)
        x_mm = vis_trans * lang_trans
        return x_mm

class mutan_fusion(nn.Module):
    def __init__(self, vis_dim, lang_dim, hid_dim):
        super(mutan_fusion, self).__init__()
        self.mutan_head1 = mutan_head(vis_dim, lang_dim, hid_dim)
        self.mutan_head2 = mutan_head(vis_dim, lang_dim, hid_dim)
        self.mutan_head3 = mutan_head(vis_dim, lang_dim, hid_dim)
        self.mutan_head4 = mutan_head(vis_dim, lang_dim, hid_dim)
        self.mutan_head5 = mutan_head(vis_dim, lang_dim, hid_dim)

    def forward(self, vis_feat, lang_feat):
        # vis_feat : [B,C,H,W]
        # lang_feat: [B,C,H,W]
        mutan_head1 = self.mutan_head1(vis_feat, lang_feat)
        mutan_head2 = self.mutan_head2(vis_feat, lang_feat)
        mutan_head3 = self.mutan_head3(vis_feat, lang_feat)
        mutan_head4 = self.mutan_head4(vis_feat, lang_feat)
        mutan_head5 = self.mutan_head5(vis_feat, lang_feat)
        fuse_feats = torch.stack((mutan_head1, mutan_head2, mutan_head3, mutan_head4, mutan_head5), dim=1)
        fuse_feats = fuse_feats.sum(dim=1)
        fuse_feats = torch.tanh(fuse_feats)
        return fuse_feats

class SANet(nn.Module):
    """
        Inputs: vis, lang
        - **vis** of shape :math:`(1, 3, H, W)`: tensor containing an input
        image in format RGB.
        - **lang** of shape :math:`(1, L)`: tensor containing an referral
        expression given as a number sequence.
    """
    def __init__(self, dict_size, emb_size=300, hid_size=256, mix_size=256,
                 vis_size=256, tree_hid_size=256, lang_layers=1,
                 output_stride=16, num_classes=1,
                 pretrained_backbone=True,
                 pretrained_embedding=None,
                 dataset=None,
                 backbone='resnet101'):

        super(SANet, self).__init__()

        # language model: word embdding + bi-lstm
        if pretrained_embedding == 'glove':
            glove_emb   = np.load('data/{}/glove_emb.npy'.format(dataset))
            self.emb    = nn.Embedding.from_pretrained(torch.from_numpy(glove_emb), freeze=False)
        else:
            self.emb    = nn.Embedding(dict_size, emb_size)
        self.lang_model = nn.LSTM(emb_size, hid_size//2, num_layers=lang_layers, bidirectional=True)

        if backbone == 'resnet101':
            self.backbone = ResNet101(output_stride, nn.BatchNorm2d, pretrained_backbone)
            c5c4c3_dim = 2048+1024+512
            low_level_inplanes = 256
        elif backbone == 'dpn92':
            self.backbone = dpn92(pretrained=pretrained_backbone)
            c5c4c3_dim = 2592
            low_level_inplanes = 64

        self.aspp       = ASPP(mix_size, output_stride, nn.BatchNorm2d)
        self.decoder    = Decoder(mix_size, low_level_inplanes, num_classes, nn.BatchNorm2d)

        self.aux_conv = nn.Sequential(nn.Conv2d(mix_size, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.Conv2d(64, num_classes, kernel_size=1, stride=1))
        self.mutan_fusion = mutan_fusion(vis_size, hid_size, hid_size)
        self.comb_conv  = nn.Sequential(nn.Conv2d(in_channels=(tree_hid_size + hid_size),
                            out_channels=mix_size,
                            kernel_size=1,
                            padding=0),
                            nn.BatchNorm2d(mix_size))
        self.visual_conv = nn.Sequential(nn.Conv2d(in_channels=c5c4c3_dim, out_channels=vis_size, kernel_size=1, padding=0),
                                         nn.BatchNorm2d(vis_size))

        self.MTGRU = AMTreeGRU(vis_size, hid_size, tree_hid_size)

    def extract_language_feat(self, lang):
        # input dim:  L x B
        # output dim: L x B x word_embbding_dim
        lang_emb = self.emb(lang)

        # output dim: L x B x hidden_dim
        lang_feat, _ = self.lang_model(lang_emb)
        return lang_feat

    def forward(self, vis, lang, adjs, words_len):
        # Extract visual feature
        # Encoder
        c5, c4, c3, low_level_feat = self.backbone(vis)
        c4 = F.interpolate(c4, size=c5.shape[2:], mode='bilinear', align_corners=True)
        c3 = F.interpolate(c3, size=c5.shape[2:], mode='bilinear', align_corners=True)

        vis_feat = self.visual_conv(torch.cat((c5, c4, c3), dim=1))

        batch_size, _, featmap_H, featmap_W = vis_feat.size()
        lang_feat = self.extract_language_feat(lang.permute(1,0)) # L X B X hidden
        lang_feat = F.normalize(lang_feat, p=2, dim=2)

        sent_feat = lang_feat[-1].unsqueeze(-1).unsqueeze(-1).expand(
                    batch_size, lang_feat.size(-1),
                    featmap_H, featmap_W)

        root_feats = []
        root_atts = []
        for i in range(batch_size):
            root_feat, root_att = self.MTGRU(vis_feat[[i]], lang_feat[:words_len[i],[i],:], adjs[i,:words_len[i],:words_len[i]])
            root_feats.append(root_feat)
            root_atts.append(root_att)
        root_feats = torch.cat(root_feats, dim=0)
        root_atts = torch.cat(root_atts, dim=0)
        mm_feat = self.mutan_fusion(vis_feat, sent_feat)
        mix_feat = torch.cat((mm_feat, root_feats), dim=1)
        mix_feat = self.comb_conv(mix_feat)

        mix_feat = self.aspp(mix_feat)
        seg_aux = self.aux_conv(mix_feat)
        # Decoder: multidal features to referring segmenation
        seg = self.decoder(mix_feat, low_level_feat)
        seg = F.interpolate(seg, size=vis.size()[2:], mode='bilinear', align_corners=True)
        seg_aux = F.interpolate(seg_aux, size=vis.size()[2:], mode='bilinear', align_corners=True)
        return seg, seg_aux, root_atts
