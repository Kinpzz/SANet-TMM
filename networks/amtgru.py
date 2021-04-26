import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        padding = self.kernel_size // 2
        self.reset_gate  = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=padding)
        self.update_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=padding)
        self.output_gate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=padding)
        # init
        for m in self.state_dict():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, hidden):
        if hidden is None:
            size_h = [x.data.size()[0], self.hidden_channels] + list(x.data.size()[2:])
            hidden = torch.zeros(size_h).cuda()

        inputs       = torch.cat((x, hidden), dim=1)
        reset_gate   = F.sigmoid(self.reset_gate(inputs))
        update_gate  = F.sigmoid(self.update_gate(inputs))

        reset_hidden = reset_gate * hidden
        reset_inputs = F.tanh(self.output_gate(torch.cat((x, reset_hidden), dim=1)))
        new_hidden   = (1 - update_gate)*reset_inputs + update_gate*hidden

        return new_hidden

class ConvTreeGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super(ConvTreeGRUCell, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2

        self.reset_gate  = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=padding, bias=bias)
        self.update_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=padding, bias=bias)
        self.output_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=padding, bias=bias)

    def forward(self, x, child_h):
        """
            inputs : B x C x H x W
            child_h: L x B x C x H x W
            output : B x C x H x W
        """
        if child_h is None:
            size_h = [1, x.size()[0], self.hidden_dim] + list(x.size()[2:])
            child_h = torch.zeros(size_h).cuda()
        child_h_sum = torch.sum(child_h, dim=0, keepdim=False)

        inputs = torch.cat([x, child_h_sum], dim=1)
        r = torch.sigmoid(
            self.reset_gate(
                torch.cat([x.repeat(child_h.size(0), 1, 1, 1, 1), child_h], dim=2).squeeze(1)
            )
        ).unsqueeze(1) # [L, batch, C, h, w]
        z = torch.sigmoid(self.update_gate(inputs))
        reset_hidden = torch.sum(r * child_h, dim=0, keepdim=False) #[batch, C, h, w]
        reset_inputs = torch.tanh(
            self.output_gate(torch.cat((x, reset_hidden), dim=1))
        )
        h = (1-z)*reset_inputs + z*child_h_sum

        return h

class MTreeGRU(nn.Module):
    def __init__(self, vis_dim, lang_dim, hidden_dim, kernel_size=3, bias=True):
        super(MTreeGRU, self).__init__()
        self.mconv = nn.Conv2d(vis_dim+lang_dim, hidden_dim, 1, 1, 0)
        self.MTreeGRUCell = ConvTreeGRUCell(hidden_dim, hidden_dim, kernel_size, bias)

    def compute_node_feat(self, node_vis_feat, node_lang_feat, child_feats):
        # combine multi-modal features
        node_feat = torch.cat((node_vis_feat, node_lang_feat), dim=1)
        node_feat = self.mconv(node_feat)
        node_feat = self.MTreeGRUCell(node_feat, child_feats)
        return node_feat

    def node_forward(self, vis_feat, lang_feat, adj, idx):
        node = (adj[idx, :] > 0).nonzero()
        if len(node) == 0: #leaf
            node_feat = self.compute_node_feat(
                vis_feat[idx], lang_feat[idx],
                None)
        else:
            child_feats = []
            for dep_idx in node:
                dep_idx = int(dep_idx)
                feat = self.node_forward(vis_feat, lang_feat, adj, dep_idx) # recursive
                child_feats.append(feat)
            child_feats = torch.stack(child_feats)
            node_feat = self.compute_node_feat(vis_feat[idx], lang_feat[idx],
                child_feats)

        return node_feat

    def forward(self, vis_feat, lang_feat, adj):
        # dep_graph direction (row_id -> col_id)
        # sum by col / keep row
        root = (torch.sum(adj, dim=0) == 0).nonzero()
        root = int(root[0])
        # L x 1 x C x H x W
        vis_feat_expand = vis_feat.unsqueeze(0).expand(lang_feat.size(0),
            1, vis_feat.size(1), vis_feat.size(2), vis_feat.size(3))
        # lang_feat : L x 1 x C -> Lx1xCxHxW
        lang_feat_tile = lang_feat.unsqueeze(-1).unsqueeze(-1).expand(lang_feat.size(0),
            1, lang_feat.size(-1), vis_feat.size(2), vis_feat.size(3))
        root_feat = self.node_forward(vis_feat_expand, lang_feat_tile, adj, root)
        return root_feat

class AMTreeGRU(nn.Module):
    def __init__(self, vis_dim, lang_dim, hidden_dim, kernel_size=3, bias=True):
        super(AMTreeGRU, self).__init__()
        self.mconv = nn.Conv2d(2*vis_dim+lang_dim, hidden_dim, 1, 1, 0, bias=True)
        self.MTreeGRUCell = ConvTreeGRUCell(hidden_dim, hidden_dim, kernel_size, bias)
        self.theta_v = nn.Conv2d(hidden_dim,  128, 1, bias=True)
        self.theta_l = nn.Conv2d(lang_dim, 128, 1, bias=True)
        self.psi = nn.Conv2d(128, 1, 1, bias=True)
        self.W = nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=True)

        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.child_fc = nn.Linear(256, 256)
        #self.node_fc = nn.Linear(256, 256)
        #self.alpha_fc = nn.Linear(512, 1)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def multimodal_attention(self, node_feat, lang_feat):
        theta_node_feat = self.theta_v(node_feat)
        theta_lang_feat = self.theta_l(lang_feat)
        f = F.softplus(theta_node_feat + theta_lang_feat) # soft relu
        att = torch.sigmoid(self.psi(f)) # 1x1xhxw
        y = att.expand_as(node_feat) * node_feat
        W_y = self.W(y)
        return W_y, att

    def compute_node_feat(self, node_vis_feat, node_lang_feat, child_feats, child_atts):
        att_in = 1 - child_atts.mean(dim=0, keepdim=True)
        # if child_feats is None: #leaf
        #     att_in = 1 - torch.zeros_like(child_atts).float().cuda()
        # else:
        #     att_in = torch.zeros_like(child_atts).float().cuda()
        #     for i in range(child_atts.size(0)):
        #         #import ipdb; ipdb.set_trace()
        #         child_f = self.child_fc(self.avgpool(child_feats[i]).squeeze())
        #         node_f = self.node_fc(self.avgpool(node_lang_feat).squeeze())
        #         alpha = torch.sigmoid(self.alpha_fc(torch.cat((child_f, node_f), dim=0)))
        #         att_in += alpha*child_atts[[i]] + (1-alpha)*(1-child_atts[[i]])
        #     att_in = att_in.mean(dim=0, keepdim=True)
        node_context = att_in.expand_as(node_vis_feat) * node_vis_feat
        # combine multi-modal features
        node_feat = torch.cat((node_vis_feat, node_context, node_lang_feat), dim=1)
        node_feat = self.mconv(node_feat)
        node_feat, node_att = self.multimodal_attention(node_feat, node_lang_feat)
        node_feat = self.MTreeGRUCell(node_feat, child_feats)
        return node_feat, node_att

    def node_forward(self, vis_feat, lang_feat, adj, idx):
        node = (adj[idx, :] > 0).nonzero()
        if len(node) == 0: #leaf
            child_atts = torch.zeros(1, 1, vis_feat.size(-2), vis_feat.size(-1)).float().cuda()
            node_feat, node_att = self.compute_node_feat(
                vis_feat[idx], lang_feat[idx],
                None, child_atts)
        else:
            child_feats = []
            child_atts = []
            for dep_idx in node:
                dep_idx = int(dep_idx)
                feat, att = self.node_forward(vis_feat, lang_feat, adj, dep_idx) # recursive
                child_feats.append(feat)
                child_atts.append(att)
            child_feats = torch.stack(child_feats)
            child_atts = torch.cat(child_atts, dim=0)
            node_feat, node_att = self.compute_node_feat(vis_feat[idx], lang_feat[idx],
                child_feats, child_atts)

        return node_feat, node_att

    def forward(self, vis_feat, lang_feat, adj):
        # dep_graph direction (row_id -> col_id)
        # sum by col / keep row
        root = (torch.sum(adj, dim=0) == 0).nonzero()
        root = int(root[0])
        # L x 1 x C x H x W
        vis_feat_expand = vis_feat.unsqueeze(0).expand(lang_feat.size(0),
            1, vis_feat.size(1), vis_feat.size(2), vis_feat.size(3))
        # lang_feat : L x 1 x C -> Lx1xCxHxW
        lang_feat_tile = lang_feat.unsqueeze(-1).unsqueeze(-1).expand(lang_feat.size(0),
            1, lang_feat.size(-1), vis_feat.size(2), vis_feat.size(3))
        root_feat, root_att = self.node_forward(vis_feat_expand, lang_feat_tile, adj, root)
        return root_feat, root_att
