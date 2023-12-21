import torch


def ckbd_split(y):
    """
    Split y to anchor and non-anchor
    anchor :
        0 1 0 1 0
        1 0 1 0 1
        0 1 0 1 0
        1 0 1 0 1
        0 1 0 1 0
    non-anchor:
        1 0 1 0 1
        0 1 0 1 0
        1 0 1 0 1
        0 1 0 1 0
        1 0 1 0 1
    """
    anchor = ckbd_anchor(y)
    nonanchor = ckbd_nonanchor(y)
    return anchor, nonanchor

def ckbd_merge(anchor, nonanchor):
    # out = torch.zeros_like(anchor).to(anchor.device)
    # out[:, :, 0::2, 0::2] = non_anchor[:, :, 0::2, 0::2]
    # out[:, :, 1::2, 1::2] = non_anchor[:, :, 1::2, 1::2]
    # out[:, :, 0::2, 1::2] = anchor[:, :, 0::2, 1::2]
    # out[:, :, 1::2, 0::2] = anchor[:, :, 1::2, 0::2]

    return anchor + nonanchor

def ckbd_anchor(y):
    anchor = torch.zeros_like(y).to(y.device)
    anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
    anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
    return anchor

def ckbd_nonanchor(y):
    nonanchor = torch.zeros_like(y).to(y.device)
    nonanchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
    nonanchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
    return nonanchor

def ckbd_anchor_sequeeze(y):
    B, C, H, W = y.shape
    anchor = torch.zeros([B, C, H, W // 2]).to(y.device)
    anchor[:, :, 0::2, :] = y[:, :, 0::2, 1::2]
    anchor[:, :, 1::2, :] = y[:, :, 1::2, 0::2]
    return anchor

def ckbd_nonanchor_sequeeze(y):
    B, C, H, W = y.shape
    nonanchor = torch.zeros([B, C, H, W // 2]).to(y.device)
    nonanchor[:, :, 0::2, :] = y[:, :, 0::2, 0::2]
    nonanchor[:, :, 1::2, :] = y[:, :, 1::2, 1::2]
    return nonanchor

def ckbd_anchor_unsequeeze(anchor):
    B, C, H, W = anchor.shape
    y_anchor = torch.zeros([B, C, H, W * 2]).to(anchor.device)
    y_anchor[:, :, 0::2, 1::2] = anchor[:, :, 0::2, :]
    y_anchor[:, :, 1::2, 0::2] = anchor[:, :, 1::2, :]
    return y_anchor

def ckbd_nonanchor_unsequeeze(nonanchor):
    B, C, H, W = nonanchor.shape
    y_nonanchor = torch.zeros([B, C, H, W * 2]).to(nonanchor.device)
    y_nonanchor[:, :, 0::2, 0::2] = nonanchor[:, :, 0::2, :]
    y_nonanchor[:, :, 1::2, 1::2] = nonanchor[:, :, 1::2, :]
    return y_nonanchor


def dual_split(y):
    ch_grp_1, ch_grp_2 = y.chunk(2, 1)
    ch_grp_1_anchor, ch_grp_1_nonanchor = ckbd_split(ch_grp_1)
    ch_grp_2_anchor, ch_grp_2_nonanchor = ckbd_split(ch_grp_2)
    y_anchor = torch.cat([ch_grp_1_anchor, ch_grp_2_nonanchor], dim=1)
    y_nonanchor = torch.cat([ch_grp_1_nonanchor, ch_grp_2_anchor], dim=1)
    return y_anchor, y_nonanchor

def dual_anchor(y):
    ch_grp_1, ch_grp_2 = y.chunk(2, 1)
    ch_grp_1_anchor = ckbd_anchor(ch_grp_1)
    ch_grp_2_anchor = ckbd_nonanchor(ch_grp_2)
    y_anchor = torch.cat([ch_grp_1_anchor, ch_grp_2_anchor], dim=1)
    return y_anchor

def dual_nonanchor(y):
    ch_grp_1, ch_grp_2 = y.chunk(2, 1)
    ch_grp_1_nonanchor = ckbd_nonanchor(ch_grp_1)
    ch_grp_2_nonanchor = ckbd_anchor(ch_grp_2)
    y_nonanchor = torch.cat([ch_grp_1_nonanchor, ch_grp_2_nonanchor], dim=1)
    return y_nonanchor

def dual_anchor_sequeeze(y):
    ch_grp_1, ch_grp_2 = y.chunk(2, 1)
    ch_grp_1_anchor = ckbd_anchor_sequeeze(ch_grp_1)
    ch_grp_2_anchor = ckbd_nonanchor_sequeeze(ch_grp_2)
    y_anchor = torch.cat([ch_grp_1_anchor, ch_grp_2_anchor], dim=1)
    return y_anchor

def dual_nonanchor_sequeeze(y):
    ch_grp_1, ch_grp_2 = y.chunk(2, 1)
    ch_grp_1_nonanchor = ckbd_nonanchor_sequeeze(ch_grp_1)
    ch_grp_2_nonanchor = ckbd_anchor_sequeeze(ch_grp_2)
    y_nonanchor = torch.cat([ch_grp_1_nonanchor, ch_grp_2_nonanchor], dim=1)
    return y_nonanchor

def dual_anchor_unsequeeze(y):
    ch_grp_1, ch_grp_2 = y.chunk(2, 1)
    ch_grp_1_anchor = ckbd_anchor_unsequeeze(ch_grp_1)
    ch_grp_2_anchor = ckbd_nonanchor_unsequeeze(ch_grp_2)
    y_anchor = torch.cat([ch_grp_1_anchor, ch_grp_2_anchor], dim=1)
    return y_anchor

def dual_nonanchor_unsequeeze(y):
    ch_grp_1, ch_grp_2 = y.chunk(2, 1)
    ch_grp_1_nonanchor = ckbd_nonanchor_unsequeeze(ch_grp_1)
    ch_grp_2_nonanchor = ckbd_anchor_unsequeeze(ch_grp_2)
    y_nonanchor = torch.cat([ch_grp_1_nonanchor, ch_grp_2_nonanchor], dim=1)
    return y_nonanchor
