import torch
import torch.nn as nn


def AF(bev_proposal_rois, img_proposal_rois):
    """
        objs : 1xDxN
    """
    gate_p = nn.Sequential(
            nn.Conv1d(bev_proposal_rois, bev_proposal_rois, 1, 1),
            nn.Sigmoid(),)  # 2xDxL

    gate_i = nn.Sequential(
            nn.Conv1d(img_proposal_rois, img_proposal_rois, 1, 1),
            nn.Sigmoid(),)  # 2xDxL

    obj_fused = gate_p.mul(bev_proposal_rois) + 
                gate_i.mul(img_proposal_rois)

    return obj_fused

def CF(bev_proposal_rois, img_proposal_rois):
    """
        objs : 1xDxN
    """
    gate_p = nn.Sequential(
            nn.Conv1d(bev_proposal_rois, bev_proposal_rois, 1, 1),
            nn.Sigmoid(),)  # 2xDxL

    gate_i = nn.Sequential(
            nn.Conv1d(img_proposal_rois, img_proposal_rois, 1, 1),
            nn.Sigmoid(),)  # 2xDxL

    obj_fused = torch.cat(gate_p.mul(bev_proposal_rois) , 
                gate_i.mul(img_proposal_rois))

    return obj_fused




