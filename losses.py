import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn



class MaskedNCC:
    def __init__(self, win=9):
        self.win = win

    def loss(self, y_true, y_pred):
        Ii = y_true.squeeze(dim=2)
        Ji = y_pred.squeeze(dim=2)

        c = y_pred.shape[1]

        # Check dimension
        ndims = len(Ii.size()) - 2


        assert ndims in [1, 2, 3], "Only supports 1-3D inputs."

        win = [self.win] * ndims
        pad_no = self.win // 2
        padding = (pad_no,) * ndims
        stride = (1,) * ndims

        sum_filt = torch.ones([1, c, *win]).to(y_pred.device)

        conv_fn = getattr(F, f'conv{ndims}d')

        # Compute masks of valid regions (non-zero)
        mask_I = (Ii != 0).float()
        mask_J = (Ji != 0).float()
        valid_mask = mask_I * mask_J  # intersection mask

        # Shrink mask to exclude border areas affected by convolution
        valid_mask = conv_fn(valid_mask, sum_filt, stride=stride, padding=padding)
        threshold = 0.9 * np.prod(win)
        valid_mask = (valid_mask >= threshold).float()
        # valid_mask = (valid_mask == np.prod(win)).float()

        # print("Valid elements in NCC mask:", torch.sum(valid_mask).item())

        # Compute local sums
        I2, J2, IJ = Ii*Ii, Ji*Ji, Ii*Ji
        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2*u_I*I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2*u_J*J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + 1e-5)

        # Apply mask to cc map
        cc_masked = cc * valid_mask

        # Avoid division by zero
        valid_elements = torch.sum(valid_mask) + 1e-5

        return -torch.sum(cc_masked) / valid_elements

def global_cosine_loss(I, J):
    I_flat = I.view(I.size(0), -1)
    J_flat = J.view(J.size(0), -1)
    sim = F.cosine_similarity(I_flat, J_flat, dim=1)
    return -sim.mean()


class NCC:
    def __init__(self, win=9):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred
        c = y_pred.shape[1]

        # get dimension of volume
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([1, c, *win]).to(y_pred.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)





# This NJD loss works at PyTorch=1.10(cuda10.2) but failed at PyTorch=1.13 for unknown reasons
class NJD:
    def __init__(self, Lambda=1e-5):
        self.Lambda = Lambda

    def get_Ja(self, displacement):
        D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
        D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
        D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])

        D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
        D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
        D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

        return D1 - D2 + D3

    def loss(self, _, y_pred):
        displacement = y_pred.permute(0, 2, 3, 4, 1)
        Ja = self.get_Ja(displacement)
        Neg_Jac = 0.5 * (torch.abs(Ja) - Ja)

        return self.Lambda * torch.sum(Neg_Jac)

class Grad:
    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, y_pred):

        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dy = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad
def Regu_loss( y_pred):
    return Grad('l2').loss( y_pred)
    # Disable NJD loss if PyTorch>1.10
    # return Grad('l2').loss(y_true, y_pred) + NJD(1e-5).loss(y_true, y_pred)



class CurvatureLoss(nn.Module):
    def forward(self, flow):
        # flow: [B, C, D, H, W]
        loss = 0.0
        count = 0

        # Check H dimension (Height)
        if flow.shape[3] >= 3:
            dx = flow[..., :, 1:, :] - flow[..., :, :-1, :]
            dxx = dx[..., :, 1:, :] - dx[..., :, :-1, :]
            if dxx.numel() > 0:
                loss += (dxx ** 2).mean()
                count += 1

        # Check W dimension (Width)
        if flow.shape[4] >= 3:
            dy = flow[..., :, :, 1:] - flow[..., :, :, :-1]
            dyy = dy[..., :, :, 1:] - dy[..., :, :, :-1]
            if dyy.numel() > 0:
                loss += (dyy ** 2).mean()
                count += 1

        # Do NOT use D axis if D == 1
        return loss / count if count > 0 else torch.tensor(0.0, device=flow.device)


class ReguLossCurvature(nn.Module):
    def __init__(self, alpha=1.0, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.curvature_loss = CurvatureLoss()

    def forward(self, flow):
        curvature = self.curvature_loss(flow)
        sparsity = torch.var(torch.norm(flow, dim=1, keepdim=True))
        return self.alpha * curvature + self.beta * sparsity
