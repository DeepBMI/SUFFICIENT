
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# from torch.autograd import Function
# import pysitk.simple_itk_helper as sitkh
# import itk
# from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import SimpleITK as sitk
import argparse
import sys
import pysitk.python_helper as ph
from sufficient.networks.s2vnet import S2vnet, SpatialTransformer
import logging
import torchgeometry as tgm
import math
import matplotlib.pyplot as plot
from sufficient.transform import axisangle2mat,RigidTransform,mat_update_resolution
from sufficient.slice_acquisition import slice_acquisition
from scipy import ndimage

# dtype = torch.FloatTensor
dtype = torch.float32
np.set_printoptions(precision=8)
# torch.manual_seed(0)


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, device,win=None):
        self.win = win
        self.device = device

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device)

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



def normalized_cross_correlation(x, x_ref):
    if x.shape != x_ref.shape:
        raise ValueError("Input data shapes do not match")

    ncc = np.sum((x - x.mean()) * (x_ref - x_ref.mean()))
    ncc /= float(x.size * x.std(ddof=1) * x_ref.std(ddof=1))

    return ncc

def resampleSize(sitkImage, x,y,z):
    euler3d = sitk.Euler3DTransform()

    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    new_spacing_x = xspacing/(x/float(xsize))
    new_spacing_y = yspacing/(y/float(ysize))
    new_spacing_z = zspacing/(z/float(zsize))

    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    newsize = (int(xsize*xspacing/new_spacing_x),int(ysize*yspacing/new_spacing_y),int(zsize*zspacing/new_spacing_z))
    newspace = (new_spacing_x, new_spacing_y, new_spacing_z)
    sitkImage = sitk.Resample(sitkImage,newsize,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
    return sitkImage

def resampleSpacing(sitkImage, newspace=(0.8,0.8,0.8)):
    euler3d = sitk.Euler3DTransform()
    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    new_size = (int(xsize*xspacing/newspace[0]),int(ysize*yspacing/newspace[1]),int(zsize*zspacing/newspace[2]))
    sitkImage = sitk.Resample(sitkImage,new_size,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
    return sitkImage


def defineModel(model_type,device):
    pretrain_model_str = model_type

    if model_type == 'S2vnet_50':
        model_ft = S2vnet(layers=[3, 4, 6, 3])
    elif model_type == 'S2vnet_101' or model_type == 'test':
        model_ft = S2vnet(layers=[3, 4, 23, 3])
    elif model_type == 'S2vnet_150' or 'S2vnet_150_l1':
        model_ft = S2vnet(layers=[3, 8, 36, 3])
    # elif model_type == 'mynet4_150':
    #     model_ft = mynet.mynet4()
    else:
        print('Network type {} not supported!'.format(model_type))
        sys.exit()

    # model_path = path.join(model_folder, '3d_best_Generator_{}.pth'.format(pretrain_model_str))  # 10
    # model_ft.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    # model_ft.cuda()
    # model_ft.eval()
    # model_ft = model_ft.type(dtype)
    model_ft = model_ft.to(device)

    return model_ft

def s2v_registration(iter,mov_slices,fix_img,mat,psf,res_s, res_r,device,num_epochs=1,method="NCC"):
    if iter % 1000 == 0 and iter < 5000:
        return v2s_registration(mov_slices, fix_img, mat, psf, res_s, res_r, device, num_epochs=1, method="NCC")
    else:
        return mat_update_resolution(
        axisangle2mat(mat.view(-1, 6)), res_s, res_r
    )
class Select_Profile(torch.nn.Module):
    def __init__(self,n_samples,device):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.device = device
        self.n_samples = n_samples

    def mat_transform_points(self,
            mat: torch.Tensor, x: torch.Tensor, trans_first=True
    ) -> torch.Tensor:
        # mat (*, 3, 4)
        # x (*, 3)
        R = mat[..., :-1]  # (*, 3, 3)
        T = mat[..., -1:]  # (*, 3, 1)
        x = x[..., None]  # (*, 3, 1)

        # print("R",R.shape)
        # print("T",T.shape)
        # print("x",x.shape)

        if trans_first:
            x = torch.matmul(R, x + T)  # (*, 3)
        else:
            x = torch.matmul(R, x) + T
        return x[..., 0]


    def dof2mat_tensor(self,input_dof):


        rad = tgm.deg2rad(input_dof[:, :3])

        rx = rad[:, 0]
        ry = rad[:, 1]
        rz = rad[:, 2]

        cx, sx, cy, sy, cz, sz = torch.cos(rx), torch.sin(rx), torch.cos(ry), torch.sin(ry), torch.cos(rz), torch.sin(
            rz)

        tmp1 = torch.cat((cy * cz - sx * sy * sz, cy * sz + cz * sx * sy, -cx * sy, input_dof[:, 3]), dim=0).to(self.device)
        tmp2 = torch.cat((-cx * sz, cx * cz, sx, input_dof[:, 4]), dim=0).to(self.device)
        tmp3 = torch.cat((cz * sy + cy * sx * sz, sz * sy - cz * sx * cy, cx * cy, input_dof[:, 5]), dim=0).to(self.device)
        # tmp4 = torch.tensor([[0, 0, 0, 1]], requires_grad=False).to(device)

        tmp1 = tmp1.unsqueeze(0)
        tmp2 = tmp2.unsqueeze(0)
        tmp3 = tmp3.unsqueeze(0)
        return torch.cat((tmp1, tmp2, tmp3), dim=0).unsqueeze(0)

    def forward(self, x,sampleGrid,ax,bound,InvCovScaled):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # print("x",x.shape)
        # print("sampleGrid",sampleGrid.shape)
        # print("ax",ax.shape)
        # print("bound",bound.shape)
        # print("InvCovScaled",InvCovScaled.shape)
        size = [x.shape[-1],x.shape[-2],x.shape[-3]]
        sampleGrid=sampleGrid.view(-1,3)
        # bound = bound.repeat(sampleGrid.shape[0],1,1)
        InvCovScaled = InvCovScaled.repeat(sampleGrid.shape[0],1,1)

        xyz_psf = torch.randn(ax.shape[0], self.n_samples, 3, device=self.device) - 0.5
        bound_psf = xyz_psf * ((bound[:, 1, :] - bound[:, 0, :])/2)[:, None]
        # xyz_psf = torch.randn(ax.shape[0],self.n_samples,1,1,3,device=self.device)




        mat = axisangle2mat(ax.view(-1, 6)).view(ax.shape[:-1] + (3, 4))
        # mat = self.dof2mat_tensor(ax.view(-1, 6)).view(ax.shape[:-1] + (3, 4))
        mat_sampleGrid = self.mat_transform_points(mat, sampleGrid[:, None])
        psf_sampleGrid = mat_sampleGrid + bound_psf


        center = bound_psf

        for i in range(3):
            # mat_sampleGrid[..., i] = 2 * (mat_sampleGrid[..., i] / (size[i] - 1) - 0.5)
            psf_sampleGrid[..., i] = 2 * (psf_sampleGrid[..., i] / (size[i] - 1) - 0.5)
        # print("mat_sampleGrid", mat_sampleGrid.shape)





        shape = psf_sampleGrid.shape[:-1]

        # psf_x = F.grid_sample(x, mat_sampleGrid.view(1, 1, 1, -1, 3), mode='bilinear').view(shape)
        psf_x = F.grid_sample(x, psf_sampleGrid.view(1, 1, 1, -1, 3), mode='bilinear').view(shape)

        temp = torch.matmul(InvCovScaled[:,None], center[...,None])

        weight_psf_x = torch.matmul(center[:,:,None], temp)
        # print("weight_psf_x[0,:]", weight_psf_x[ 0, :])


        weight_psf_x = torch.exp(-0.5*weight_psf_x)

        psf_x = psf_x * weight_psf_x[:,:,0,0]
        # print("psf_x",psf_x.shape)



        y = psf_x.sum(-1)/weight_psf_x[:,:,0,0].sum(-1)
        return y


def train_model(model, slice, fix_img,axisangle,psf,res_s, res_r, optimizer, exp_lr_scheduler,device, num_epochs=1,method="NCC"):

    mse = nn.MSELoss()
    ncc_loss = NCC(device).loss
    criterion_l1 = nn.SmoothL1Loss()
    best_mse = 0
    mat = mat_update_resolution(
        axisangle2mat(axisangle.view(-1, 6)), res_s, res_r
    )


    vol_npy = fix_img.detach().cpu().numpy()

    _,_,z,y,x = vol_npy.shape
    centerX = x // 2
    centerY = y // 2
    centerZ = z // 2


    range_x = [centerX - 64, centerX+ 64]
    range_y = [centerY - 64, centerY+ 64]
    range_z = [centerZ - 64, centerZ+ 64]
    vol_npy = vol_npy[:,:,range_z[0]:range_z[1],range_y[0]:range_y[1],range_x[0]:range_x[1]]

    # vol_npy = np.expand_dims(vol_npy, 0)
    # vol_npy = np.expand_dims(vol_npy, 0)
    # vol_npy = vol_npy[80 - 64:80 + 64, 80 - 64:80 + 64, 80 - 64:80 + 64]


    slice_npy = slice.cpu().numpy()

    slice_shape = slice_npy.shape


    _,z, y, x = slice_shape
    if y < z <= x or y < x <= z:

        slice_npy = np.transpose(slice_npy, (1, 0, 2))

    elif x < y <= z or x < z <= y:
        slice_npy = np.transpose(slice_npy, (2, 1, 0))


    # print("slice_shape:",slice_shape)
    # print("vol_shape:",vol_shape)

    # z, y, x =  slice_npy.shape
    # centerX = x // 2
    # centerY = y // 2
    #
    #
    # range_x = [centerX - 64, centerX + 64]
    # range_y = [centerY - 64, centerY + 64]
    #
    # slice_npy = slice_npy[:, range_y[0]:range_y[1], range_x[0]:range_x[1]]

    slice_shape = slice_npy.shape
    if slice_shape[2] ==160:
        slice_npy = slice_npy[:,:,80-64:80+64,:]
        slice_shape = slice_npy.shape
    if slice_shape[3] == 160:
        slice_npy = slice_npy[:,:,:,80-64:80+64]
        slice_shape = slice_npy.shape


    if slice_shape[2] <=128 and slice_shape[3]<=128:
        pady1 = (128 - slice_shape[1]) // 2 + (128 - slice_shape[1]) % 2
        pady2 = (128 - slice_shape[1]) // 2
        padx1 = (128 - slice_shape[2]) // 2 + (128 - slice_shape[2]) % 2
        padx2 = (128 - slice_shape[2]) // 2

        slice_npy = np.pad(slice_npy, ((0, 0),(0, 0), (pady1, pady2), (padx1, padx2)))
        slice_shape = slice_npy.shape

    elif slice_shape[2] <= 128 and slice_shape[3] >= 128:
        pady1 = (128 - slice_shape[1]) // 2 + (128 - slice_shape[1]) % 2
        pady2 = (128 - slice_shape[1]) // 2

        slice_npy = np.pad(slice_npy, ((0, 0),(0, 0), (pady1, pady2), (0, 0)))
        slice_shape = slice_npy.shape

    elif slice_shape[2] >= 128 and slice_shape[3] <= 128:
        padx1 = (128 - slice_shape[2]) // 2 + (128 - slice_shape[2]) % 2
        padx2 = (128 - slice_shape[2]) // 2

        slice_npy = np.pad(slice_npy, ((0, 0),(0, 0), (0, 0), (padx1, padx2)))
        slice_shape = slice_npy.shape



    slice_npy = slice_npy[:,:,0:128,0:128]

    # slice_npy = np.expand_dims(slice_npy, 0)

    slice_tensor = torch.from_numpy(slice_npy).type(dtype)
    vol_tensor = torch.from_numpy(vol_npy).type(dtype)

    # print('vol_tensor {}'.format(vol_tensor.shape))
    # print('slice_tensor {}'.format(slice_tensor.shape))
    #
    # sys.exit()

    vol_tensor = vol_tensor.to(device)
    slice_tensor = slice_tensor.to(device)



    for epoch in range(num_epochs):

        global current_epoch
        current_epoch = epoch + 1

        model.train()

        # forward
        # track history if only in train
        optimizer.zero_grad()
        ax = model(vol=vol_tensor, slice=slice_tensor)
        ax[...,:3] = tgm.deg2rad(ax[...,:3])
        matax = RigidTransform(ax).matrix()


        pre_slice = slice_acquisition(
            # mat,
            matax,
            fix_img,
            None,
            None,
            psf,
            slice.shape[-2:],
            res_s / res_r,
            False,
            False,
        )

        if method == "NCC":
            loss =ncc_loss(slice, pre_slice.reshape(slice.shape))
        else:
            loss = mse(slice,  pre_slice.reshape(slice.shape))

        loss.backward()
        optimizer.step()
        exp_lr_scheduler.step()



    return mat


def v2s_registration(mov_slices,fix_img,mat,psf,res_s, res_r,device,num_epochs=1,method="NCC"):
    model_type = 'S2vnet_50'
    s2vnet = defineModel(model_type=model_type, device=device)
    for p in s2vnet.parameters():
        torch.nn.init.uniform_(p, a=-1e-4, b=1e-4)
    optimizer = optim.Adam(s2vnet.parameters(), lr=1e-3)
    # exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    pre_dof = train_model(s2vnet,mov_slices,fix_img,mat,psf,res_s, res_r,
                optimizer,
                exp_lr_scheduler,
                device=device,
                num_epochs=num_epochs,
                method=method)
    pre_dof = pre_dof
    del s2vnet
    return pre_dof









