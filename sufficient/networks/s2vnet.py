import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import time
import sys
import torchgeometry as tgm
import os
from copy import copy
import numpy as np
import SimpleITK as sitk

dtype = torch.cuda.FloatTensor

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class S2vnet(nn.Module):
    """ First working model! """

    def __init__(self, layers):
        self.inplanes = 64
        super(S2vnet, self).__init__()
        """ Balance """
        layers = layers
        # layers = [3, 4, 6, 3]  # resnext50
        # layers = [3, 4, 23, 3]  # resnext101
        # layers = [3, 8, 36, 3]  # resnext150
        self.conv1_vol = nn.Conv3d(1, 32, kernel_size=9, stride=(2, 2, 2), padding=(4, 4, 4), bias=False)
        self.conv2_vol = nn.Conv3d(32, 64, kernel_size=7, stride=(2, 1, 1), padding=(3, 3, 3), bias=False)
        # self.conv1_vol = nn.Conv3d(1, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        # self.conv2_vol = nn.Conv3d(32, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        # self.conv3_vol = nn.Conv3d(128, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        # self.conv4_vol = nn.Conv3d(64, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.bn1_vol = nn.BatchNorm3d(32)
        # self.bn2_vol = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        # self.conv_pool = nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))


        self.conv2d_slice = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv1_slice = nn.Conv3d(1, 32, kernel_size=9, stride=(2, 2, 2), padding=(4, 4, 4), bias=False)
        self.conv2_slice = nn.Conv3d(32, 64, kernel_size=7, stride=(2, 1, 1), padding=(3, 3, 3), bias=False)
        # self.conv1_slice = nn.Conv3d(1, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        # self.conv2_slice = nn.Conv3d(32, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        # self.conv3_slice = nn.Conv3d(128, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

        # self.conv2d_slice = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)


        # self.bn1_slice = nn.BatchNorm3d(32)
        # self.bn2_slice = nn.BatchNorm3d(64)

        self.layer1 = self._make_layer(
            ResNeXtBottleneck, 128, layers[0], shortcut_type='B', cardinality=32, stride=2)
        self.layer2 = self._make_layer(
            ResNeXtBottleneck, 256, layers[1], shortcut_type='B', cardinality=32, stride=2)
        self.layer3 = self._make_layer(
            ResNeXtBottleneck, 512, layers[2], shortcut_type='B', cardinality=32, stride=2)
        self.layer4 = self._make_layer(
            ResNeXtBottleneck, 1024, layers[3], shortcut_type='B', cardinality=32, stride=2)

        self.avgpool = nn.AvgPool3d((3, 4, 4), stride=1)
        self.maxpool = nn.MaxPool3d((1, 4, 7), stride=1)

        # self.fc1 = nn.Linear(2048, 512)
        self.fc1 = nn.Linear(2048, 512)
        # self.fc1 = nn.Linear(8192, 512)
        self.fc2 = nn.Linear(512, 6)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))
                # downsample = nn.Sequential(
                #     nn.Conv3d(
                #         self.inplanes,
                #         planes * block.expansion,
                #         kernel_size=1,
                #         stride=stride,
                #         bias=False),)

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def volBranch(self, vol):
        # print('\n********* Vol *********')
        # print('vol {}'.format(vol.shape))
        vol = self.conv1_vol(vol)
        vol = self.bn1_vol(vol)
        vol = self.relu(vol)
        # print('conv1 {}'.format(vol.shape))

        vol = self.conv2_vol(vol)
        vol = self.relu(vol)
        # print('conv2 {}'.format(vol.shape))

        return vol

    def sliceBranch(self, slice):
        # print('\n********* slice *********')
        # print('slice {}'.format(slice.shape))
        # slice = slice.squeeze(1)
        # print('squeeze {}'.format(slice.shape))

        slice = self.conv2d_slice(slice)
        # print('conv2d_slice {}'.format(slice.shape))

        slice = slice.unsqueeze(1)
        # print('unsqueeze {}'.format(slice.shape))

        slice = self.conv1_slice(slice)
        slice = self.bn1_vol(slice)
        # slice = self.bn1_slice(slice)
        slice = self.relu(slice)
        # print('conv1 {}'.format(slice.shape))

        slice = self.conv2_slice(slice)
        slice = self.relu(slice)
        # print('conv2 {}\n'.format(slice.shape))

        return slice

    def forward(self, vol, slice,):
        # input_vol = vol.clone()

        input_vol = vol
        n = slice.shape[0]

        slice = slice[0:1,:,:,:]

        show_size = False
        # show_size = True
        if show_size and input_vol is not None:
            vol = self.volBranch(vol)
            slice = self.sliceBranch(slice)

            x = torch.cat((vol, slice), 2)
            print('cat {}'.format(x.shape))
            # sys.exit()

            x = self.layer1(x)
            print('layer1 {}'.format(x.shape))

            x = self.layer2(x)
            print('layer2 {}'.format(x.shape))

            x = self.layer3(x)
            print('layer3 {}'.format(x.shape))

            x = self.layer4(x)
            print('layer4 {}'.format(x.shape))

            x = self.avgpool(x)
            print('avgpool {}'.format(x.shape))


            # x = x.unsqueeze(0)

            # x = x.view(x.size(1), -1)
            x = x.view(x.size(0), -1)
            print('view {}'.format(x.shape))

            x = self.fc1(x)
            print('fc1 {}'.format(x.shape))
            # dof_out = x.clone()
            x = self.relu(x)
            x = self.fc2(x)

            # mat = dof2mat_tensor(input_dof=x, device=device)
            # print('mat {}'.format(mat.shape))
            #
            # print('input_vol {}'.format(input_vol.shape))
            # grid = myAffineGrid(input_tensor=input_vol, input_mat=mat,
            #                            input_spacing=spacing, device=device)
            # # grid = grid.to(device)
            # print('grid {}'.format(grid.shape))
            # vol_resampled = F.grid_sample(input_vol, grid, align_corners=True)
            # print('resample {}'.format(vol_resampled.shape))
            # print('mat_out {}'.format(x.shape))

            sys.exit()
        else:
            vol = self.volBranch(vol)
            slice = self.sliceBranch(slice)


            x = torch.cat((vol, slice), 2)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            # print('layer4 {}'.format(x.shape))

            x = self.avgpool(x)
            # print('avgpool {}'.format(x.shape))
            # x = self.maxpool(x)
            x = x.view(x.size(0), -1)
            # print('view {}'.format(x.shape))
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = x.repeat(n,1)



        return x
        # return vol_resampled, x

class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)

        # if stride == 2:
        #     stride = (1, 2, 2)

        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        # print('stride {}'.format(stride))
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def dof2mat_inverse(input_dof,identity=False):
    if identity:
        tmp1 = torch.tensor([[1, 0, 0, 0]], requires_grad=False).type(dtype)
        tmp2 = torch.tensor([[0, 1, 0, 0]], requires_grad=False).type(dtype)
        tmp3 = torch.tensor([[0, 0, 1, 0]], requires_grad=False).type(dtype)
        tmp4 = torch.tensor([[0, 0, 0, 1]], requires_grad=False).type(dtype)

        return torch.cat((tmp1, tmp2, tmp3, tmp4), dim=0).unsqueeze(0).detach().cpu()
    rad = tgm.deg2rad(input_dof[:, 3:])
    # print("rad",rad)
    # return rad

    rx = float(rad[:, 0])
    ry = float(rad[:, 1])
    rz = float(rad[:, 2])

    x = float(input_dof[:, 0])
    y = float(input_dof[:, 1])
    z = float(input_dof[:, 2])

    motion_sitk = sitk.Euler3DTransform()
    motion_sitk.SetRotation(rx, ry, rz)
    # motion_sitk.SetCenter(center_3D)
    motion_sitk.SetTranslation((x,y,z))


    motion_sitk_inverse = sitk.Euler3DTransform(motion_sitk.GetInverse())
    matrix_inverse = motion_sitk_inverse.GetMatrix()
    translation_inverse = motion_sitk_inverse.GetTranslation()
    mat_inv = torch.tensor([[[matrix_inverse[0],matrix_inverse[3],matrix_inverse[6],translation_inverse[0]],
                            [matrix_inverse[1],matrix_inverse[4],matrix_inverse[7],translation_inverse[1]],
                            [matrix_inverse[2],matrix_inverse[5],matrix_inverse[8],translation_inverse[2]],
                            [0,0,0,1]
                            ]],requires_grad=False).type(dtype).detach().cpu()

    return mat_inv


def dof2mat_tensor(input_dof, device, identity=False):
    if identity:
        tmp1 = torch.tensor([[1, 0, 0, 0]], requires_grad=False).type(dtype).to(device)
        tmp2 = torch.tensor([[0, 1, 0, 0]], requires_grad=False).type(dtype).to(device)
        tmp3 = torch.tensor([[0, 0, 1, 0]], requires_grad=False).type(dtype).to(device)
        tmp4 = torch.tensor([[0, 0, 0, 1]], requires_grad=False).type(dtype).to(device)

        return torch.cat((tmp1, tmp2, tmp3, tmp4), dim=0).unsqueeze(0)

    rad = tgm.deg2rad(input_dof[:, 3:])
    # print("rad",rad)
    # return rad



    rx = rad[:, 0]
    ry = rad[:, 1]
    rz = rad[:, 2]

    # rx = input_dof[:, 3]
    # ry = input_dof[:, 4]
    # rz = input_dof[:, 5]

    cx, sx, cy, sy, cz, sz = torch.cos(rx), torch.sin(rx), torch.cos(ry), torch.sin(ry), torch.cos(rz), torch.sin(rz)

    # tmp1 = torch.cat((cy * cz - sx*sy*sz, -cx*sz, cz*sy+cy*sx*sz, input_dof[:, 0]), dim=0)
    #
    # tmp2 = torch.cat((cy*sz+cz*sx*sy, cx*cz, sz*sy-cz*sx*cy, input_dof[:, 1]), dim=0)
    # tmp3 = torch.cat((-cx*sy, sx, cx*cy, input_dof[:, 2]), dim=0)


    tmp1 = torch.cat((cy * cz - sx * sy * sz, cy*sz+cz*sx*sy, -cx*sy, input_dof[:, 0]), dim=0)
    tmp2 = torch.cat((-cx*sz, cx*cz, sx, input_dof[:, 1]), dim=0)
    tmp3 = torch.cat((cz*sy+cy*sx*sz, sz*sy-cz*sx*cy, cx*cy, input_dof[:, 2]), dim=0)
    tmp4 = torch.tensor([[0, 0,0,1]],requires_grad=False).to(device)



    tmp1 = tmp1.unsqueeze(0)
    tmp2 = tmp2.unsqueeze(0)
    tmp3 = tmp3.unsqueeze(0)
    return torch.cat((tmp1, tmp2, tmp3, tmp4), dim=0).unsqueeze(0)
    # return torch.cat((tmp1, tmp2, tmp3, torch.zeros_like(tmp1)), dim=0).unsqueeze(0)
    # tmp1 =  torch.cat((torch.tensor([1]).to(device),torch.tensor([0]).to(device),torch.tensor([0]).to(device)), dim=0).unsqueeze(0)
    # tmp2 = torch.cat((torch.tensor([0]).to(device),cx,-sx), dim=0).unsqueeze(0)
    # tmp3 = torch.cat((torch.tensor([0]).to(device),sx, cx), dim=0).unsqueeze(0)
    # Rx = torch.cat((tmp1, tmp2, tmp3), dim=0)
    #
    # tmp1 = torch.cat((cy,torch.tensor([0]).to(device),sy), dim=0).unsqueeze(0)
    # tmp2 = torch.cat((torch.tensor([0]).to(device),torch.tensor([1]).to(device),torch.tensor([0]).to(device)), dim=0).unsqueeze(0)
    # tmp3 = torch.cat((-sy, torch.tensor([0]).to(device), cy), dim=0).unsqueeze(0)
    # Ry = torch.cat((tmp1, tmp2, tmp3), dim=0)
    #
    # tmp1 = torch.cat((cz,-sz,torch.tensor([0]).to(device)), dim=0).unsqueeze(0)
    # tmp2 = torch.cat((sz, cz, torch.tensor([0]).to(device)), dim=0).unsqueeze(0)
    # tmp3 = torch.cat((torch.tensor([0]).to(device),torch.tensor([0]).to(device),torch.tensor([1]).to(device)), dim=0).unsqueeze(0)
    # Rz = torch.cat((tmp1, tmp2, tmp3), dim=0)
    #
    #
    #
    #
    # Rzxy = torch.matmul(torch.matmul(Rz, Rx),Ry)
    # Rzxy = Rzxy.view(-1,9)
    #
    # tmp1 = torch.cat((Rzxy[:,0], Rzxy[:,1],Rzxy[:,2],input_dof[:, 0]), dim=0)
    # tmp2 = torch.cat((Rzxy[:,3], Rzxy[:,4],Rzxy[:,5], input_dof[:, 1]), dim=0)
    # tmp3 = torch.cat((Rzxy[:,6], Rzxy[:,7],Rzxy[:,8], input_dof[:, 2]), dim=0)
    # tmp1 = tmp1.unsqueeze(0)
    # tmp2 = tmp2.unsqueeze(0)
    # tmp3 = tmp3.unsqueeze(0)
    #
    # print(Rx)
    # Rzxy = torch.cat((tmp1, tmp2, tmp3, torch.zeros_like(tmp1)), dim=0).unsqueeze(0)
    # print(Rzxy)
    # sys.exit()
    # return torch.cat((tmp1, tmp2, tmp3, torch.zeros_like(tmp1)), dim=0).unsqueeze(0)

    # M = torch.zeros((input_dof.shape[0], 4, 4))
    #
    # if device:
    #     M = M.to(device)
    #     M.requires_grad = True




    # print("ai",ai)

    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    # H = torch.tensor([[
    #     [cj * ck, sj * sc - cs, sj * cc + ss, input_dof[:, 0]],
    #     [cj * sk, sj * ss + cc, sj * cs - sc, input_dof[:, 1]],
    #     [-sj, cj * si, cj * ci, input_dof[:, 2]],
    #     [0, 0, 0, 0]
    # ]], requires_grad=True).to(device)
    # print("H",H)
    # print("H.shape",H.shape)

    tmp1 = torch.cat((cj * ck, sj * sc - cs, sj * cc + ss, input_dof[:, 0]),dim=0)

    tmp2 = torch.cat((cj * sk, sj * ss + cc, sj * cs - sc, input_dof[:, 1]),dim=0)
    tmp3 = torch.cat((-sj, cj * si, cj * ci, input_dof[:, 2]),dim=0)
    tmp1 = tmp1.unsqueeze(0)
    tmp2 = tmp2.unsqueeze(0)
    tmp3 = tmp3.unsqueeze(0)
    # print(torch.cat((tmp1,tmp2,tmp3,torch.zeros_like(tmp1)),dim=0).unsqueeze(0))
    # print(torch.cat((tmp1,tmp2,tmp3,torch.zeros_like(tmp1)),dim=0).unsqueeze(0).shape)

    return torch.cat((tmp1,tmp2,tmp3,torch.zeros_like(tmp1)),dim=0).unsqueeze(0)



    # M = torch.zeros((input_dof.shape[0], 4, 4))

    # if device:
    #     M = M.to(device)
    #     M.requires_grad = True
    #
    #
    #
    # with torch.no_grad():
    #
    #     M[:, 0, 0] = cj * ck
    #     M[:, 0, 1] = sj * sc - cs
    #     M[:, 0, 2] = sj * cc + ss
    #     M[:, 1, 0] = cj * sk
    #     M[:, 1, 1] = sj * ss + cc
    #     M[:, 1, 2] = sj * cs - sc
    #     M[:, 2, 0] = -sj
    #     M[:, 2, 1] = cj * si
    #     M[:, 2, 2] = cj * ci
    #     M[:, :3, 3] = input_dof[:, :3]

    # print("M",M)
    # H = torch.tensor([[
    #     [cj * ck, sj * sc - cs, sj * cc + ss, input_dof[:, 0]],
    #     [cj * sk, sj * ss + cc, sj * cs - sc, input_dof[:, 1]],
    #     [-sj, cj * si, cj * ci, input_dof[:, 2]],
    #     [0, 0, 0, 0]
    # ]], requires_grad=True).to(device)
    # print(H)
    # sys.exit()

    return torch.tensor([[
        [cj * ck, sj * sc - cs, sj * cc + ss, input_dof[:, 0]],
        [cj * sk, sj * ss + cc, sj * cs - sc, input_dof[:, 1]],
        [-sj, cj * si, cj * ci, input_dof[:, 2]],
        [0, 0, 0, 0]
    ]], requires_grad=True).to(device)

    # M = torch.zeros((input_dof.shape[0], 4, 4))






    # print('out_mat {}\n{}'.format(M.shape, M))
    # sys.exit()
    # return M

def myAffineGrid(input_tensor, input_mat, input_spacing=[1, 1, 1], device=None):
    # print('input_tensor shape {}'.format(input_tensor.shape))
    # print('input_mat shape {}'.format(input_mat.shape))
    # sys.exit()

    input_spacing = np.asarray(input_spacing)
    image_size = np.asarray([input_tensor.shape[4], input_tensor.shape[3], input_tensor.shape[2]])
    print("image_size",image_size)
    image_phy_size = (image_size - 1) * input_spacing
    print("image_phy_size",image_phy_size)
    # image_phy_size = [input_tensor.shape[4], input_tensor.shape[3], input_tensor.shape[2]]
    grid_size = input_tensor.shape

    # generate grid of input image (i.e., the coordinate of the each pixel in the input image. The center point of the input image volume is assigned as (0, 0, 0).)
    grid_x_1d = torch.linspace(-0.5 * image_phy_size[0], 0.5 * image_phy_size[0], steps=grid_size[4])
    grid_y_1d = torch.linspace(-0.5 * image_phy_size[1], 0.5 * image_phy_size[1], steps=grid_size[3])
    grid_z_1d = torch.linspace(-0.5 * image_phy_size[2], 0.5 * image_phy_size[2], steps=grid_size[2])
    grid_z, grid_y, grid_x = torch.meshgrid(grid_z_1d, grid_y_1d, grid_x_1d)
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)
    grid_z = grid_z.unsqueeze(0)
    print("grid_x",grid_x.shape)
    print("grid_y",grid_y.shape)
    print("grid_z",grid_z.shape)
    origin_grid = torch.cat([grid_x, grid_y, grid_z, torch.ones_like(grid_x)], dim=0)
    origin_grid = origin_grid.view(4, -1)
    if device:
        origin_grid = origin_grid.to(device)
        origin_grid.requires_grad = True

    print('origin_grid {}'.format(origin_grid.shape))
    # compute the rasample grid through matrix multiplication
    # print('t_mat {}, origin_grid {}'.format(t_mat.shape, origin_grid.shape))
    # t_mat = input_mat
    # t_mat = torch.tensor(t_mat)
    # t_mat = t_mat.float()
    # t_mat.requires_grad = True

    # t_mat = t_mat.squeeze()
    # origin_grid = origin_grid.unsqueeze(0)
    # print('t_mat shape {}'.format(t_mat.shape))
    # print('origin_grid shape {}'.format(origin_grid.shape))
    # resample_grid = torch.matmul(t_mat, origin_grid)[0:3, :]
    resample_grid = torch.matmul(input_mat, origin_grid)[:, 0:3, :]
    print('resample_grid {}'.format(resample_grid.shape))
    # print(resample_grid)

    # convert the resample grid coordinate from physical coordinate system to a range of [-1, 1] (which is required by the PyTorch interface 'grid_sample').
    resample_grid[:, 0, :] = (resample_grid[:, 0, :] + 0.5 * image_phy_size[0]) / image_phy_size[0] * 2 - 1
    resample_grid[:, 1, :] = (resample_grid[:, 1, :] + 0.5 * image_phy_size[1]) / image_phy_size[1] * 2 - 1
    resample_grid[:, 2, :] = (resample_grid[:, 2, :] + 0.5 * image_phy_size[2]) / image_phy_size[2] * 2 - 1
    print('resample_grid1 {}'.format(resample_grid.shape))
    resample_grid = resample_grid.permute(0,2,1).contiguous()
    resample_grid = resample_grid.reshape(grid_size[0], grid_size[2], grid_size[3], grid_size[4], 3)
    # resample_grid = resample_grid.unsqueeze(1)
    print('resample_grid2 {}'.format(resample_grid.shape))
    # print(resample_grid)
    sys.exit()


    return resample_grid

class SpatialTransformer(nn.Module):
    def __init__(self, input_tensor,input_spacing=[1, 1, 1] ,mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        input_spacing = np.asarray(input_spacing)
        image_size = np.asarray([input_tensor.shape[4], input_tensor.shape[3], input_tensor.shape[2]])
        image_phy_size = (image_size - 1) * input_spacing
        # image_phy_size = [input_tensor.shape[4], input_tensor.shape[3], input_tensor.shape[2]]
        grid_size = input_tensor.shape

        # generate grid of input image (i.e., the coordinate of the each pixel in the input image. The center point of the input image volume is assigned as (0, 0, 0).)
        grid_x_1d = torch.linspace(-0.5 * image_phy_size[0], 0.5 * image_phy_size[0], steps=grid_size[4])
        grid_y_1d = torch.linspace(-0.5 * image_phy_size[1], 0.5 * image_phy_size[1], steps=grid_size[3])
        grid_z_1d = torch.linspace(-0.5 * image_phy_size[2], 0.5 * image_phy_size[2], steps=grid_size[2])
        grid_z, grid_y, grid_x = torch.meshgrid(grid_z_1d, grid_y_1d, grid_x_1d)
        grid_x = grid_x.unsqueeze(0)
        grid_y = grid_y.unsqueeze(0)
        grid_z = grid_z.unsqueeze(0)
        grid= torch.cat([grid_x, grid_y, grid_z, torch.ones_like(grid_x)], dim=0)
        grid = grid.view(4, -1)
        grid = grid.type(torch.FloatTensor)
        grid.requires_grad = True
        self.register_buffer('grid', grid)
        self.image_phy_size = image_phy_size
        self.grid_size = grid_size

        self.mode = mode

    def forward(self, input_vol, mat):








        resample_grid = torch.matmul(mat, self.grid)[:, 0:3, :]


        # print(resample_grid)
        # print(resample_grid.grad)
        resample_grid[:, 0, :] = (resample_grid[:, 0, :] + 0.5 * self.image_phy_size[0]) / self.image_phy_size[0] * 2 - 1
        resample_grid[:, 1, :] = (resample_grid[:, 1, :] + 0.5 * self.image_phy_size[1]) / self.image_phy_size[1] * 2 - 1
        resample_grid[:, 2, :] = (resample_grid[:, 2, :] + 0.5 * self.image_phy_size[2]) / self.image_phy_size[2] * 2 - 1
        resample_grid = resample_grid.permute(0, 2, 1).contiguous()
        resample_grid = resample_grid.reshape(mat.shape[0], self.grid_size[2], self.grid_size[3], self.grid_size[4], 3)
        # print(resample_grid)
        # print(resample_grid)

        return F.grid_sample(input_vol, resample_grid, mode=self.mode)




        # new_locs = self.grid + flow
        # shape = flow.shape[2:]
        #
        # # Need to normalize grid values to [-1, 1] for resampler
        # for i in range(len(shape)):
        #     new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        #
        # if len(shape) == 2:
        #     new_locs = new_locs.permute(0, 2, 3, 1)
        #     new_locs = new_locs[..., [1, 0]]
        # elif len(shape) == 3:
        #     new_locs = new_locs.permute(0, 2, 3, 4, 1)
        #     new_locs = new_locs[..., [2, 1, 0]]
        #
        # return F.grid_sample(src, new_locs, mode=self.mode)

