from __future__ import print_function

#%matplotlib notebook

import os
import argparse
import warnings
warnings.filterwarnings('ignore')
import torch
torch.manual_seed(4399)
from sufficient.networks.decoder import Deep_Decoder

import numpy as np
import torch.optim
from torch.autograd import Variable

import SimpleITK as sitk
# import ray
import pysitk.simple_itk_helper as sitkh
import itk
import math

import torch.nn.functional as F

import torch.utils.data
import torch.nn as nn
import torchgeometry as tgm
import pysitk.python_helper as ph



from typing import Dict, Any, TYPE_CHECKING
from sufficient.transform import RigidTransform, axisangle2mat,mat_update_resolution
from sufficient.slice_acquisition import slice_acquisition
from sufficient.util.tools import load_stack, Volume, get_PSF,ssim,l1_loss,l2_loss
from sufficient.util.s2v_tool import s2v_registration



GPU = True
# if GPU is True:
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# dtype = torch.cuda.FloatTensor
dtype = torch.float32


GAUSSIAN_FWHM = 1 / (2 * math.sqrt(2 * math.log(2)))
SINC_FWHM = 1.206709128803223 * GAUSSIAN_FWHM
ctype = torch.FloatTensor

grads = {}



def np_to_var(img_np, dtype=torch.cuda.FloatTensor):
    '''Converts image in numpy.array to torch.Variable.

    From C x W x H [0..1] to  1 x C x W x H [0..1]
    '''
    return Variable(np_to_tensor(img_np)[None, :])

def np_to_tensor(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL).astype(np.float32)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    # return ar.astype(np.float32) / 255.

    return (ar.astype(np.float32) - np.min(ar))/ (np.max(ar)-np.min(ar))

def dof2mat_tensor(input_dof, identity=False):


    rad = tgm.deg2rad(input_dof[:, :3])
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


    tmp1 = torch.cat((cy * cz - sx * sy * sz, cy*sz+cz*sx*sy, -cx*sy, input_dof[:, 3]), dim=0).to(args.device)
    tmp2 = torch.cat((-cx*sz, cx*cz, sx, input_dof[:, 4]), dim=0).to(args.device)
    tmp3 = torch.cat((cz*sy+cy*sx*sz, sz*sy-cz*sx*cy, cx*cy, input_dof[:, 5]), dim=0).to(args.device)
    # tmp4 = torch.tensor([[0, 0,0,1]],requires_grad=False)



    tmp1 = tmp1.unsqueeze(0)
    tmp2 = tmp2.unsqueeze(0)
    tmp3 = tmp3.unsqueeze(0)
    # return torch.cat((tmp1, tmp2, tmp3, tmp4), dim=0).unsqueeze(0)
    return torch.cat((tmp1, tmp2, tmp3), dim=0).unsqueeze(0)


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


    def forward(self, x,sampleGrid,ax,bound,InvCovScaled,psf_sigma):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        size = [x.shape[-1],x.shape[-2],x.shape[-3]]

        xyz_psf = torch.randn(ax.shape[0],self.n_samples,3,device=self.device)
        scaled_xyz_psf = 2 * torch.sigmoid(xyz_psf) - 1
        # xyz_psf = xyz_psf.normal_(mean=0,std=1)
        # xyz_psf = torch.rand(ax.shape[0],self.n_samples,3,device=self.device) - 0.5
        bound_psf = scaled_xyz_psf * ((bound[:, 1, :] - bound[:, 0, :])/2)[:, None]
        # bound_psf = xyz_psf * ((bound[:, 1, :] - bound[:, 0, :])/2)[:, None]



        mat = axisangle2mat(ax.view(-1, 6)).view(ax.shape[:-1] + (3, 4))


        mat_sampleGrid = self.mat_transform_points(mat, sampleGrid[:,None])

        psf_sampleGrid = mat_sampleGrid+ bound_psf
        # mat_sampleGrid = mat_sampleGrid + bound

        # with torch.no_grad():
        #     center = self.mat_transform_points(mat,bound_psf)
        center = bound_psf
        # # center = int_mat_sampleGrid - sampleGrid
        # # center = bound


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
        #

        weight_psf_x = torch.exp(-0.5*weight_psf_x)
        psf_x = psf_x * weight_psf_x[:,:,0,0]
        y = psf_x.sum(-1)/weight_psf_x[:,:,0,0].sum(-1)
        return y





def get_covariance_full_3d(reconstruction_itk,slice_itk,slice_spacing):

    reconstruction_direction_sitk = sitkh.get_sitk_from_itk_direction(
        reconstruction_itk.GetDirection())
    slice_direction_sitk = sitkh.get_sitk_from_itk_direction(
        slice_itk.GetDirection())


    # Compute rotation matrix to express the PSF in the coordinate system
    # of the reconstruction space

    dim = np.sqrt(len(slice_direction_sitk)).astype('int')
    direction_matrix_reconstruction = np.array(
        reconstruction_direction_sitk).reshape(dim, dim)
    direction_matrix_slice = np.array(
        slice_direction_sitk).reshape(dim, dim)

    U = direction_matrix_reconstruction.transpose().dot(
        direction_matrix_slice)

    # Get axis-aligned PSF
    # Compute Gaussian to approximate in-plane PSF:
    sigma_x2 = (1.2 * slice_spacing[0]) ** 2 / (8 * np.log(2))
    sigma_y2 = (1.2 * slice_spacing[1]) ** 2 / (8 * np.log(2))

    # Compute Gaussian to approximate through-plane PSF:
    sigma_z2 = slice_spacing[2] ** 2 / (8 * np.log(2))

    cov =  np.diag([sigma_x2, sigma_y2, sigma_z2])

    # Return Gaussian blurring variance covariance matrix of slice in
    # reconstruction space coordinates
    cov = U.dot(cov).dot(U.transpose())

    # cov = get_covariance_matrix_in_reconstruction_space_sitk(
    #     reconstruction_direction_sitk=reconstruction_direction_sitk,
    #     slice_direction_sitk=slice_direction_sitk,
    #     slice_spacing=slice_spacing)

    return cov




def get_stacks_b(stacks,HR_volume):
    b_stacks = []
    mask_stacks = []

    image_type = itk.Image.D3
    itk2np = itk.PyBuffer[image_type]

    stacks_scale = np.max(np.array([float(sitk.GetArrayFromImage(stacks[i].sitk).max()) for i in range(len(stacks)) ]))
    # stacks_scale = np.min(np.array([float(sitk.GetArrayFromImage(stacks[i].sitk).max()) for i in range(len(stacks)) ]))
    # stacks_scale =float(sitk.GetArrayFromImage(stacks[target_stack_index].sitk).max())
    # HR_scale = float(sitk.GetArrayFromImage(HR_volume.sitk).max())

    for i, stack in enumerate(stacks):
        b_stack = []
        mask_stack = []
        slices = stack.get_slices()

        # stack_scale = float(sitk.GetArrayFromImage(stack.sitk).max())

        for j, slice_j in enumerate(slices):

            slice_sitk = slice_j.sitk
            slice_mask = sitk.GetArrayFromImage(slice_j.sitk)
            slice_mask[slice_mask  > 0] = 1
            slice_nda = sitk.GetArrayFromImage(
                slice_sitk)

            # slice_scale = float(sitk.GetArrayFromImage(slice_j.sitk).max())
            # if i == 0:
            #     slice_nda -= 150
            slice_nda = slice_nda / stacks_scale
            # slice_nda = slice_nda / slice_scale
            # slice_nda = slice_nda / HR_scale
            # slice_nda = slice_nda / stack_scale

            # b_stack.append(np.clip(slice_nda,0,1.2))
            # if i == 0:
            #     slice_nda = np.fliplr(slice_nda).copy()
            #     slice_mask = np.fliplr(slice_mask).copy()
            b_stack.append(slice_nda)
            mask_stack.append(slice_mask)
        b_stacks.append(b_stack)
        mask_stacks.append(mask_stack)

    return b_stacks,mask_stacks





class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1.0):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        d_x = x.size()[4]
        count_h = self._tensor_size(x[:,:,1:,:,:])
        count_w = self._tensor_size(x[:,:,:,1:,:])
        count_d = self._tensor_size(x[:,:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:,:]-x[:,:,:h_x-1,:,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:,:]-x[:,:,:,:w_x-1,:]),2).sum()
        d_tv = torch.pow((x[:,:,:,:,1:]-x[:,:,:,:,:d_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w+d_tv/count_d)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]*t.size()[4]




def get_cov(reconstruction_itk,slice):
    in_plane_res = slice.get_inplane_resolution()
    slice_thickness = slice.get_slice_thickness()
    slice_spacing = np.array([in_plane_res, in_plane_res, slice_thickness])
    cov = get_covariance_full_3d(reconstruction_itk, slice.itk, slice_spacing)
    return cov

def get_sigma(cov):
    m_Sigma = []
    for d in range(3):
        m_Sigma.append(math.sqrt(cov[d * 3 + d]))

    return m_Sigma

def ComputeBoundingBox(fix_img,slice):
    m_Alpha = 3
    input = fix_img.itk
    spacing = input.GetSpacing()
    size = input.GetBufferedRegion().GetSize()
    cov = get_cov(fix_img.itk,slice)
    m_Sigma = get_sigma(cov.flatten())

    m_BoundingBoxStart = []
    m_BoundingBoxEnd = []
    m_CutoffDistance = []
    for d in range(3):
        m_BoundingBoxStart.append(-0.5)
        m_BoundingBoxEnd.append(float(size[d])-0.5)
        m_CutoffDistance.append(m_Sigma[d] * m_Alpha / spacing[d])

    return m_BoundingBoxStart,m_BoundingBoxEnd,m_CutoffDistance

def getInvCovScaled(fix_img,slice):
    input = fix_img.itk
    ImageDimension = 3
    spacing = input.GetSpacing()
    cov = get_cov(fix_img.itk, slice).flatten()
    m_Covariance = cov
    S = np.zeros((ImageDimension, ImageDimension))
    covariance = np.zeros((ImageDimension, ImageDimension))
    for d in range(ImageDimension):
        S[d, d] = spacing[d]
    for i in range(ImageDimension):
        for j in range(ImageDimension):
            covariance[i, j] = m_Covariance[i * ImageDimension + j]
    InvCov = np.linalg.inv(covariance)
    InvCovScaled = S * InvCov * S
    return InvCovScaled

def ComputeExponentialFunction(coor,center,InvCovScaled):
    # W = np.zeros((coor[1]-coor[0],coor[3]-coor[2],coor[5]-coor[4]))
    W = np.zeros((coor[5]-coor[4],coor[3]-coor[2],coor[1]-coor[0]))
    for i,z in enumerate(range(coor[4],coor[5])):
        for j,y in enumerate(range(coor[2],coor[3])):
            for k,x in enumerate(range(coor[0],coor[1])):
                diff = np.array([x,y,z]) - center
                # print("diff",diff)
                tmp = np.dot(InvCovScaled, diff.T)
                # print("tmp",tmp)
                result = np.dot(diff, tmp)
                # print("result",result)
                w = np.exp(-0.5 * result)
                # W[k,j,i] = w
                W[i,j,k] = w
                # W[i,j,k] = 1
    return W

def resolution2sigma(rx, ry=None, rz=None, isotropic=False):
    if isotropic:
        fx = fy = fz = GAUSSIAN_FWHM
    else:
        fx = fy = SINC_FWHM
        fz = GAUSSIAN_FWHM
    assert not ((ry is None) ^ (rz is None))

    return torch.tensor([fx * rx, fy * ry, fz * rz]).resize(1,3)

def generate_filter_volume(fix_img, slice, input_tensor):
    m_BoundingBoxStart, m_BoundingBoxEnd, m_CutoffDistance = ComputeBoundingBox(fix_img, slice)
    n_samples = 128
    fix_itk = fix_img.itk
    slice_itk = slice.itk
    fix_sitk = fix_img.sitk
    slice_sitk = slice.sitk
    x, y, z = slice_sitk.GetSize()
    resolution = slice_sitk.GetSpacing()
    grid_size = np.asarray([z, y, x])
    vectors = [torch.arange(0, s) for s in grid_size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)  # y, x, z
    grid = torch.unsqueeze(grid, 0)  # add batch
    grid = grid.type(torch.FloatTensor)

    # center
    outputPhy = slice_itk.TransformIndexToPhysicalPoint([int(x / 2), int(y / 2), 0])
    inputCIndex = np.array(fix_itk.TransformPhysicalPointToContinuousIndex(outputPhy)).tolist()
    bound = torch.zeros(2, 3)
    for d in range(3):
        boundingBoxSize = m_BoundingBoxEnd[d] - m_BoundingBoxStart[d] + 0.5
        # print(boundingBoxSize)
        begin = np.max([0, inputCIndex[d] - m_BoundingBoxStart[d] - m_CutoffDistance[d]]) - inputCIndex[d]
        end = np.min([boundingBoxSize, inputCIndex[d] - m_BoundingBoxStart[d] + m_CutoffDistance[d]]) - inputCIndex[d]

        # if (entireInputRegion.IsInside(inputCIndex)):
        bound[0, d] = begin
        bound[1, d] = end
        # bound.append(begin)
        # bound.append(end)
    # print("m_BoundingBoxStart",m_BoundingBoxStart)
    # print("m_BoundingBoxEnd",m_BoundingBoxEnd)
    # print("m_CutoffDistance",m_CutoffDistance)
    # print("bound",bound)
    InvCovScaled = getInvCovScaled(fix_img, slice)
    resample_grid = grid.permute(2, 0, 3, 4, 1)  # z,batch,y,x,coor

    return resample_grid, bound, InvCovScaled



def generate_filter(fix_img, slice, input_tensor):
    m_BoundingBoxStart, m_BoundingBoxEnd, m_CutoffDistance = ComputeBoundingBox(fix_img, slice)
    n_samples = 128
    fix_itk = fix_img.itk
    slice_itk = slice.itk
    fix_sitk = fix_img.sitk
    slice_sitk = slice.sitk
    x, y, z = slice_sitk.GetSize()
    resolution = slice_sitk.GetSpacing()
    entireInputRegion = fix_itk.GetBufferedRegion()

    fix_size = np.asarray([input_tensor.shape[4], input_tensor.shape[3], input_tensor.shape[2]])
    grid_size = np.asarray([z, y, x])
    vectors = [torch.arange(0, s) for s in grid_size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)  # y, x, z
    grid = torch.unsqueeze(grid, 0)  # add batch
    grid = grid.type(torch.FloatTensor)

    # center
    outputPhy = slice_itk.TransformIndexToPhysicalPoint([int(x / 2), int(y / 2), 0])
    inputCIndex = np.array(fix_itk.TransformPhysicalPointToContinuousIndex(outputPhy)).tolist()
    bound = torch.zeros(2,3)
    for d in range(3):
        boundingBoxSize = m_BoundingBoxEnd[d] - m_BoundingBoxStart[d] + 0.5
        # print(boundingBoxSize)
        # begin = int(
        #     np.max([0, int(math.floor(inputCIndex[d] - m_BoundingBoxStart[d] - m_CutoffDistance[d]))]))
        # end = int(np.min([boundingBoxSize,
        #                   int(math.ceil(inputCIndex[d] - m_BoundingBoxStart[d] + m_CutoffDistance[d]))]))

        begin = np.max([0, inputCIndex[d] - m_BoundingBoxStart[d] - m_CutoffDistance[d]]) - inputCIndex[d]
        end = np.min([boundingBoxSize, inputCIndex[d] - m_BoundingBoxStart[d] + m_CutoffDistance[d]]) - inputCIndex[d]
        # if (entireInputRegion.IsInside(inputCIndex)):
        bound[0,d] = begin
        bound[1,d] = end
        # bound.append(begin)
        # bound.append(end)


    InvCovScaled = getInvCovScaled(fix_img, slice)
    # W = ComputeExponentialFunction(bound, inputCIndex, InvCovScaled)

    # sys.exit()

    for i in range(x):
        for j in range(y):
            outputPhy = slice_itk.TransformIndexToPhysicalPoint([i, j, 0])
            inputCIndex_double = torch.from_numpy(np.array(fix_itk.TransformPhysicalPointToContinuousIndex(outputPhy)))
            grid[:, 0:3, 0, j, i] = inputCIndex_double


    resample_grid = grid.permute(2, 0, 3, 4, 1) #z,batch,y,x,coor


    return resample_grid,bound,InvCovScaled


def get_cropped_sitk(HR_volume):
    x,y,z = HR_volume.sitk.GetSize()
    centerX = x // 2
    centerY = y // 2
    centerZ = z // 2


    range_x = [centerX - 64, centerX+ 64]
    range_y = [centerY - 80, centerY+ 80]
    range_z = [centerZ - 64, centerZ+ 64]

    image_crop_sitk = crop_image_to_region(HR_volume.sitk,range_x,range_y,range_z)
    mask_crop_sitk = crop_image_to_region(HR_volume.sitk_mask,range_x,range_y,range_z)

    return image_crop_sitk,mask_crop_sitk


def crop_image_to_region(image_sitk, range_x, range_y, range_z):

    image_cropped_sitk = image_sitk[
        range_x[0]:range_x[1],
        range_y[0]:range_y[1],
        range_z[0]:range_z[1]
    ]

    return image_cropped_sitk



class SUFFICENT(nn.Module):
    def __init__(self,input_stacks, input_stack_masks,args) -> None:
        super().__init__()
    # device: torch.device = DeviceConfigs()
        self.device= args.device
        self.img_clean_var = None
        self.image_channels = 1
        # learning_rate: float = 0.001
        # self.learning_rate = 0.0001
        self.learning_rate = 0.005
        self.pose_learning_rate = 0.005
        self.epochs = args.max_iter
        self.reg_num=1000
        # first_reg: int=200
        # first_reg: int=3000
        # self.first_reg=2000
        self.res_r = 0.8
        self.res_s = 0.8
        self.s_thick = 6
        self.real_slice = None
        self.slice_shape = None
        self.psf = None
        self.dataset = None
        self.first_reg=500
        self.tv: float = 0.000
        # self.tv: float = 0.01
        # tv: float = 0.1
        self.outlier_factor= 0.0
        self.results_dir = args.results_dir
        self.num_channels = [256,128,128,64,64]
        # self.num_channels = [256,128,64,64]
        self.OPTIMIZER = 'adam'
        self.opt_input = False
        self.reg_noise_std = 0.005
        self.reg_noise_decayevery = 20_000
        self.mask_var = None
        self.apply_f = None
        self.lr_decay_epoch =3000
        self.net_input = None
        self.fixed_noise = None
        self.net_input_path=None
        self.volume_sampleGrid = None
        self.volume_bound = None
        self.volume_InvCovScaled = None
        self.find_best = False
        self.weight_decay = 0
        self.upsample_first = True

        self.init(args,input_stacks,input_stack_masks)

    @property
    def transformation(self) -> RigidTransform:
        return RigidTransform(self.axisangle.detach(), trans_first=True)

    @transformation.setter
    def transformation(self, value: RigidTransform) -> None:
        if self.n_slices == 0:
            self.n_slices = len(value)
        else:
            assert self.n_slices == len(value)
        axisangle = value.axisangle(trans_first=True)
        if TYPE_CHECKING:
            self.axisangle_init: torch.Tensor
        self.register_buffer("axisangle_init", axisangle.detach().clone())
        # if not self.args.no_transformation_optimization:
        self.axisangle = nn.Parameter(axisangle.detach().clone(),requires_grad=True)
        # else:
        # self.register_buffer("axisangle", axisangle.detach().clone())
        # print(self.axisangle[5])
        # print("init grad",self.axisangle[5].grad)
        # print(self.axisangle)
        # print("axisangle",self.axisangle.grad)

    def init(self,args, input_stacks, stack_masks):
        input_dict: Dict[str, Any] = dict()
        # if getattr(args, "input_stacks", None) is not None:
        ori_dataset = []
        # input_stacks = []
        # slices = []
        for i, f in enumerate(input_stacks):
            stack = load_stack(
                f,
                # stack_masks[i] if args.stack_masks is not None else None,
                # stack_masks[i],
                None,
                device=args.device,
            )
            stack.slices /= torch.quantile(stack.slices, 0.99)
            ori_dataset.append(stack)

        self.dataset = ori_dataset
        self.real_slice = torch.cat([s.slices for s in ori_dataset], dim=0)
        self.slice_shape = ori_dataset[0].shape[-2:]

        res_s = ori_dataset[0].resolution_x
        s_thick = ori_dataset[0].thickness
        self.res_s = res_s
        self.s_thick = s_thick
        psf = get_PSF(
            res_ratio=(res_s / self.res_r, res_s / self.res_r, s_thick / self.res_r),
            device=args.device,
        )
        self.psf = psf


        self.SP = Select_Profile(n_samples=256,device=self.device).to(self.device)

        self.n_slices = len(self.real_slice)

        # dofs = self.dataset.dof
        self.transformation = RigidTransform.cat(
            [s.transformation for s in ori_dataset]
        )


        self.logit_coef = nn.Parameter(
            torch.zeros(self.n_slices, dtype=torch.float32,device=args.device),requires_grad=True
        )
        self.slice_weight = nn.Parameter(
            torch.ones(self.n_slices, dtype=torch.float32, device=args.device), requires_grad=True
        )




        volume = Volume.zeros((128, 160, 128), self.res_r, device=args.device)
        img_clean_var = volume.image.unsqueeze(0).unsqueeze(0)
        self.img_clean_var = img_clean_var
        # totalupsample = 2 ** len(self.num_channels)
        totalupsample = 2 ** (len(self.num_channels)-2)

        depth = int(self.img_clean_var.data.shape[2] / totalupsample)
        height = int(self.img_clean_var.data.shape[3] / totalupsample)
        width = int(self.img_clean_var.data.shape[4] / totalupsample)
        shape = [1, self.num_channels[0], depth, height, width]
        print("shape: ", shape)
        net_input = torch.rand(shape).to(args.device)
        # net_input = torch.rand_like(volume).to(args.device)
        net_input.data *= 0.1

        self.net_input = net_input

        self.net_input_saved = self.net_input.data.clone()
        self.noise = self.net_input.data.clone()


        self.model = Deep_Decoder(
        self.image_channels, num_channels_up=self.num_channels,
        upsample_first=True, filter_size_up=3,need_sigmoid=True,need_relu=False,upsample_mode="trilinear",
    ).to(args.device)

        self.tvloss = TVLoss(TVLoss_weight=self.tv)




        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr_srr
        )
        self.optimizer_mat = torch.optim.AdamW(
            [self.axisangle],
            lr=args.lr_srr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=8000, gamma=0.33)
        self.scheduler_mat = torch.optim.lr_scheduler.StepLR(self.optimizer_mat, step_size=8000, gamma=0.33)
        # self.scheduler_input = torch.optim.lr_scheduler.StepLR(self.optimizer_mat, step_size=3000, gamma=0.33)


        # self.exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
        self.mse = torch.nn.MSELoss()
        # self.l1_loss = torch.nn.L1Loss()
        if self.find_best:
            self.best_mse = 1000000.0

        # tracker.set_image("valuate",True)


    def trans_loss(self, trans_first: bool = True) -> torch.Tensor:
        x = RigidTransform(self.axisangle, trans_first=trans_first)
        y = RigidTransform(self.axisangle_init, trans_first=trans_first)
        err = y.inv().compose(x).axisangle(trans_first=trans_first)
        loss_R = torch.mean(err[:, :3] ** 2)
        loss_T = torch.mean(err[:, 3:] ** 2)
        # return loss_R + 1e-3 * self.spatial_scaling * self.spatial_scaling * loss_T
        return loss_R + 1e-3 * loss_T


    def valuate(self, current_epoch):

        with torch.no_grad():
            volume = self.model(self.net_input)
            ph.create_directory(self.results_dir + os.sep)
            out_path = self.results_dir + os.sep +"{}.nii.gz".format(int(current_epoch/ 1000))
            gaussian_volume = Volume(volume[0, 0], None, None, self.res_r)
            gaussian_volume.save(out_path)


    def train_epoch(self, epoch,mat):

        # tracker.add_global_step()


        count = 0
        epoch_loss = 0.0


        self.optimizer.zero_grad()
        self.optimizer_mat.zero_grad()
        net_input = Variable(self.net_input_saved + (self.noise.normal_() * self.reg_noise_std))
        # volume = self.model(self.net_input)
        volume = self.model(net_input)

        c: Any = F.softmax(self.logit_coef,0) * self.n_slices
        w: Any = self.slice_weight

        slices = slice_acquisition(
            # mat,
            mat,
            volume,
            None,
            None,
            self.psf,
            self.slice_shape,
            self.res_s / self.res_r,
            False,
            False,
        )
        # print(slices.shape)
        # print(data["stacks"].shape)

        # l1 = l1_loss(slices, self.real_slice)
        l1 = l2_loss(slices, self.real_slice)
        if epoch > 100:
            ltrans = self.trans_loss()
        else:
            ltrans = 0.0 * self.trans_loss()
        ltv = self.tvloss(volume)
        # lssim = ssim(slices,self.real_slice)

        loss = l1 + 0.1*ltrans+ltv

        # print("iter: {} l1: {} trans: {} tv:{} ssim:{} loss: {} lr:{}".format(epoch, l1.item(), ltrans.item(), ltv.item(),
        if epoch+1 % 500 == 0:
            print("iter: {} l1: {} trans: {} tv:{} loss: {} lr:{}".format(epoch, l1.item(), ltrans.item(), ltv.item(),
                                                                      loss.item(), self.scheduler.get_lr()))



        loss.backward()

        self.optimizer.step()
        self.optimizer_mat.step()
        self.scheduler.step()
        self.scheduler_mat.step()




    def generate_filters(self,stacks,fix_tensor):
        for i,stack in enumerate(stacks):
            slices = stack.get_slices()
            for j,slice in enumerate(slices):
                sampleGrid,bound,InvCovScaled = generate_filter(self.HR_volume, slice,fix_tensor)
                # W, sampleGrid = generate_filter(self.HR_volume, slice,fix_tensor)
                # slice.set_W(W)
                slice.set_sampleGrid(sampleGrid)
                slice.set_bound(bound)
                slice.set_InvCovScaled(InvCovScaled)



    def s2v_register(self,epoch,method="NCC"):

        with torch.no_grad():
            # out = self.model(self.fixed_noise.type(dtype).to(self.device)) * out_mask
            volume = self.model(self.net_input)

        mat = s2v_registration(epoch,self.real_slice, volume.detach(), self.axisangle, self.psf,self.res_s,self.res_r,
                               self.device)


        return mat


    def run(self):

        for i in range(self.epochs):
            # gaussian_matrix = axisangle2mat(gaussian_mat.view(-1, 6)).view(gaussian_mat.shape[:-1] + (3, 4))
            self.optimizer.step()
            self.optimizer_mat.step()

            if i < 100:
                mat = mat_update_resolution(
                    axisangle2mat(self.axisangle.view(-1, 6)), self.res_s, self.res_r
                )
                # mat = mat.detach().clone()

            if i >= 100 :
                mat = self.s2v_register(i,"NCC")

            self.train_epoch( i+1,mat)
            # outlier_count += 1
            #
            # if (i + 1) > self.first_reg and (outlier_count + 1) % 200 == 0:
            #     if outlier_factor + ((outlier_count + 1) / 2500.0) < 0.85:
            #         self.outlier((i + 1), outlier_factor + ((outlier_count + 1) / 2500.0), stacks)
            #         b = get_stacks_b(stacks, self.HR_volume)
            # self.exp_lr_scheduler.step()
            if (i + 1) % 1000 == 0:
                self.valuate(i + 1)



    def exp_lr_scheduler(self,optimizer, epoch, init_lr=0.001, lr_decay_epoch=500):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = init_lr * (0.5 ** (epoch // lr_decay_epoch))
        print("lr:",lr)

        if epoch % lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer


if __name__ == '__main__':
    desc = 'Training registration generator'
    parser = argparse.ArgumentParser(description=desc)

    # data organization parameters
    parser.add_argument('--gpu', default=7, help='GPU ID number(s), comma-separated (default: 3)')
    parser.add_argument('--tv', default="0.05", help='TV weight ')
    parser.add_argument('--lr-srr', default=5e-4, help='learn rate')
    parser.add_argument('--lr-svr', default=1e-2, help='learn rate')
    parser.add_argument('--max_iter', type=int, default=8000)
    args = parser.parse_args()
    args.results_dir = "data/out"
    args.device = torch.device('cuda:{}'.format(args.gpu) if args.gpu is not None else 'cpu')

    input_stacks = []
    input_stacks_masks = []

    input_stacks.append("data/stack1.nii.gz")
    input_stacks.append("data/stack2.nii.gz")
    input_stacks.append("data/stack3.nii.gz")


    input_stacks_masks.append("data/seg/stack1.nii.gz")
    input_stacks_masks.append("data/seg/stack2.nii.gz")
    input_stacks_masks.append("data/seg/stack3.nii.gz")


    sufficient = SUFFICENT(input_stacks,input_stacks_masks,args)
    sufficient.run()
