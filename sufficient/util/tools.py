from __future__ import annotations
import os
from typing import Dict, Optional, Union, List, Tuple, Sequence, cast, Any, Collection, Sequence, Iterable
import collections
from math import log, sqrt
from sufficient.transform import (
    RigidTransform,
    transform_points,
    init_stack_transform,
    init_zero_transform,
)

from typing import Tuple, Union, Optional
import os
import nibabel as nib
import torch
import numpy as np
from typing import Union
from os import PathLike
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

PathType = Union[str, PathLike[str]]
DeviceType = Union[torch.device, str, None]
def tv_3d_loss(vol, reduction="sum"):

    dx = torch.abs(torch.diff(vol, dim=0))
    dy = torch.abs(torch.diff(vol, dim=1))
    dz = torch.abs(torch.diff(vol, dim=2))

    tv = torch.sum(dx) + torch.sum(dy) + torch.sum(dz)

    if reduction == "mean":
        total_elements = (
            (vol.shape[0] - 1) * vol.shape[1] * vol.shape[2]
            + vol.shape[0] * (vol.shape[1] - 1) * vol.shape[2]
            + vol.shape[0] * vol.shape[1] * (vol.shape[2] - 1)
        )
        tv = tv / total_elements
    return tv


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def resample(
    x: torch.Tensor, res_xyz_old: Sequence, res_xyz_new: Sequence
) -> torch.Tensor:
    ndim = x.ndim - 2
    assert len(res_xyz_new) == len(res_xyz_old) == ndim
    if all(r_new == r_old for (r_new, r_old) in zip(res_xyz_new, res_xyz_old)):
        return x
    grids = []
    for i in range(ndim):
        fac = res_xyz_old[i] / res_xyz_new[i]
        size_new = int(x.shape[-i - 1] * fac)
        grid_max = (size_new - 1) / fac / (x.shape[-i - 1] - 1)
        grids.append(
            torch.linspace(
                -grid_max, grid_max, size_new, dtype=x.dtype, device=x.device
            )
        )
    grid = torch.stack(torch.meshgrid(*grids[::-1], indexing="ij")[::-1], -1)
    y = F.grid_sample(
        x, grid[None].expand((x.shape[0],) + (-1,) * (ndim + 1)), align_corners=True
    )
    return y


def meshgrid(
    shape_xyz: Collection,
    resolution_xyz: Collection,
    min_xyz: Optional[Collection] = None,
    device: DeviceType = None,
    stack_output: bool = True,
):
    assert len(shape_xyz) == len(resolution_xyz)
    if min_xyz is None:
        min_xyz = tuple(-(s - 1) * r / 2 for s, r in zip(shape_xyz, resolution_xyz))
    else:
        assert len(shape_xyz) == len(min_xyz)

    if device is None:
        if isinstance(shape_xyz, torch.Tensor):
            device = shape_xyz.device
        elif isinstance(resolution_xyz, torch.Tensor):
            device = resolution_xyz.device
        else:
            device = torch.device("cpu")
    dtype = torch.float32

    arr_xyz = [
        torch.arange(s, dtype=dtype, device=device) * r + m
        for s, r, m in zip(shape_xyz, resolution_xyz, min_xyz)
    ]
    grid_xyz = torch.meshgrid(arr_xyz[::-1], indexing="ij")[::-1]
    if stack_output:
        return torch.stack(grid_xyz, -1)
    else:
        return grid_xyz


def gaussian_blur(
    x: torch.Tensor, sigma: Union[float, collections.abc.Iterable], truncated: float
) -> torch.Tensor:
    spatial_dims = len(x.shape) - 2
    if not isinstance(sigma, collections.abc.Iterable):
        sigma = [sigma] * spatial_dims
    kernels = [gaussian_1d_kernel(s, truncated, x.device) for s in sigma]
    c = x.shape[1]
    conv_fn = [F.conv1d, F.conv2d, F.conv3d][spatial_dims - 1]
    for d in range(spatial_dims):
        s = [1] * len(x.shape)
        s[d + 2] = -1
        k = kernels[d].reshape(s).repeat(*([c, 1] + [1] * spatial_dims))
        padding = [0] * spatial_dims
        padding[d] = (k.shape[d + 2] - 1) // 2
        x = conv_fn(x, k, padding=padding, groups=c)
    return x


# from MONAI
def gaussian_1d_kernel(
    sigma: float, truncated: float, device: DeviceType
) -> torch.Tensor:
    tail = int(max(sigma * truncated, 0.5) + 0.5)
    x = torch.arange(-tail, tail + 1, dtype=torch.float, device=device)
    t = 0.70710678 / sigma
    kernel = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
    return kernel.clamp(min=0)


def compare_resolution_shape(r1, r2, s1, s2) -> bool:
    r1 = np.array(r1)

    r2 = np.array(r2)

    if s1 != s2:
        return False
    if r1.shape != r2.shape:
        return False
    if np.amax(np.abs(r1 - r2)) > 1e-3:
        return False
    return True

def compare_resolution_affine(r1, a1, r2, a2, s1, s2) -> bool:
    r1 = np.array(r1)
    a1 = np.array(a1)
    r2 = np.array(r2)
    a2 = np.array(a2)
    if s1 != s2:
        return False
    if r1.shape != r2.shape:
        return False
    if np.amax(np.abs(r1 - r2)) > 1e-3:
        return False
    if a1.shape != a2.shape:
        return False
    if np.amax(np.abs(a1 - a2)) > 1e-3:
        return False
    return True

GAUSSIAN_FWHM = 1 / (2 * sqrt(2 * log(2)))
SINC_FWHM = 1.206709128803223 * GAUSSIAN_FWHM


def resolution2sigma(rx, ry=None, rz=None, /, isotropic=False):
    if isotropic:
        fx = fy = fz = GAUSSIAN_FWHM
    else:
        fx = fy = SINC_FWHM
        fz = GAUSSIAN_FWHM
    assert not ((ry is None) ^ (rz is None))
    if ry is None:
        if isinstance(rx, float) or isinstance(rx, int):
            if isotropic:
                return fx * rx
            else:
                return fx * rx, fy * rx, fz * rx
        elif isinstance(rx, torch.Tensor):
            if isotropic:
                return fx * rx
            else:
                assert rx.shape[-1] == 3
                return rx * torch.tensor([fx, fy, fz], dtype=rx.dtype, device=rx.device)
        elif isinstance(rx, List) or isinstance(rx, Tuple):
            assert len(rx) == 3
            return resolution2sigma(rx[0], rx[1], rx[2], isotropic=isotropic)
        else:
            raise Exception(str(type(rx)))
    else:
        return fx * rx, fy * ry, fz * rz


def get_PSF(
    r_max: Optional[int] = None,
    res_ratio: Tuple[float, float, float] = (1, 1, 3),
    threshold: float = 1e-3,
    device: DeviceType = torch.device("cpu"),
) -> torch.Tensor:
    sigma_x, sigma_y, sigma_z = resolution2sigma(res_ratio, isotropic=False)
    if r_max is None:
        r_max = max(int(2 * r + 1) for r in (sigma_x, sigma_y, sigma_z))
        r_max = max(r_max, 4)
    x = torch.linspace(-r_max, r_max, 2 * r_max + 1, dtype=torch.float32, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(x, x, x, indexing="ij")

    psf = torch.exp(
        -0.5
        * (
            grid_x**2 / sigma_x**2
            + grid_y**2 / sigma_y**2
            + grid_z**2 / sigma_z**2
        )
    )

    psf[psf.abs() < threshold] = 0
    rx = int(torch.nonzero(psf.sum((0, 1)) > 0)[0, 0].item())
    ry = int(torch.nonzero(psf.sum((0, 2)) > 0)[0, 0].item())
    rz = int(torch.nonzero(psf.sum((1, 2)) > 0)[0, 0].item())
    psf = psf[
        rz : 2 * r_max + 1 - rz, ry : 2 * r_max + 1 - ry, rx : 2 * r_max + 1 - rx
    ].contiguous()
    psf = psf / psf.sum()
    return psf


def affine2transformation(
    volume: torch.Tensor,
    mask: torch.Tensor,
    resolutions: np.ndarray,
    affine: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, RigidTransform]:
    device = volume.device
    d, h, w = volume.shape

    R = affine[:3, :3]
    negative_det = np.linalg.det(R) < 0

    T = affine[:3, -1:]  # T = R @ (-T0 + T_r)
    R = R @ np.linalg.inv(np.diag(resolutions))

    T0 = np.array([(w - 1) / 2 * resolutions[0], (h - 1) / 2 * resolutions[1], 0])
    T = np.linalg.inv(R) @ T + T0.reshape(3, 1)

    tz = (
        torch.arange(0, d, device=device, dtype=torch.float32) * resolutions[2]
        + T[2].item()
    )
    tx = torch.ones_like(tz) * T[0].item()
    ty = torch.ones_like(tz) * T[1].item()
    t = torch.stack((tx, ty, tz), -1).view(-1, 3, 1)
    R = torch.tensor(R, device=device).unsqueeze(0).repeat(d, 1, 1)

    if negative_det:
        volume = torch.flip(volume, (-1,))
        mask = torch.flip(mask, (-1,))
        t[:, 0, -1] *= -1
        R[:, :, 0] *= -1

    transformation = RigidTransform(
        torch.cat((R, t), -1).to(torch.float32), trans_first=True
    )

    return volume, mask, transformation


def transformation2affine(
    volume: torch.Tensor,
    transformation: RigidTransform,
    resolution_x: float,
    resolution_y: float,
    resolution_z: float,
) -> np.ndarray:
    mat = transformation.matrix(trans_first=True).detach().cpu().numpy()
    assert mat.shape[0] == 1
    R = mat[0, :, :-1]
    T = mat[0, :, -1:]
    d, h, w = volume.shape
    affine = np.eye(4)
    T[0] -= (w - 1) / 2 * resolution_x
    T[1] -= (h - 1) / 2 * resolution_y
    T[2] -= (d - 1) / 2 * resolution_z
    T = R @ T.reshape(3, 1)
    R = R @ np.diag([resolution_x, resolution_y, resolution_z])
    affine[:3, :] = np.concatenate((R, T), -1)
    return affine


def save_nii_volume(
    path: PathType,
    volume: Union[torch.Tensor, np.ndarray],
    affine: Optional[Union[torch.Tensor, np.ndarray]],
) -> None:
    assert len(volume.shape) == 3 or (len(volume.shape) == 4 and volume.shape[1] == 1)
    if len(volume.shape) == 4:
        volume = volume.squeeze(1)
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy().transpose(2, 1, 0)
    else:
        volume = volume.transpose(2, 1, 0)
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    if affine is None:
        affine = np.eye(4)
    if volume.dtype == bool and isinstance(
        volume, np.ndarray
    ):  # bool type is not supported
        volume = volume.astype(np.int16)
    img = nib.nifti1.Nifti1Image(volume, affine)
    img.header.set_xyzt_units(2)
    img.header.set_qform(affine, code="aligned")
    img.header.set_sform(affine, code="scanner")
    nib.save(img, os.fspath(path))


def load_nii_volume(path: PathType) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = nib.load(os.fspath(path))

    dim = img.header["dim"]
    assert dim[0] == 3 or (dim[0] > 3 and all(d == 1 for d in dim[4:])), (
        "Expect a 3D volume but the input is %dD" % dim[0]
    )

    volume = img.get_fdata().astype(np.float32)
    while volume.ndim > 3:
        volume = volume.squeeze(-1)
    volume = volume.transpose(2, 1, 0)

    resolutions = img.header["pixdim"][1:4]

    affine = img.affine
    if np.any(np.isnan(affine)):
        affine = img.get_qform()

    return volume, resolutions, affine


class _Data(object):
    def __init__(
        self,
        data: torch.Tensor,
        mask: Optional[torch.Tensor],
        transformation: Optional[RigidTransform],
    ) -> None:
        if mask is None:
            mask = torch.ones_like(data, dtype=torch.bool)
        if transformation is None:
            transformation = init_zero_transform(1, data.device)
        self.data = data
        self.mask = mask
        self.transformation = transformation

    def check_data(self, value) -> None:
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("Data must be Tensor!")

    def check_mask(self, value) -> None:
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("Mask must be Tensor!")
        if value.shape != self.shape:
            raise RuntimeError("Mask has a shape different from image!")
        if value.dtype != torch.bool:
            raise RuntimeError("Mask must be bool!")
        if value.device != self.device:
            raise RuntimeError("The device of mask is different!")

    def check_transformation(self, value) -> None:
        if not isinstance(value, RigidTransform):
            raise RuntimeError("Transformation must be RigidTransform")
        if value.device != self.device:
            raise RuntimeError("The device of transformation must be the same as data!")

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, value: torch.Tensor) -> None:
        self.check_data(value)
        self._data = value

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    @mask.setter
    def mask(self, value: torch.Tensor) -> None:
        self.check_mask(value)
        self._mask = value

    @property
    def transformation(self) -> RigidTransform:
        return self._transformation

    @transformation.setter
    def transformation(self, value: RigidTransform) -> None:
        self.check_transformation(value)
        self._transformation = value

    @property
    def device(self) -> DeviceType:
        return self.data.device

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def clone(self, *, zero: bool = False, deep: bool = True) -> _Data:
        raise NotImplementedError()

    def _clone_dict(self, zero: bool = False, deep: bool = True) -> Dict:
        data = self.data
        mask = self.mask
        transformation = self.transformation
        if zero:
            data = torch.zeros_like(data)
            mask = torch.zeros_like(mask)
        elif deep:
            data = data.clone()
            mask = mask.clone()
        if deep:
            transformation = transformation.clone()
        return {
            "data": data,
            "mask": mask,
            "transformation": self.transformation,
        }


class Image(_Data):
    def __init__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        transformation: Optional[RigidTransform] = None,
        resolution_x: Union[float, torch.Tensor] = 1.0,
        resolution_y: Union[float, torch.Tensor, None] = None,
        resolution_z: Union[float, torch.Tensor, None] = None,
    ) -> None:
        super().__init__(image, mask, transformation)
        if resolution_y is None:
            resolution_y = resolution_x
        if resolution_z is None:
            resolution_z = resolution_x
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z

    def check_data(self, value) -> None:
        super().check_data(value)
        if value.ndim != 3:
            raise RuntimeError("The dimension of image must be 3!")

    def check_transformation(self, value) -> None:
        super().check_transformation(value)
        if len(value) != 1:
            raise RuntimeError("The len of transformation must be 1!")

    @property
    def image(self) -> torch.Tensor:
        return self.data

    @image.setter
    def image(self, value: torch.Tensor) -> None:
        self.data = value

    def _clone_dict(self, zero: bool = False, deep: bool = True) -> Dict:
        d = super()._clone_dict(zero, deep)
        d["resolution_x"] = float(self.resolution_x)
        d["resolution_y"] = float(self.resolution_y)
        d["resolution_z"] = float(self.resolution_z)
        d["image"] = d.pop("data")
        return d

    @property
    def shape_xyz(self) -> torch.Tensor:
        return torch.tensor(self.image.shape[::-1], device=self.image.device)

    @property
    def resolution_xyz(self) -> torch.Tensor:
        return torch.tensor(
            [self.resolution_x, self.resolution_y, self.resolution_z],
            device=self.image.device,
        )

    def save(self, path: PathType, masked=True) -> None:
        affine = transformation2affine(
            self.image,
            self.transformation,
            float(self.resolution_x),
            float(self.resolution_y),
            float(self.resolution_z),
        )
        if masked:
            output_volume = self.image * self.mask.to(self.image.dtype)
        else:
            output_volume = self.image
        save_nii_volume(path, output_volume, affine)

    def save_mask(self, path: PathType) -> None:
        affine = transformation2affine(
            self.image,
            self.transformation,
            float(self.resolution_x),
            float(self.resolution_y),
            float(self.resolution_z),
        )
        output_volume = self.mask.to(self.image.dtype)
        save_nii_volume(path, output_volume, affine)

    @property
    def xyz_masked(self) -> torch.Tensor:
        return transform_points(self.transformation, self.xyz_masked_untransformed)

    @property
    def xyz_masked_untransformed(self) -> torch.Tensor:
        kji = torch.flip(torch.nonzero(self.mask), (-1,))
        return (kji - (self.shape_xyz - 1) / 2) * self.resolution_xyz

    @property
    def v_masked(self) -> torch.Tensor:
        return self.image[self.mask]

    def rescale(
        self, intensity_mean: Union[float, torch.Tensor], masked: bool = True
    ) -> None:
        if masked:
            scale_factor = intensity_mean / self.image[self.mask].mean()
        else:
            scale_factor = intensity_mean / self.image.mean()
        self.image *= scale_factor

    @staticmethod
    def like(
        old: Image,
        image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        deep: bool = True,
    ) -> Image:
        if image is None:
            image = old.image.clone() if deep else old.image
        if mask is None:
            mask = old.mask.clone() if deep else old.mask
        transformation = old.transformation.clone() if deep else old.transformation
        return old.__class__(
            image=image,
            mask=mask,
            transformation=transformation,
            resolution_x=old.resolution_x,
            resolution_y=old.resolution_y,
            resolution_z=old.resolution_z,
        )


class Slice(Image):
    def check_data(self, value) -> None:
        super().check_data(value)
        if value.shape[0] != 1:
            raise RuntimeError("The shape of a slice must be (1, H, W)!")

    def clone(self, *, zero: bool = False, deep: bool = True) -> Slice:
        return Slice(
            **self._clone_dict(zero, deep),
        )

    def resample(
        self,
        resolution_new: Union[float, Sequence],
    ) -> Slice:
        if isinstance(resolution_new, float) or len(resolution_new) == 1:
            resolution_new = [resolution_new, resolution_new]

        if len(resolution_new) == 3:
            resolution_z_new = resolution_new[-1]
            resolution_new = resolution_new[:-1]
        else:
            resolution_z_new = self.resolution_z

        image = resample(
            self.image[None],
            (self.resolution_x, self.resolution_y),
            resolution_new,
        )[0]
        mask = (
            resample(
                self.mask[None].float(),
                (self.resolution_x, self.resolution_y),
                resolution_new,
            )[0]
            > 0
        )

        new_slice = cast(Slice, Slice.like(self, image, mask, deep=True))
        new_slice.resolution_z = resolution_z_new

        return new_slice


class Volume(Image):
    def sample_points(self, xyz: torch.Tensor) -> torch.Tensor:
        shape = xyz.shape[:-1]
        xyz = transform_points(self.transformation.inv(), xyz.view(-1, 3))
        xyz = xyz / ((self.shape_xyz - 1) * self.resolution_xyz / 2)
        return F.grid_sample(
            self.image[None, None],
            xyz.view(1, 1, 1, -1, 3),
            align_corners=True,
        ).view(shape)

    def resample(
        self,
        resolution_new: Optional[Union[float, torch.Tensor]],
        transformation_new: Optional[RigidTransform],
    ) -> Volume:
        if transformation_new is None:
            transformation_new = self.transformation
        R = transformation_new.matrix()[0, :3, :3]
        dtype = R.dtype
        device = R.device
        if resolution_new is None:
            resolution_new = self.resolution_xyz
        elif isinstance(resolution_new, float) or resolution_new.numel == 1:
            resolution_new = torch.tensor(
                [resolution_new] * 3, dtype=dtype, device=device
            )

        xyz = self.xyz_masked
        # new rotation
        xyz = torch.matmul(torch.inverse(R), xyz.view(-1, 3, 1))[..., 0]

        xyz_min = xyz.amin(0) - resolution_new * 10
        xyz_max = xyz.amax(0) + resolution_new * 10
        shape_xyz = ((xyz_max - xyz_min) / resolution_new).ceil().long()

        mat = torch.zeros((1, 3, 4), dtype=R.dtype, device=R.device)
        mat[0, :, :3] = R
        mat[0, :, -1] = xyz_min + (shape_xyz - 1) / 2 * resolution_new

        xyz = meshgrid(shape_xyz, resolution_new, xyz_min, device, True)

        xyz = torch.matmul(R, xyz[..., None])[..., 0]

        v = self.sample_points(xyz)

        return Volume(
            v,
            v > 0,
            RigidTransform(mat, trans_first=True),
            resolution_new[0].item(),
            resolution_new[1].item(),
            resolution_new[2].item(),
        )

    def clone(self, *, zero: bool = False, deep: bool = True) -> Volume:
        return Volume(**self._clone_dict(zero))

    @staticmethod
    def zeros(
        shape: Tuple,
        resolution_x,
        resolution_y=None,
        resolution_z=None,
        device: DeviceType = None,
    ) -> Volume:
        image = torch.zeros(shape, dtype=torch.float32, device=device)
        mask = torch.ones_like(image, dtype=torch.bool)
        return Volume(
            image,
            mask,
            transformation=None,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            resolution_z=resolution_z,
        )


class Stack(_Data):
    def __init__(
        self,
        slices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        transformation: Optional[RigidTransform] = None,
        resolution_x: float = 1.0,
        resolution_y: Optional[float] = None,
        thickness: Optional[float] = None,
        gap: Optional[float] = None,
        name: str = "",
    ) -> None:
        if resolution_y is None:
            resolution_y = resolution_x
        if thickness is None:
            thickness = gap if gap is not None else resolution_x
        if gap is None:
            gap = thickness
        if transformation is None:
            transformation = init_stack_transform(slices.shape[0], gap, slices.device)
        super().__init__(slices, mask, transformation)
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.thickness = thickness
        self.gap = gap
        self.name = name

    def check_data(self, value) -> None:
        super().check_data(value)
        if value.ndim != 4:
            raise RuntimeError("Stack must be 4D data")
        if value.shape[1] != 1:
            raise RuntimeError("Stack must has shape (N, 1, H, W)")

    def check_transformation(self, value) -> None:
        super().check_transformation(value)
        if len(value) != self.slices.shape[0]:
            raise RuntimeError(
                "The number of transformatons is not equal to the number of slices!"
            )

    @property
    def slices(self) -> torch.Tensor:
        return self.data

    @slices.setter
    def slices(self, value: torch.Tensor) -> None:
        self.data = value

    def __len__(self) -> int:
        return self.slices.shape[0]

    def __getitem__(self, idx):
        slices = self.slices[idx]
        masks = self.mask[idx]
        transformation = self.transformation[idx]
        if slices.ndim < self.slices.ndim:
            return Slice(
                slices,
                masks,
                transformation,
                self.resolution_x,
                self.resolution_y,
                self.thickness,
            )
        else:
            return [
                Slice(
                    slices[i],
                    masks[i],
                    transformation[i],
                    self.resolution_x,
                    self.resolution_y,
                    self.thickness,
                )
                for i in range(len(transformation))
            ]

    def get_substack(self, idx_from=None, idx_to=None, /) -> Stack:
        if idx_to is None:
            slices = self.slices[idx_from]
            masks = self.mask[idx_from]
            transformation = self.transformation[idx_from]
        else:
            slices = self.slices[idx_from:idx_to]
            masks = self.mask[idx_from:idx_to]
            transformation = self.transformation[idx_from:idx_to]
        return Stack(
            slices,
            masks,
            transformation,
            self.resolution_x,
            self.resolution_y,
            self.thickness,
            self.gap,
            self.name,
        )

    def get_mask_volume(self) -> Volume:
        mask = self.mask.squeeze(1).clone()
        return Volume(
            image=mask.float(),
            mask=mask > 0,
            transformation=self.transformation.mean(),
            resolution_x=self.resolution_x,
            resolution_y=self.resolution_y,
            resolution_z=self.gap,
        )

    def get_volume(self, copy=True) -> Volume:
        image = self.slices.squeeze(1)
        mask = self.mask.squeeze(1)
        if copy:
            image = image.clone()
            mask = mask.clone()
        return Volume(
            image=image,
            mask=mask,
            transformation=self.transformation.mean(),
            resolution_x=self.resolution_x,
            resolution_y=self.resolution_y,
            resolution_z=self.gap,
        )

    def apply_volume_mask(self, mask: Volume) -> None:
        for i in range(len(self)):
            s = self[i]
            assign_mask = self.mask[i].clone()
            self.mask[i][assign_mask] = mask.sample_points(s.xyz_masked) > 0

    def _clone_dict(self, zero: bool = False, deep: bool = True) -> Dict:
        d = super()._clone_dict(zero, deep)
        d["slices"] = d.pop("data")
        d["resolution_x"] = float(self.resolution_x)
        d["resolution_y"] = float(self.resolution_y)
        d["thickness"] = float(self.thickness)
        d["gap"] = float(self.gap)
        d["name"] = self.name
        return d

    def clone(self, *, zero: bool = False, deep: bool = True) -> Stack:
        return Stack(**self._clone_dict(zero, deep))

    @staticmethod
    def like(
        stack: Stack,
        slices: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        deep: bool = True,
    ) -> Stack:
        if slices is None:
            slices = stack.slices.clone() if deep else stack.slices
        if mask is None:
            mask = stack.mask.clone() if deep else stack.mask
        transformation = stack.transformation.clone() if deep else stack.transformation
        return Stack(
            slices=slices,
            mask=mask,
            transformation=transformation,
            resolution_x=stack.resolution_x,
            resolution_y=stack.resolution_y,
            thickness=stack.thickness,
            gap=stack.gap,
        )

    @staticmethod
    def pad_stacks(stacks: List) -> List:
        size_max = max([max(s.shape[-2:]) for s in stacks])
        lists_pad = []
        for s in stacks:
            if s.shape[-1] < size_max or s.shape[-2] < size_max:
                dx1 = (size_max - s.shape[-1]) // 2
                dx2 = (size_max - s.shape[-1]) - dx1
                dy1 = (size_max - s.shape[-2]) // 2
                dy2 = (size_max - s.shape[-2]) - dy1
                data = F.pad(s.data, (dx1, dx2, dy1, dy2))
                mask = F.pad(s.mask, (dx1, dx2, dy1, dy2))
            else:
                data = s.data
                mask = s.mask
            lists_pad.append(s.__class__.like(s, data, mask, deep=False))
        return lists_pad

    @staticmethod
    def cat(inputs: List) -> Stack:
        data = []
        mask = []
        transformation = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, Slice):
                data.append(inp.image[None])
                mask.append(inp.mask[None])
                transformation.append(inp.transformation)
                if i == 0:
                    resolution_x = float(inp.resolution_x)
                    resolution_y = float(inp.resolution_y)
                    thickness = float(inp.resolution_z)
                    gap = float(inp.resolution_z)
            elif isinstance(inp, Stack):
                data.append(inp.slices)
                mask.append(inp.mask)
                transformation.append(inp.transformation)
                if i == 0:
                    resolution_x = inp.resolution_x
                    resolution_y = inp.resolution_y
                    thickness = inp.thickness
                    gap = inp.gap
            else:
                raise TypeError("unkonwn type!")

        return Stack(
            slices=torch.cat(data, 0),
            mask=torch.cat(mask, 0),
            transformation=RigidTransform.cat(transformation),
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            thickness=thickness,
            gap=gap,
        )

    def init_stack_transform(self) -> RigidTransform:
        return init_stack_transform(len(self), self.gap, self.device)


MASK_PREFIX = "mask_"


def save_slices(folder: PathType, images: List[Slice], sep: bool = False) -> None:
    for i, image in enumerate(images):
        if sep:
            image.save(os.path.join(folder, f"{i}.nii.gz"), masked=False)
            image.save_mask(os.path.join(folder, f"{MASK_PREFIX}{i}.nii.gz"))
        else:
            image.save(os.path.join(folder, f"{i}.nii.gz"), masked=True)


def load_slices(
    folder: PathType, device: DeviceType = torch.device("cpu")
) -> List[Slice]:
    slices = []
    ids = []
    for f in os.listdir(folder):
        if not (f.endswith("nii") or f.endswith("nii.gz")):
            continue
        if f.startswith(MASK_PREFIX):
            continue
        ids.append(int(f.split(".nii")[0]))
        slice, resolutions, affine = load_nii_volume(os.path.join(folder, f))
        slice_tensor = torch.tensor(slice, device=device)
        if os.path.exists(os.path.join(folder, MASK_PREFIX + f)):
            mask, _, _ = load_nii_volume(os.path.join(folder, MASK_PREFIX + f))
            mask_tensor = torch.tensor(mask, device=device, dtype=torch.bool)
        else:
            mask_tensor = torch.ones_like(slice_tensor, dtype=torch.bool)
        # slice_tensor > 0
        slice_tensor, mask_tensor, transformation = affine2transformation(
            slice_tensor, mask_tensor, resolutions, affine
        )
        slices.append(
            Slice(
                image=slice_tensor,
                mask=mask_tensor,
                transformation=transformation,
                resolution_x=resolutions[0],
                resolution_y=resolutions[1],
                resolution_z=resolutions[2],
            )
        )
    return [slice for _, slice in sorted(zip(ids, slices))]


def load_stack(
    path_vol: PathType,
    path_mask: Optional[PathType] = None,
    device: DeviceType = torch.device("cpu"),
) -> Stack:
    slices, resolutions, affine = load_nii_volume(path_vol)
    if path_mask is None:
        mask = np.ones_like(slices, dtype=bool)
    else:
        mask, resolutions_m, affine_m = load_nii_volume(path_mask)
        mask = mask > 0

        if not compare_resolution_shape(
            resolutions, resolutions_m, slices.shape, mask.shape
        ):
            raise Exception(
                "Error: the sizes/resolutions/affine transformations of the input stack and stack mask do not match!"
            )

    slices_tensor = torch.tensor(slices, device=device)
    mask_tensor = torch.tensor(mask, device=device)

    slices_tensor, mask_tensor, transformation = affine2transformation(
        slices_tensor, mask_tensor, resolutions, affine
    )

    return Stack(
        slices=slices_tensor.unsqueeze(1),
        mask=mask_tensor.unsqueeze(1),
        transformation=transformation,
        resolution_x=resolutions[0],
        resolution_y=resolutions[1],
        thickness=resolutions[2],
        gap=resolutions[2],
        name=str(path_vol),
    )


def load_volume(
    path_vol: PathType,
    path_mask: Optional[PathType] = None,
    device: DeviceType = torch.device("cpu"),
) -> Volume:
    vol, resolutions, affine = load_nii_volume(path_vol)
    if path_mask is None:
        # mask = vol > 0
        mask = np.ones_like(vol, dtype=bool)
    else:
        mask, resolutions_m, affine_m = load_nii_volume(path_mask)
        mask = mask > 0
        if not compare_resolution_affine(
            resolutions, affine, resolutions_m, affine_m, vol.shape, mask.shape
        ):
            raise Exception(
                "Error: the sizes/resolutions/affine transformations of the input stack and stack mask do not match!"
            )

    vol_tensor = torch.tensor(vol, device=device)
    mask_tensor = torch.tensor(mask, device=device)

    vol_tensor, mask_tensor, transformation = affine2transformation(
        vol_tensor, mask_tensor, resolutions, affine
    )

    transformation = RigidTransform(transformation.axisangle().mean(0, keepdim=True))

    return Volume(
        image=vol_tensor,
        mask=mask_tensor,
        transformation=transformation,
        resolution_x=resolutions[0],
        resolution_y=resolutions[1],
        resolution_z=resolutions[2],
    )


def load_mask(path_mask: PathType, device: DeviceType = torch.device("cpu")) -> Volume:
    return load_volume(path_mask, path_mask, device)
