# -*- coding: utf-8 -*-
# ------------------------------------
# @Author  : Jiangjie Wu
# @FileName: preprocessing.py
# @Software: PyCharm
# @Email   : wujj@shanghaitech.edu.cn
# @Date    : 2025-05-29
import numpy as np
import SimpleITK as sitk
import time
import pandas as pd
from os import path
import os
import sys
# import cv2
import imageio
import re
import torch
import torchgeometry as tgm
import math
# from utils import transformations as tfms
import random
import pysitk.python_helper as ph
import niftymic.base.data_writer as dw
import niftymic.utilities.n4_bias_field_correction as n4itk
import yaml
import monaifbs
from scipy import ndimage
from torch.utils import data


import niftymic.utilities.target_stack_estimator as ts_estimator
import niftymic.reconstruction.scattered_data_approximation as sda
import niftymic.registration.niftyreg as niftyreg
import niftymic.utilities.joint_image_mask_builder as imb
import niftymic.utilities.data_preprocessing as dp
import niftymic.utilities.intensity_correction as ic
import niftymic.utilities.segmentation_propagation as segprop
import niftymic.base.stack as st
import niftymic.base.data_reader as dr
import niftymic.utilities.volumetric_reconstruction_pipeline as pipeline
import niftymic.utilities.outlier_rejector as outre

def get_cropped_sitk(HR_volume,size=80):
    x,y,z = HR_volume.sitk.GetSize()
    centerX = x // 2
    centerY = y // 2
    centerZ = z // 2


    # range_x = [centerX - 64, centerX+ 64]
    # range_y = [centerY - 64, centerY+ 64]
    # range_z = [centerZ - 64, centerZ+ 64]

    range_x = [centerX - size, centerX+ size]
    range_y = [centerY - size, centerY+ size]
    range_z = [centerZ - size, centerZ+ size]

    # range_x = [centerX - 208, centerX + 208]
    # range_y = [centerY - 208, centerY + 208]
    # range_z = [centerZ - 192, centerZ + 192]

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

def erosion_sitk(sitk_mask,erosion_factor=3):

    # if not os.path.exists(file['recon_template']):
    #     out_path = file['masked_recon']+"/erosion_subject_mask.nii.gz"
    #     mask_path = file['recon_subject_mask']
    img = {}
    img['img'] = sitk.GetArrayFromImage(sitk_mask)
    img['spacing'] = sitk_mask.GetSpacing()
    img['origin'] = sitk_mask.GetOrigin()
    img['direction'] = sitk_mask.GetDirection()
    ndrr = ndimage.binary_dilation(img['img'],iterations=erosion_factor).astype(img['img'].dtype)
    out = sitk.GetImageFromArray(ndrr)
    out.SetSpacing(img['spacing'])
    out.SetOrigin(img['origin'])
    out.SetDirection(img['direction'])
    return out

def masking_sitk(sitk_img,sitk_mask,erosion_mask=False,erosion_factor=3):


    img = {}

    img['img'] = sitk.GetArrayFromImage(sitk_img)
    img['spacing'] = sitk_img.GetSpacing()
    img['origin'] = sitk_img.GetOrigin()
    img['direction'] = sitk_img.GetDirection()
    if erosion_mask:
        sitk_mask = erosion_sitk(sitk_mask,erosion_factor=erosion_factor)
    # else:
    #     itk_mask = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(sitk_mask)
    # out = img['img']
    out = img['img'] * mask
    out = sitk.GetImageFromArray(out)
    out.SetSpacing(img['spacing'])
    out.SetOrigin(img['origin'])
    out.SetDirection(img['direction'])
    return out


def find_files(file_dir):
    out_dir = os.path.join(file_dir,"out")
    ph.create_directory(out_dir)
    log = ""
    files = {}
    files['files'] = []
    files['mask_output'] = file_dir + "/seg/"
    files['masked_out'] = out_dir + "/masked_out/"
    ph.create_directory(os.path.dirname(files['mask_output']))
    ph.create_directory(os.path.dirname(files['masked_out']))
    imgs_dict = {}
    global global_path
    g = os.walk(file_dir)
    for path, dir_list, file_list in g:
        global_path = path
        if "seg" in path or "out" in path or "results" in path or "masked_out" in path  or "bestData" in path or "testset" in path or "tmp" in path or "adj" in path:
            continue
        for file_name in file_list:


            if file_name.endswith(".nii.gz"):
                # names = file_name.split('t2haste')
                names = re.findall(r"\d+\_\d+",file_name)
                # name = names[0]
                name = names[-1]
                if not name in imgs_dict.keys():
                    imgs = []
                    imgs.append(os.path.join(path,file_name))
                    imgs_dict[name] = imgs
                else:
                    imgs_dict[name].append(os.path.join(path,file_name))
    for name,file_list in imgs_dict.items():
        print(name)
        print(file_list)
        file = {}
        file['img'] = []
        file['mask'] = []
        file['masked_out'] = []
        for file_name in file_list:
            name = file_name.split('/')[-1]
            names = re.findall(r"\d+\_\d+",file_name)
            file['name'] = names[0]
            # file['name'] = name
            file['img'].append(file_name)
            file['mask'].append(os.path.join(files['mask_output'], name))
            file['masked_out'].append(out_dir + "/masked_out/" + name)
        file['img'].sort()
        file['mask'].sort()
        file['masked_out'].sort()
        print(file['img'])
        print(file['mask'])

        files['files'].append(file)

    return files


def read_preprocess(file_dir,init_HR=False,crop=False):
    np.set_printoptions(precision=3)

    tmp = os.path.join(file_dir, "tmp")
    ph.create_directory(tmp)
    init = os.path.join(tmp, "init")

    if crop:
        crop_path  = os.path.join(tmp, "crop")
        file_names = []
        filenames_masks = []
        files = os.listdir(file_dir)
        for file in files:
            if "nii.gz" not in file:
                continue
            if "crop" in file:
                continue
            # if "seg" in file or "out" in file or "masked_out" in file or "bestData" in file or "tmp" in file:
            #     continue
            file_path = os.path.join(crop_path, file)
            file_names.append(file_path)
            file_mask_path = os.path.join(crop_path, file)
            file_mask_path = file_mask_path.replace(".nii.gz", "_mask.nii.gz")
            filenames_masks.append(file_mask_path)
        file_names.sort()
        filenames_masks.sort()
        HR_volume = st.Stack.from_filename(os.path.join(crop_path, "HR_volume.nii.gz"),
                                           file_path_mask=None)
        # ----------Read data-------------
        ph.print_title("Read Data")
        suffix_mask = ""
        slice_thicknesses = None
        print(file_names)

        data_reader = dr.MultipleImagesReader(
            file_paths=file_names,
            # file_paths_masks=filenames_masks,
            file_paths_masks=None,
            suffix_mask=suffix_mask,
            dir_motion_correction=None,
            stacks_slice_thicknesses=slice_thicknesses,
            volume_motion_only=True,
        )
        data_reader.read_data()
        stacks = data_reader.get_data()




        for stack in stacks:
            stack_mask = sitk.GetArrayFromImage(stack.sitk)
            stack_mask[stack_mask > 0] = 1
            image_sitk_mask = sitk.GetImageFromArray(stack_mask)

            stack.sitk_mask = sitk.Cast(image_sitk_mask, sitk.sitkUInt8)
            try:
                # ensure mask occupies the same physical space
                stack.sitk_mask.CopyInformation(stack.sitk)
            except RuntimeError as e:
                raise IOError(
                    "Given image and its mask do not occupy the same space: %s" %
                    e.message)

        return stacks, HR_volume


    # seg = os.path.join(file_dir,"seg")
    file_names = []
    filenames_masks = []
    files = os.listdir(file_dir)
    for file in files:
        if "nii.gz" not in file:
            continue
        if "crop" in file:
            continue
        # if "seg" in file or "out" in file or "masked_out" in file or "bestData" in file or "tmp" in file:
        #     continue
        file_path = os.path.join(init, file)
        file_names.append(file_path)
        file_mask_path = os.path.join(init, file)
        file_mask_path = file_mask_path.replace(".nii.gz","_mask.nii.gz")
        filenames_masks.append(file_mask_path)
    file_names.sort()
    filenames_masks.sort()


    # ----------Read data-------------
    ph.print_title("Read Data")
    suffix_mask = ""
    slice_thicknesses = None
    print(file_names)

    dir_motion_correction = os.path.join(tmp,"motion correction")
    if not os.path.exists(dir_motion_correction):
        dir_motion_correction = None

    data_reader = dr.MultipleImagesReader(
        file_paths=file_names,
        file_paths_masks=filenames_masks,
        suffix_mask=suffix_mask,
        dir_motion_correction=dir_motion_correction,
        stacks_slice_thicknesses=slice_thicknesses,
        volume_motion_only=True,
        transform_inverse=True,
    )
    data_reader.read_data()
    stacks = data_reader.get_data()




    # ---------------------------Data Preprocessing---------------------------
    ph.print_title("Data Preprocessing")

    ph.print_info(
        "Searching for suitable target stack ... ", newline=False)
    target_stack_estimator = \
        ts_estimator.TargetStackEstimator.from_motion_score(
            file_paths=file_names,
            file_paths_masks=filenames_masks,
        )
    print("done")
    elapsed_time_target_stack = \
        target_stack_estimator.get_computational_time()
    ph.print_info(
        "Computational time for target stack selection: %s" % (elapsed_time_target_stack))
    target_stack_index = \
        target_stack_estimator.get_target_stack_index()
    target_stack = file_names[target_stack_index]
    ph.print_info("Chosen target stack: %s" % target_stack)
    # target_stack_index = 1
    # print("max value for target select ",sitk.GetArrayFromImage(stacks[0].sitk).max())

    segmentation_propagator = segprop.SegmentationPropagation(
        # registration_method=regflirt.FLIRT(use_verbose=args.verbose),
        # registration_method=niftyreg.RegAladin(use_verbose=False),
        dilation_radius=3,
        dilation_kernel="Ball",
    )

    data_preprocessing = dp.DataPreprocessing(
        stacks=stacks,
        segmentation_propagator=segmentation_propagator,
        # segmentation_propagator=None,
        use_cropping_to_mask=False,
        # use_N4BiasFieldCorrector=True,
        use_N4BiasFieldCorrector=True,
        target_stack_index=target_stack_index,
        boundary_i=10,
        boundary_j=10,
        boundary_k=0,
        unit="mm",
    )
    data_preprocessing.run()
    time_data_preprocessing = data_preprocessing.get_computational_time()

    # Get preprocessed stacks
    stacks = data_preprocessing.get_preprocessed_stacks()

    ph.print_title("Intensity Correction")
    intensity_corrector = ic.IntensityCorrection()
    intensity_corrector.use_individual_slice_correction(False)
    # intensity_corrector.use_individual_slice_correction(True)
    intensity_corrector.use_reference_mask(True)
    intensity_corrector.use_stack_mask(True)
    # intensity_corrector.use_verbose(False)
    intensity_corrector.use_verbose(True)

    for i, stack in enumerate(stacks):
        if i == target_stack_index:
            ph.print_info("Stack %d (%s): Reference image. Skipped." % (
                i + 1, stack.get_filename()))
            continue
        else:
            ph.print_info("Stack %d (%s): Intensity Correction ... " % (
                i + 1, stack.get_filename()), newline=False)
        intensity_corrector.set_stack(stack)
        intensity_corrector.set_reference(
            stacks[target_stack_index].get_resampled_stack(
                resampling_grid=stack.sitk,
                interpolator="NearestNeighbor",
                # interpolator="Linear",
            ))
        intensity_corrector.run_linear_intensity_correction()
        # intensity_corrector.run_affine_intensity_correction()
        stacks[i] = intensity_corrector.get_intensity_corrected_stack()


    if init_HR:
        HR_volume = stacks[target_stack_index].get_isotropically_resampled_stack(
            # )
            resolution=0.8)
        joint_image_mask_builder = imb.JointImageMaskBuilder(
            stacks=stacks,
            target=HR_volume,
            dilation_radius=1,
        )
        joint_image_mask_builder.run()
        HR_volume = joint_image_mask_builder.get_stack()
        ph.print_info(
            "Isotropic reconstruction space is centered around "
            "joint stack masks. ")

        # Crop to space defined by mask (plus extra margin)
        # HR_volume = HR_volume.get_cropped_stack_based_on_mask(
        #     boundary_i=0,
        #     boundary_j=0,
        #     boundary_k=0,
        #     unit="mm",
        # )

        dir_output = os.path.join(file_dir+"/tmp/", "init")
        ph.create_directory(dir_output)
        HR_volume.write(dir_output,
                        filename="HR_volume",
                        write_stack=True,
                        write_mask=False,
                        write_slices=False,
                        write_transforms=False,
                        suffix_mask="_mask",
                        write_transforms_history=False)


        print("HR_volume size",HR_volume.sitk.GetSize())
        image_crop_sitk,mask_crop_sitk = get_cropped_sitk(HR_volume,size=128)
        # image_crop_sitk,mask_crop_sitk = get_cropped_sitk(HR_volume,size=104)
        # image_crop_sitk,mask_crop_sitk = pad_image(HR_volume,size=160)
        HR_volume = st.Stack.from_sitk_image(
            image_sitk=image_crop_sitk,
            slice_thickness=HR_volume.get_slice_thickness(),
            filename=HR_volume._filename,
            image_sitk_mask=mask_crop_sitk)
        print("crop HR_volume size", HR_volume.sitk.GetSize())

        # ---------avoid slice sitk +1 ------
        stacks_sda = []
        for stack in stacks:
            stack_sda = st.Stack.from_stack(stack)
            stacks_sda.append(stack_sda)


        # ph.print_subtitle("SDA Approximation Image")
        # # HR_volume = st.Stack.from_filename(template_path)
        # SDA = sda.ScatteredDataApproximation(
        #     stacks_sda, HR_volume, sigma=1)
        # SDA.run()
        # HR_volume = SDA.get_reconstruction()
        #
        #
        # ph.print_subtitle("SDA Approximation Image Mask")
        # SDA = sda.ScatteredDataApproximation(
        #     stacks_sda, HR_volume, sigma=1, sda_mask=True)
        # SDA.run()
        # # HR volume contains updated mask based on SDA
        # HR_volume = SDA.get_reconstruction()


        dir_output = os.path.join(file_dir+"/tmp/", "init")
        ph.create_directory(dir_output)
        HR_volume.write(dir_output,
                        filename="crop_HR_volume",
                         write_stack=True,
                         write_mask=False,
                         write_slices=False,
                         write_transforms=False,
                         suffix_mask="_mask",
                         write_transforms_history=False)








    # Identify and reject outliers
    # ph.print_subtitle("Eliminate slice outliers (%s < %g)" % (
    #     "NCC", 0.3))
    # outlier_rejector = outre.OutlierRejector(
    #     stacks=stacks,
    #     reference=HR_volume,
    #     threshold=0.3,
    #     measure="NCC",
    #     verbose=True,
    # )
    # outlier_rejector.run()
    # stacks = outlier_rejector.get_stacks()

    # --------maksing--------
    HR_volume.sitk = masking_sitk(HR_volume.sitk, HR_volume.sitk_mask,erosion_mask=True,erosion_factor=3)
    for stack in stacks:
        stack.sitk = masking_sitk(stack.sitk, stack.sitk_mask,erosion_mask=False,erosion_factor=0)
        for slice in stack.get_slices():
            slice.sitk = masking_sitk(slice.sitk, slice.sitk_mask,erosion_mask=False,erosion_factor=1)
            if sitk.GetArrayFromImage(slice.sitk).max() < 0.1:
                stack.delete_slice(slice)

    #-----------pad-to-same-size-----------
    x1,y1,z1 = stacks[0].sitk.GetSize()
    x2,y2,z2 = stacks[1].sitk.GetSize()
    x3,y3,z3 = stacks[2].sitk.GetSize()
    xmax = np.max([x1,x2,x3])
    ymax = np.max([y1,y2,y3])
    for stack in stacks:
        size = stack.sitk.GetSize()

        pad_x = xmax - size[0]
        pad_y = ymax - size[1]
        upadding = [0] * 3

        lpadding = [0] * 3
        padz1 = 0 // 2 + 0 % 2
        padz2 = 0 // 2
        pady1 = pad_y // 2 + pad_y % 2
        pady2 = pad_y // 2
        padx1 = pad_x // 2 + pad_x % 2
        padx2 = pad_x // 2

        upadding[0] = int(padx1)
        upadding[1] = int(pady1)
        upadding[2] = int(padz1)
        lpadding[0] = int(padx2)
        lpadding[1] = int(pady2)
        lpadding[2] = int(padz2)

        padded_image = sitk.ConstantPad(stack.sitk, upadding, lpadding, 0.0)
        padded_mask = sitk.ConstantPad(stack.sitk_mask, upadding, lpadding, 0.0)

        stack.sitk = padded_image
        stack.sitk_mask = padded_mask


        # stack.sitk = masking_sitk(stack.sitk, stack.sitk_mask, erosion_mask=False, erosion_factor=1)
        # stack.sitk = masking_sitk(stack.sitk, stack.sitk_mask, erosion_mask=True, erosion_factor=3)
        # for slice in stack.get_slices():
        #     size = slice.sitk.GetSize()
        #
        #     pad_x = xmax - size[0]
        #     pad_y = ymax - size[1]
        #     upadding = [0] * 3
        #
        #     lpadding = [0] * 3
        #     padz1 = 0 // 2 + 0 % 2
        #     padz2 = 0 // 2
        #     pady1 = pad_y // 2 + pad_y % 2
        #     pady2 = pad_y // 2
        #     padx1 = pad_x // 2 + pad_x % 2
        #     padx2 = pad_x // 2
        #
        #     upadding[0] = int(padx1)
        #     upadding[1] = int(pady1)
        #     upadding[2] = int(padz1)
        #     lpadding[0] = int(padx2)
        #     lpadding[1] = int(pady2)
        #     lpadding[2] = int(padz2)
        #
        #
        #     padded_image = sitk.ConstantPad(slice.sitk, upadding, lpadding, 0.0)
        #     padded_mask = sitk.ConstantPad(slice.sitk_mask, upadding, lpadding, 0.0)
        #
        #     slice.sitk = padded_image
        #     slice.sitk_mask = padded_mask



    # -----------save correction------------
    dir_output = os.path.join(tmp, "crop")
    ph.create_directory(dir_output)
    HR_volume.write(dir_output,HR_volume.get_filename())
    for i,stack in enumerate(stacks):
        file_name = stack.get_filename()
        stack.write(dir_output,
                    filename=file_name,
                    write_stack=True,
                    write_mask=True,
                    write_slices=False,
                    write_transforms=False,
                    suffix_mask="_mask",
                    write_transforms_history=False)
        stacks[i] = st.Stack.from_filename(file_path= "%s.nii.gz" % os.path.join(dir_output,file_name),
                                           file_path_mask="%s_mask.nii.gz" % os.path.join(dir_output,file_name))




    return stacks, HR_volume


if __name__ == '__main__':
    data_path = ""

    files_train = find_files(data_path)
    train_stacks, _ = read_preprocess(data_path,init_HR=True,  crop=False)