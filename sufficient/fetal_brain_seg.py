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
from monaifbs.src.inference.monai_dynunet_inference import run_inference


def erosion(mask_path):
    out_path = mask_path.replace(".nii.gz","_erosion_mask.nii.gz")
    # if not os.path.exists(file['recon_template']):
    #     out_path = file['masked_recon']+"/erosion_subject_mask.nii.gz"
    #     mask_path = file['recon_subject_mask']
    img = {}
    itk_img = sitk.ReadImage(mask_path)
    img['img'] = sitk.GetArrayFromImage(itk_img)
    img['spacing'] = itk_img.GetSpacing()
    img['origin'] = itk_img.GetOrigin()
    img['direction'] = itk_img.GetDirection()
    ndrr = ndimage.binary_dilation(img['img'],iterations=1).astype(img['img'].dtype)
    out = sitk.GetImageFromArray(ndrr)
    out.SetSpacing(img['spacing'])
    out.SetOrigin(img['origin'])
    out.SetDirection(img['direction'])
    sitk.WriteImage(out, out_path)
    return out

def masking_one_image(file,erosion_mask=False):
    for mri_path,mask_path,out_path in zip(file['img'],file['mask'],file['masked_out']):
        print(mri_path)
        img = {}
        itk_img = sitk.ReadImage(mri_path)
        img['img'] = sitk.GetArrayFromImage(itk_img)
        img['spacing'] = itk_img.GetSpacing()
        img['origin'] = itk_img.GetOrigin()
        img['direction'] = itk_img.GetDirection()
        if erosion_mask:
            itk_mask = erosion(mask_path)
        else:
            itk_mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(itk_mask)
        out = img['img']
        # out = img['img'] * mask
        out = sitk.GetImageFromArray(out)
        out.SetSpacing(img['spacing'])
        out.SetOrigin(img['origin'])
        out.SetDirection(img['direction'])
        sitk.WriteImage(out, out_path)



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


def fetal_brain_seg(files,config_file = None):


    # check existence of config file and read it
    if config_file is None:
        config_file = os.path.join(*[os.path.dirname(monaifbs.__file__),
                                     "config", "monai_dynUnet_inference_config.yml"])
    if not os.path.isfile(config_file):
        raise FileNotFoundError('Expected config file: {} not found'.format(config_file))
    with open(config_file) as f:
        print("*** Config file")
        print(config_file)
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config['inference']['model_to_load'] == "default":
        config['inference']['model_to_load'] = os.path.join(*[os.path.dirname(monaifbs.__file__),
                                                              "models", "checkpoint_dynUnet_DiceXent.pt"])


    # loop over all input files and run inference for each of them

    for file in files['files']:
        print(file)
        # set the output folder and add to the config file
        out_folder = os.path.dirname(files['mask_output'])
        if not out_folder:
            out_folder = os.getcwd()
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        config['output'] = {'out_postfix': 'seg', 'out_dir': out_folder}

        for img,mask in zip(file['img'],file['mask']):

            # run inference
            run_inference(input_data=img, config_info=config)

            # recover the filename generated by the inference code (as defined by MONAI output)
            img_filename = os.path.basename(img)
            flag_zip = 0
            if 'gz' in img_filename:
                img_filename = img_filename[:-7]
                flag_zip = 1
            else:
                img_filename = img_filename[:-4]
            out_filename = img_filename + '_' + config['output']['out_postfix'] + '.nii.gz' if flag_zip \
                else img_filename + '_' + config['output']['out_postfix'] + '.nii'
            out_filename = os.path.join(*[out_folder, img_filename, out_filename])

            # check existence of segmentation file
            if not os.path.exists(out_filename):
                raise FileNotFoundError("Network output file {} not found, "
                                        "check if the segmentation pipeline has failed".format(out_filename))

            # rename file with the indicated output name
            os.rename(out_filename, mask)
            if os.path.exists(mask):
                os.rmdir(os.path.join(out_folder, img_filename))
        masking_one_image(file, erosion_mask=False)

if __name__ == '__main__':
    data_path = ""

    files_train = find_files(data_path)
    fetal_brain_seg(files_train)