import cv2
import scipy.misc

import SimpleITK as sitk #reading MR images

import glob


readfolderT = glob.glob('~/Downloads/ADAM/*/pre/TOF.nii.gz')
readfolderL = glob.glob('~/Downloads/ADAM/*/aneurysms.nii.gz')


TrainingImagesList = []
TrainingLabelsList = []


for i in range(len(readfolderT)):
    y_folder = readfolderT[i]
    yread = sitk.ReadImage(y_folder)
    yimage = sitk.GetArrayFromImage(yread)
    x = yimage[:184,:232,112:136]
    x = scipy.rot90(x)
    x = scipy.rot90(x)
    for j in range(x.shape[2]):
        TrainingImagesList.append((x[:184,:224,j]))

for i in range(len(readfolderL)):
    y_folder = readfolderL[i]
    yread = sitk.ReadImage(y_folder)
    yimage = sitk.GetArrayFromImage(yread)
    x = yimage[:184,:232,112:136]
    x = scipy.rot90(x)
    x = scipy.rot90(x)
    for j in range(x.shape[2]):
        TrainingLabelsList.append((x[:184,:224,j]))

for i in range(len(TrainingImagesList)):

    xchangeL = TrainingImagesList[i]
    xchangeL = cv2.resize(xchangeL,(128,128))
    scipy.misc.imsave('~/Downloads/ADAM2DImgs/'+str(i)+'.png',xchangeL)

for i in range(len(TrainingLabelsList)):

    xchangeL = TrainingLabelsList[i]
    xchangeL = cv2.resize(xchangeL,(128,128))
    scipy.misc.imsave('~/Downloads/ADAM2DLabels/'+str(i)+'.png',xchangeL)