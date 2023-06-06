import math
import math as m
import os

# built-in modules
import sys

import cv2
import numpy as np
import svm
import svmutil
from exif import Image as exif_image
from libsvm import svmutil as svmutil
from matplotlib import pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS

# for gamma function, called
from scipy.special import gamma as tgamma
from svm import *

import exif_utils
from print_utils import Printer

plt.rcParams.update({"font.size": 10})


def print_exif_dict(exif_dict):
    for k, v in exif_dict.items():
        if v["raw"] is not None:
            print(k)
            print("-" * len(k))
            print("    tag:       {}".format(v["tag"]))
            print("    raw:       {}".format(v["raw"]))
            print("    processed: {}\n".format(v["processed"]))


# AGGD fit model, takes input as the MSCN Image / Pair-wise Product
def AGGDfit(structdis):
    # variables to count positive pixels / negative pixels and their squared sum
    poscount = 0
    negcount = 0
    possqsum = 0
    negsqsum = 0
    abssum = 0

    poscount = len(structdis[structdis > 0])  # number of positive pixels
    negcount = len(structdis[structdis < 0])  # number of negative pixels

    # calculate squared sum of positive pixels and negative pixels
    possqsum = np.sum(np.power(structdis[structdis > 0], 2))
    negsqsum = np.sum(np.power(structdis[structdis < 0], 2))

    # absolute squared sum
    abssum = np.sum(structdis[structdis > 0]) + np.sum(-1 * structdis[structdis < 0])

    # calculate left sigma variance and right sigma variance
    lsigma_best = np.sqrt((negsqsum / negcount))
    rsigma_best = np.sqrt((possqsum / poscount))

    gammahat = lsigma_best / rsigma_best

    # total number of pixels - totalcount
    totalcount = structdis.shape[1] * structdis.shape[0]

    rhat = m.pow(abssum / totalcount, 2) / ((negsqsum + possqsum) / totalcount)
    rhatnorm = (
        rhat
        * (m.pow(gammahat, 3) + 1)
        * (gammahat + 1)
        / (m.pow(m.pow(gammahat, 2) + 1, 2))
    )

    prevgamma = 0
    prevdiff = 1e10
    sampling = 0.001
    gam = 0.2

    # vectorized function call for best fitting parameters
    vectfunc = np.vectorize(func, otypes=[float], cache=False)

    # calculate best fit params
    gamma_best = vectfunc(gam, prevgamma, prevdiff, sampling, rhatnorm)

    return [lsigma_best, rsigma_best, gamma_best]


def func(gam, prevgamma, prevdiff, sampling, rhatnorm):
    while gam < 10:
        r_gam = tgamma(2 / gam) * tgamma(2 / gam) / (tgamma(1 / gam) * tgamma(3 / gam))
        diff = abs(r_gam - rhatnorm)
        if diff > prevdiff:
            break
        prevdiff = diff
        prevgamma = gam
        gam += sampling
    gamma_best = prevgamma
    return gamma_best


def compute_features(img):
    scalenum = 2
    feat = []
    # make a copy of the image
    im_original = img.copy()

    # scale the images twice
    for itr_scale in range(scalenum):
        im = im_original.copy()
        # normalize the image
        im = im / 255.0

        # calculating MSCN coefficients
        mu = cv2.GaussianBlur(im, (7, 7), 1.166)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(im * im, (7, 7), 1.166)
        sigma = abs(sigma - mu_sq) ** 0.5

        # structdis is the MSCN image
        structdis = im - mu
        structdis /= sigma + 1.0 / 255

        # calculate best fitted parameters from MSCN image
        best_fit_params = AGGDfit(structdis)
        # unwrap the best fit parameters
        lsigma_best = best_fit_params[0]
        rsigma_best = best_fit_params[1]
        gamma_best = best_fit_params[2]

        # append the best fit parameters for MSCN image
        feat.append(gamma_best)
        feat.append((lsigma_best * lsigma_best + rsigma_best * rsigma_best) / 2)

        # shifting indices for creating pair-wise products
        shifts = [[0, 1], [1, 0], [1, 1], [-1, 1]]  # H V D1 D2

        for itr_shift in range(1, len(shifts) + 1):
            OrigArr = structdis
            reqshift = shifts[itr_shift - 1]  # shifting index

            # create transformation matrix for warpAffine function
            M = np.float32([[1, 0, reqshift[1]], [0, 1, reqshift[0]]])
            ShiftArr = cv2.warpAffine(
                OrigArr, M, (structdis.shape[1], structdis.shape[0])
            )

            Shifted_new_structdis = ShiftArr
            Shifted_new_structdis = Shifted_new_structdis * structdis
            # shifted_new_structdis is the pairwise product
            # best fit the pairwise product
            best_fit_params = AGGDfit(Shifted_new_structdis)
            lsigma_best = best_fit_params[0]
            rsigma_best = best_fit_params[1]
            gamma_best = best_fit_params[2]

            constant = m.pow(tgamma(1 / gamma_best), 0.5) / m.pow(
                tgamma(3 / gamma_best), 0.5
            )
            meanparam = (
                (rsigma_best - lsigma_best)
                * (tgamma(2 / gamma_best) / tgamma(1 / gamma_best))
                * constant
            )

            # append the best fit calculated parameters
            feat.append(gamma_best)  # gamma best
            feat.append(meanparam)  # mean shape
            feat.append(m.pow(lsigma_best, 2))  # left variance square
            feat.append(m.pow(rsigma_best, 2))  # right variance square

        # resize the image on next iteration
        im_original = cv2.resize(
            im_original, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC
        )
    return feat


# function to calculate BRISQUE quality score
# takes input of the image path
def test_measure_BRISQUE(imgPath):
    # read image from given path
    dis = cv2.imread(imgPath, 1)
    if dis is None:
        print("Wrong image path given")
        print("Exiting...")
        sys.exit(0)
    # convert to gray scale
    dis = cv2.cvtColor(dis, cv2.COLOR_BGR2GRAY)

    # compute feature vectors of the image
    features = compute_features(dis)

    # rescale the brisqueFeatures vector from -1 to 1
    x = [0]

    # pre loaded lists from C++ Module to rescale brisquefeatures vector to [-1, 1]
    min_ = [
        0.336999,
        0.019667,
        0.230000,
        -0.125959,
        0.000167,
        0.000616,
        0.231000,
        -0.125873,
        0.000165,
        0.000600,
        0.241000,
        -0.128814,
        0.000179,
        0.000386,
        0.243000,
        -0.133080,
        0.000182,
        0.000421,
        0.436998,
        0.016929,
        0.247000,
        -0.200231,
        0.000104,
        0.000834,
        0.257000,
        -0.200017,
        0.000112,
        0.000876,
        0.257000,
        -0.155072,
        0.000112,
        0.000356,
        0.258000,
        -0.154374,
        0.000117,
        0.000351,
    ]

    max_ = [
        9.999411,
        0.807472,
        1.644021,
        0.202917,
        0.712384,
        0.468672,
        1.644021,
        0.169548,
        0.713132,
        0.467896,
        1.553016,
        0.101368,
        0.687324,
        0.533087,
        1.554016,
        0.101000,
        0.689177,
        0.533133,
        3.639918,
        0.800955,
        1.096995,
        0.175286,
        0.755547,
        0.399270,
        1.095995,
        0.155928,
        0.751488,
        0.402398,
        1.041992,
        0.093209,
        0.623516,
        0.532925,
        1.042992,
        0.093714,
        0.621958,
        0.534484,
    ]

    # append the rescaled vector to x
    for i in range(0, 36):
        min = min_[i]
        max = max_[i]
        x.append(-1 + (2.0 / (max - min) * (features[i] - min)))

    # load model
    model = svmutil.svm_load_model("./bin/allmodel")

    # create svm node array from python list
    x, idx = gen_svm_nodearray(x[1:], isKernel=(model.param.kernel_type == PRECOMPUTED))
    x[36].index = -1  # set last index to -1 to indicate the end.

    # get important parameters from model
    svm_type = model.get_svm_type()
    is_prob_model = model.is_probability_model()
    nr_class = model.get_nr_class()

    if svm_type in (ONE_CLASS, EPSILON_SVR, NU_SVC):
        # here svm_type is EPSILON_SVR as it's regression problem
        nr_classifier = 1
    dec_values = (c_double * nr_classifier)()

    # calculate the quality score of the image using the model and svm_node_array
    qualityscore = svmutil.libsvm.svm_predict_probability(model, x, dec_values)
    return round(qualityscore, 2)


class App:
    def set_scale(self, val):
        self.hist_scale = val

    def run(self, fname):
        hsv_map = np.zeros((180, 256, 3), np.uint8)
        h, s = np.indices(hsv_map.shape[:2])
        hsv_map[:, :, 0] = h
        hsv_map[:, :, 1] = s
        hsv_map[:, :, 2] = 255
        hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)
        # cv.imshow('hsv_map', hsv_map)
        cv2.imwrite("./images/hsv_map.JPG", hsv_map)

        cv2.namedWindow(fname, cv2.WINDOW_FULLSCREEN)
        self.hist_scale = 10

        cv2.createTrackbar("scale", "hist", self.hist_scale, 32, self.set_scale)

        # try:
        #     fn = sys.argv[1]
        # except:
        #     fn = 0
        # cam = video.create_capture(fn, fallback='synth:bg=baboon.jpg:class=chess:noise=0.05')
        cam = cv2.imread(fname)

        while True:
            # _flag, frame = cam.read()
            frame = cam
            # cv.imshow('camera', frame)

            small = cv2.pyrDown(frame)

            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            dark = hsv[..., 2] < 32
            hsv[dark] = 0
            h = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            h = np.clip(h * 0.005 * self.hist_scale, 0, 1)
            vis = hsv_map * h[:, :, np.newaxis] / 255.0
            scale_percent = 100  # percent of original size
            width = int(vis.shape[1] * scale_percent / 100)
            height = int(vis.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            resized = cv2.resize(vis, dim, interpolation=cv2.INTER_AREA)
            img = cv2.convertScaleAbs(resized, alpha=(255.0))
            cv2.imwrite("./images/newimage" + fname + ".JPG", img)
            number_of_black_pix = np.sum(img == 0)
            number_of_color_pix = np.sum(img != 0)
            totpix = number_of_black_pix + number_of_color_pix
            sz = img.size
            # if(sz==totpix): print("Yes")
            # print(number_of_black_pix,number_of_color_pix,sz)
            # cv2.imshow('hist', resized)
            ratio = round(number_of_color_pix / totpix, 4)
            return ratio, str(number_of_color_pix) + "/" + str(totpix)
            # break
            # ch = cv2.waitKey(1)
            # if ch == 27:
            #     img = cv2.convertScaleAbs(resized, alpha=(255.0))
            #     cv2.imwrite("newimage"+fname+".JPG", img)
            #     break

        # cv2.destroyAllWindows()
        # print('Done')


def getStats(file):
    """_summary_
    34853 GPSInfo
    296 ResolutionUnit
    34665 ExifOffset
    271 Make
    272 Model
    305 Software
    274 Orientation
    306 DateTime
    531 YCbCrPositioning
    282 XResolution
    283 YResolution
    36864 ExifVersion
    37121 ComponentsConfiguration
    37122 CompressedBitsPerPixel
    36867 DateTimeOriginal
    36868 DateTimeDigitized
    37380 ExposureBiasValue
    37381 MaxApertureValue
    37383 MeteringMode
    37384 LightSource
    37385 Flash
    37386 FocalLength
    37510 UserComment
    40961 ColorSpace
    40962 ExifImageWidth
    40965 ExifInteroperabilityOffset
    41990 SceneCaptureType
    37520 SubsecTime
    37521 SubsecTimeOriginal
    37522 SubsecTimeDigitized
    40963 ExifImageHeight
    41996 SubjectDistanceRange
    41495 SensingMethod
    41728 FileSource
    33434 ExposureTime
    33437 FNumber
    41729 SceneType
    34850 ExposureProgram
    41730 CFAPattern
    41985 CustomRendered
    34855 ISOSpeedRatings
    41986 ExposureMode
    40960 FlashPixVersion
    34864 SensitivityType
    41987 WhiteBalance
    41988 DigitalZoomRatio
    41989 FocalLengthIn35mmFilm
    41991 GainControl
    41992 Contrast
    41993 Saturation
    41994 Sharpness
    37500 MakerNote
    Args:
        file (_type_): _description_
    """

    # try:

    #     filepath = file

    #     exif_dict = exif_utils.generate_exif_dict(filepath)

    #     print_exif_dict(exif_dict)

    # except IOError as ioe:

    #     print(ioe)
    exif_img = exif_image(file)
    list_exif_available = exif_img.list_all()
    # for attr in list_exif_available:
    # print(attr,":",exif_img.get(attr))

    w = exif_img.pixel_x_dimension
    h = exif_img.pixel_y_dimension
    model = exif_img.model
    focallength = exif_img.focal_length
    shutterspeed = exif_img.exposure_time
    aperture = exif_img.f_number
    iso = exif_img.photographic_sensitivity
    val = (100 * (aperture) ** 2) / (iso * (shutterspeed))

    EV = math.log(val, 2)
    EV = round(EV, 2)

    # sys.exit()

    # print(file)
    img = cv2.imread(file)

    # image=Image.open(file)
    # #print(image.getexif())
    # exifdata = image._getexif()
    # # print(exifdata)
    # # tagnames = []
    # values = []
    # image_exif_data = {}
    # for tagid in exifdata:
    #     # getting the tag name instead of tag id
    #     tagname = TAGS.get(tagid, tagid)
    #     # EV = log2(100 * aperture**2 / (ISO * shutter speed)) : https://www.omnicalculator.com/other/exposure
    #      #shutterspeed,ISO,...,aperture
    #     if(tagname in ["Model","ExifImageWidth","ExifImageHeight","ExposureTime","ISOSpeedRatings","FocalLength","FNumber"]):
    #     # # tagnames.append(tagname)
    #     # # passing the tagid to get its respective value
    #         image_exif_data.update({tagname:exifdata.get(tagid)})
    #         value = exifdata.get(tagid)
    #         values.append(value)
    #     #     # printing the final result
    #         print(f"{tagname:25}: {value}")

    # print(values)
    # print(image_exif_data)

    # w = image_exif_data.get("ExifImageWidth",{})
    # h = image_exif_data.get("ExifImageHeight",{})

    # sys.exit()

    # sys.exit()
    check = 1
    if check:
        # cv2.imshow("as",img)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tonalrange, text_tonalrange = App().run(file)
        blurval = blurCheck(img)
        color = ("b", "g", "r")
        qualityscore = test_measure_BRISQUE(file)

        # print ("Score of the given image: {}".format(qualityscore))
        text = ""
        vals = ""
        tagname = "File name"
        print(f"{tagname:25}: {file}")
        text += tagname + "\n"
        vals += str(file) + "\n"

        tagname = "Device Name"
        print(f"{tagname:25}: {model}")
        text += tagname + "\n"
        vals += str(model) + "\n"

        tagname = "Width"
        print(f"{tagname:25}: {w}")
        text += tagname + "\n"
        vals += str(w) + "\n"

        tagname = "Height"
        print(f"{tagname:25}: {h}")
        text += tagname + "\n"
        vals += str(h) + "\n"

        tagname = "Aperture"
        print(f"{tagname:25}: {aperture}")
        text += tagname + "\n"
        vals += str(aperture) + "\n"

        tagname = "ISO"
        print(f"{tagname:25}: {iso}")
        text += tagname + "\n"
        vals += str(iso) + "\n"

        tagname = "ShutterSpeed"
        print(f"{tagname:25}: {shutterspeed}")
        text += tagname + "\n"
        vals += str(shutterspeed) + "\n"

        tagname = "FocalLength"
        print(f"{tagname:25}: {focallength}")
        text += tagname + "\n"
        vals += str(focallength) + " mm\n"

        tagname = "Blur Value"
        print(f"{tagname:25}: {blurval}")
        text += tagname + "\n"
        vals += str(blurval) + "\n"

        tagname = "Score Value/Distortion"
        print(f"{tagname:25}: {qualityscore}")
        text += tagname + "\n"
        vals += str(qualityscore) + "\n"

        tagname = "Exposure value"
        print(f"{tagname:25}: {EV}")
        text += tagname + "\n"
        vals += str(EV) + "\n"

        text += "Color Range\n"
        vals += str(tonalrange) + " , " + text_tonalrange + "\n"

        inner1 = [["innerA"], ["innerB"]]
        inner2 = [["innerC"], ["innerD"]]

        outer = [["upper left", inner1], ["lower left", inner2]]
        # outer = [['ul','ur'],['ml','mr'],['ll','lr']]
        # outer = [['ul',['ml','mr']],['ur',['ll','lr']]]

        fig, axd = plt.subplot_mosaic(outer, constrained_layout=True)
        flag = 0
        for k in axd:
            annotate_axes(img, grey, axd[k], flag, file, text, vals)
            flag = flag + 1

        # plt.show()
        # fig = plt.gcf()
        # plt.show()
        # plt.pause(1)
        # plt.close("all")
        fig.set_size_inches((16, 12), forward=False)

        directory = "output"

        # Check if the directory exists
        if not os.path.exists(directory):
            # If it doesn't exist, create it
            os.makedirs(directory)

        fig.savefig(
            directory + "/output" + file + ".png", dpi=500
        )  # Change is over here
        # plt.savefig('./output/output'+file+'.png', bbox_inches='tight',dpi=600) # bbox_inches removes extra white spaces


def blurCheck(img):
    val = cv2.Laplacian(img, cv2.CV_64F).var()  # 100 threshold
    # print ("Blur of the given image: {}".format(val))
    return round(val, 2)


def annotate_axes(img, grey, ax, flag, file, text, vals, fontsize=14):
    # ax.text(0.5, 0.5, text, transform=ax.transAxes,ha="center", va="center", fontsize=fontsize, color="darkgrey")

    if flag == 1:
        im = plt.imread("./images/newimage" + file + ".JPG")
        # plt.subplot(2, 2, 3)
        im = ax.imshow(im)
        # plt.xlabel("Blur: "+str(blurval))
        ax.set_title("Tonal Range", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)

    if flag == 2:
        hsv_map = plt.imread("./images/hsv_map.JPG")
        # plt.subplot(2, 2, 4)
        # plt.xlabel("Qscore: "+str(qualityscore))
        hsv_map = ax.imshow(hsv_map)
        ax.set_title("HSV Map", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)

    if flag == 4:
        color = ("b", "g", "r")
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            # plt.subplot(2, 2, 1)
            # plt.ylabel("No. of Pixels")
            # plt.xlabel("Intensity")
            ax.plot(histr, color=col)
        ax.set_xlabel("Intensity", fontsize=fontsize)
        ax.set_ylabel("# Pixels", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)
        ax.set_title("Color Histogram", fontsize=fontsize)

    if flag == 5:
        dst = cv2.calcHist([grey], [0], None, [256], [0, 256])
        # plt.subplot(2, 2, 2)
        ax.plot(dst, color="black")
        ax.set_xlabel("Intensity", fontsize=fontsize)
        ax.set_ylabel("# Pixels", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)
        ax.set_title("Histogram", fontsize=fontsize)

    if flag == 0:
        img = cv2.imread(file)
        im = ax.imshow(img[:, :, ::-1])
        ax.set_title("Image", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)

    if flag == 3:
        # img = cv2.imread(file)
        # im = ax.imshow(img[:,:,::-1])
        ax.text(
            0.1,
            0.5,
            text,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=fontsize,
            color="black",
        )
        ax.text(
            0.5,
            0.5,
            vals,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=fontsize,
            color="black",
        )
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title("Image Details", fontsize=fontsize)


def main():
    # folder_dir = "/Users/vishalgattani/Desktop/624/camera_images"
    # dir_name = "/captured_images/"

    folder_dir = os.getcwd()
    folder_dir = "./shortlisted_images/"

    files = []

    # try:
    for file in os.listdir(folder_dir):
        # check if the image ends with png
        if file.endswith(".JPG") or file.endswith(".jpg"):
            # print(file)
            files.append(file)
            # getStats(file)
            # print("="*50)

    # files = ["DSC_0018.JPG","IMG_3748.JPG"]
    # files =['IMG_3772.JPG']

    files.sort()
    print(files)
    print(len(files))
    print("=" * 50)
    for i in range(len(files)):
        Printer.green(files[i])
        getStats(files[i])
        Printer.red(
            len(files) - i - 1,
            " Files remaining...",
        )


if __name__ == "__main__":
    main()
