# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 09:44:10 2020

@author: chv2szh
"""
import os, sys
import warnings
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
import platform

g_lc_annotation_path    = os.getcwd()
g_lc_train_file         = "./train.txt"
g_lc_image_path         = "./"

g_lc_color = [
    "firebrick",
    "cyan",
    "steelblue",#2
    "fuchsia",
    "lime",
    "green",#5
    "khaki",
    "violet",
    "tan"]

#defined in odet.namess
g_lc_cate_odet = {
    "other"         :0,
    "car"           :1,
    "cyclist"       :2,
    "e-scooter"     :3,
    "pedestrian"    :4,
    "traffic-light" :5,
    "traffic-sign"  :6,
    "truck"         :7}

#mapping bosch type name from label tools to SRD
g_lc_cate_gen3 = {
    "other"         :0,
    "car"           :1,
    "van"           :1, #mapping to car
    "cyclist"       :2,
    "motorcyclist"  :3,
    "motorbike"     :3,
    "pedestrian"    :4,
    "truck"         :7,
    "bus"           :7} #mapping to truck

g_lc_debug      = False

def get_files(path, suffix=".json"):
    '''
    obtain the json file list in the path recursively
    :param path:
    :param suffix:
    :return: json file list
    '''
    file_list = []
    try:
        if os.path.exists(path):
            # browse each folder
            for home, dirs, files in os.walk(path):
                for file in files:
                    if suffix in file:
                        file_path = os.path.join(home, file)
                        print("{}".format(file_path))
                        file_list.append(file_path)
    except Exception as e:
        print(e)
    return file_list

def darw_image_rect(image_file, boxes):
    def get_key(dict, value):
        return [k for k, v in dict.items() if v == value]

    image = Image.open(image_file)
    draw = ImageDraw.Draw(image)

    for i in range(len(boxes)):
        category =  get_key(g_lc_cate_odet, int(boxes[i][0]))
        minx = int(boxes[i][1])
        miny = int(boxes[i][2])
        maxx = int(boxes[i][3])
        maxy = int(boxes[i][4])
        color = g_lc_color[int(boxes[i][0])]
        width = 2
        draw.line((minx, miny, minx, maxy), color, width=width)
        draw.line((minx, miny, maxx, miny), color, width=width)
        draw.line((maxx, miny, maxx, maxy), color, width=width)
        draw.line((minx, maxy, maxx, maxy), color, width=width)

        #font = ImageFont.truetype("consola.ttf", 18, encoding="unic")
        font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), 18, encoding="unic")
        #draw.rectangle((minx, miny, maxx, maxy), 'black', 'cyan')
        draw.text((minx, miny), category[0], color, font)

    image.show()

class CLabelConverter(object):
    def __init__(self, annotation_path=None, train_file=None, image_path=None):
        '''
        :param annotation_path: the path contains the label files(json or xml)
        :param train_file:      the path to save the train file
        :param image_path:      the path contains the images
        '''
        self.annotation_path    = annotation_path if annotation_path != None else g_lc_annotation_path
        self.train_file         = train_file if train_file != None else g_lc_train_file
        self.image_path         = image_path if image_path != None else g_lc_image_path
        pass

    def __encode_box(self, upLeft, lowRight, category, using_float=False):
        '''
        write the related information into text for training and validation
        the format of the text is:
        index image_path image_w image_h object_type_1 minx_1 miny_1 maxx_1 maxy_1 ... object_type_n minx_n miny_n maxx_n maxy_n
        example:
        0 D:\job\sandbox_fvg3\odet_cn\tools\Demo_SuperB_fv0180v3_20200525_051918_007.mf400_remap_4I_screenRGB888_0111.jpg 1664 512 7 619 176 679 273 7 712 177 771 274 7 755 207 790 263 1 786 237 801 256
        1 D:\job\sandbox_fvg3\odet_cn\tools\aa\Demo_SuperB_fv0180v3_20200525_051918_007.mf400_remap_4I_screenRGB888_0111.jpg 1664 512 7 619 176 679 273 7 712 177 771 274 7 755 207 790 263 1 786 237 801 256

        :param upLeft:
        :param lowRight:
        :param category:
        :param using_float:
        :return:
        '''
        ret = []
        #check the validation of the boxes
        #todo check to use the float or int
        if True:
            if using_float:
                minx = "{:4.2f}".format(upLeft[0])
                miny = "{:4.2f}".format(upLeft[1])
                maxx = "{:4.2f}".format(lowRight[0])
                maxy = "{:4.2f}".format(lowRight[1])
            else:
                minx = "{:d}".format(int(upLeft[0]))
                miny = "{:d}".format(int(upLeft[1]))
                maxx = "{:d}".format(int(lowRight[0]))
                maxy = "{:d}".format(int(lowRight[1]))
        else:
            return ret

        #check the validation of the category
        if category in g_lc_cate_gen3.keys():
            cate = "{:d}".format(g_lc_cate_gen3.get(category))
        else:
            print("Found the undefined category: {}".format(category))
            cate = "{:d}".format(g_lc_cate_gen3.get("other"))

        ret = [cate, minx, miny, maxx, maxy]
        return ret

    def __merge_side_box(self, upLeft, lowRight, sideUp, sideLow, view):
        '''
        at the first stage, merge the front view and side view into a large box
        :param upLeft:
        :param lowRight:
        :param sideUp:
        :param sideLow:
        :param view:
        :return: the axis of the merged box
        '''
        x = np.array([upLeft[0], lowRight[0], sideUp[0], sideLow[0]])
        y = np.array([upLeft[1], lowRight[1], sideUp[1], sideLow[1]])
        l_upLeft    = [np.min(x), np.min(y)]
        l_lowRight  = [np.max(x), np.max(y)]
        return l_upLeft, l_lowRight

    def __parse_json_files(self, file_list):
        '''
        procee the json file and get the image file path/image size/object boxes with type
        :param file_list:
        :return:
        '''
        #save image path
        images_data = []
        #save image size, width and high
        images_size = []
        #save boxes and type id in image
        boxes_data  = []

        try:
            for file in file_list:
                #process image file name and size
                image_file = self.image_path + file.split(".json")[0]+".png"
                print(image_file)
                if not os.path.exists(image_file):
                    continue

                with Image.open(image_file) as im:
                    image_size = [im.size[0], im.size[1]]
                #todo check the size of image
                if False:
                    pass
                    continue

                #process objects in the image
                boxes       = []
                boxes_draw  = []
                with open(file, encoding='utf-8') as jfile:
                    data = json.load(jfile)
                    task    = data.get("Task")
                    objects = data.get("objects")
                    if g_lc_debug:
                        print("found {:d} object".format(len(objects)))
                    for object in objects:
                        ddtypes         = object.get("ddtypes")
                        ddAttributes    = object.get("ddAttributes")
                        attribute       = dict(zip(ddtypes,ddAttributes))

                        shape           = object.get("shape")
                        category        = object.get("class")
                        upLeft          = object.get("ul")
                        lowRight        = object.get("lr")
                        sideUp          = 0.0
                        sideLow         = 0.0
                        if task == "vdet":
                            view            = attribute.get("view")
                            variation       = attribute.get("variation")
                            is_lshape       = attribute.get("lshape")
                            if is_lshape:
                                sideUp              = object.get("su")
                                sideLow             = object.get("sl")
                                upLeft, lowRight    = self.__merge_side_box(upLeft, lowRight, sideUp, sideLow, view)
                        elif task == "pdet":
                            lc              = object.get("lc")
                            directions      = attribute.get("directions")
                        else:
                            pass

                        box = self.__encode_box(upLeft, lowRight, category)
                        boxes = boxes + box
                        boxes_draw.append(box)

                if len(boxes) == 0:
                    continue

                #used for browse the label in images
                if len(images_data) % 5 == 0:
                    if len(images_data) < 200 and platform.system() == "Windows":
                        darw_image_rect(image_file, boxes_draw)
                    if len(images_data) < 20 and platform.system() == "Linux":
                        darw_image_rect(image_file, boxes_draw)

                #appending infos
                images_data.append(image_file)
                images_size.append([image_size[0],image_size[1]])
                boxes_data.append(boxes)
        except Exception as e:
            print(e)
        return images_data, images_size, boxes_data

    def __write2file(self, images_data, images_size, boxes_data, file_path="./train.txt"):
        image_count = len(images_data)
        if image_count != len(images_size) or image_count != len(boxes_data):
            return False

        try:
            fp = open(file_path, 'w+')
            for i in range(image_count):
                line_path   = "{:d} {} ".format(i, images_data[i])
                line_size   = "{:d} {:d} ".format(images_size[i][0],images_size[i][1])
                line_boxes  = " ".join(boxes_data[i])
                if g_lc_debug:
                    print(line_path)
                    print(line_size)
                    print(line_boxes)
                line = line_path + line_size + line_boxes + "\n"
                fp.write(line)
            fp.close()
        except Exception as e:
            print(e)

        return True

    def work(self):
        file_list = get_files(self.annotation_path, suffix=".json")
        print("found {:d} json files in {}".format(len(file_list), self.annotation_path))

        images_data, images_size, boxes_data = self.__parse_json_files(file_list)
        print("found {:d} json files is valid".format(len(images_data)))

        ret = self.__write2file(images_data, images_size, boxes_data, self.train_file)
        if ret:
            print("\033[1;32m Convert Successfully \033[0m")
            print("file is saved as: %s\n" % self.train_file)
        else:
            print("\033[1;31m Convert Failed \033[0m")
        return ret

if __name__ == "__main__":
    if platform.system() == "Windows":
        # converter = CLabelConverter("//abtvdfs2.de.bosch.com/ismdfs/loc/szh/DA/Driving/System_ASW/10_Training_Database/02_LABELLED_IMAGE/00_Example", train_file="D:/job/sandbox_fvg3/odet_cn/tools/val.txt")
        # converter = CLabelConverter('//bosch.com/dfsrb/DfsCN/DIV/CC/Tech/DA/00_DataExchange/zhaopeng2/image/selected/PEDESTRAIN', train_file="D:/job/sandbox_fvg3/odet_cn/tools/val_ped.txt")
        # converter = CLabelConverter('C:/Users/wng1szh/Desktop/02_Raw_Image_Label/01_Vehicle', train_file="C:/Users/wng1szh/Desktop/02_Raw_Image_Label/01_Vehicle/val_ped.txt")
        converter = CLabelConverter('/home/clever/yolov5/videoData/20200808/imageTemp/01_Vehicle', train_file="/home/clever/yolov5/videoData/20200808/imageTemp/01_Vehicle/val_veh.txt")
        converter.work()
        del converter
        converter = CLabelConverter('/home/clever/yolov5/videoData/20200808/imageTemp/02_Pedestrian', train_file="/home/clever/yolov5/videoData/20200808/imageTemp/02_Pedestrian/val_ped.txt")
        converter.work()
        del converter
    elif platform.system() == "Linux":
        converter = CLabelConverter('/home/clever/yolov5/videoData/20200808/imageTemp/02_Pedestrian', train_file="./val_ped.txt.txt", image_path="/.")
        #converter = CLabelConverter('/home/clever/yolov5/videoData/20200808/imageTemp/02_Pedestrian',train_file="./val_ped.txt", image_path="../imageTemp/01_Pedestrian")
        converter.work()
        del converter
        #converter2 = CLabelConverter()
        #converter2.work()
    else:
        pass



