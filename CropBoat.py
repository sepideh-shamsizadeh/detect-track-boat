from os import path
import glob

import cv2
from xml.dom import minidom
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


num = 0
for fl in glob.glob('data/boat_1/*.png'):
    img = cv2.imread(fl)
    mta = minidom.parse(path.join('data/annotations/', path.splitext(path.basename(fl))[0]+'.xml'))

    for box in mta.getElementsByTagName('bndbox'):
        num += 1
        xmin = int(box.getElementsByTagName('xmin')[0].childNodes[0].data)
        xmax = int(box.getElementsByTagName('xmax')[0].childNodes[0].data)
        ymin = int(box.getElementsByTagName('ymin')[0].childNodes[0].data)
        ymax = int(box.getElementsByTagName('ymax')[0].childNodes[0].data)
        print(fl)
        print(num)
        boat = img[ymin:ymax, xmin:xmax]
        cv2.imshow('boat', boat)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite('data/train/boat/{}.png'.format(num), boat)













