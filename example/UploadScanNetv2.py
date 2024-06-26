from socket import *
import ujson
import argparse
import requests
import cv2
import numpy as np
from multiprocessing import Pool

parser = argparse.ArgumentParser(
        description='WISE UI Web Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--RKeywords', type=str,
    help='Received keyword lists')
parser.add_argument(
    '--SKeywords', type=str,
    help='Sendeded keyword lists')
parser.add_argument(
    '--DataType', type=str,
    help='Data type')
parser.add_argument(
    '--use_gpu', type=str, default='0',
    help='port number')
parser.add_argument(
    '--FACADE_SERVER_ADDR', type=str,
    help='facade server address')
parser.add_argument(
    '--ECHO_SERVER_IP', type=str, default='0.0.0.0',
    help='ip address')
parser.add_argument(
    '--ECHO_SERVER_PORT', type=int, default=35001,
    help='port number')
opt = parser.parse_args()

FACADE_SERVER_ADDR = opt.FACADE_SERVER_ADDR
keyword = "ScanNetv2"
scene_id = "0720"
path = "D:\\UVR\\simplerecon-main\\data_scripts\\ScanNetv2\\scans_test\\scene"+(scene_id)+"_00"
src1 = "scene"+(scene_id)+".color"
src2 = "scene"+(scene_id)+".depth"
sess = requests.Session()

def upload(i):
    cfile = path + "/sensor_data/frame-" + "{:06d}".format(i) + ".color.jpg"
    cimg = cv2.imread(cfile, cv2.IMREAD_UNCHANGED)
    if cimg is None:
        return i
    dfile = path + "/sensor_data/frame-" + "{:06d}".format(i) + ".depth.png"
    dimg = cv2.imread(dfile, cv2.IMREAD_UNCHANGED)
    if dimg is None:
        return i
    retval1, buf1 = cv2.imencode('.jpg', cimg, [cv2.IMWRITE_JPEG_QUALITY, 70])
    retval2, buf2 = cv2.imencode('.png', dimg, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    sess.post(FACADE_SERVER_ADDR + "/Upload?keyword=" + keyword + "&id=" + str(i) + "&src=" + src1, buf1.tobytes())
    sess.post(FACADE_SERVER_ADDR + "/Upload?keyword=" + keyword + "&id=" + str(i) + "&src=" + src2, buf2.tobytes())
    # cv2.imshow("color", cimg)
    # cv2.imshow("depth", dimg)
    # cv2.waitKey(1)
    return i

if __name__ == '__main__':

    Data = {}
    capacity = 0
    ReceivedKeywords = opt.RKeywords.split(',')
    SendKeywords = opt.SKeywords
    datatype = opt.DataType

    sess.post(FACADE_SERVER_ADDR + "/Connect", ujson.dumps({
        # 'port':opt.port,'key': keyword, 'prior':opt.prior, 'ratio':opt.ratio
        'src': 'TestServer', 'type1': 'server', 'type2': 'test', 'keyword': SendKeywords,
        'capacity': capacity, 'Additional': None
    }))
    ECHO_SERVER_ADDR = (opt.ECHO_SERVER_IP, opt.ECHO_SERVER_PORT)
    ECHO_SOCKET = socket(AF_INET, SOCK_DGRAM)
    for rkeyword in ReceivedKeywords:
        temp = ujson.dumps({'type1': 'connect', 'keyword': rkeyword, 'src': 'TestServer', 'type2': 'all'})
        ECHO_SOCKET.sendto(temp.encode(), ECHO_SERVER_ADDR)
        Data[rkeyword] = {}

    #path = "E:\\SLAM_DATASET\\TUM\\rgbd_dataset_freiburg2_desk_with_person"



    file = "scene"+(scene_id)+"_00.txt"
    f = open(path+"/"+file, mode='r')
    lines = f.readlines()
    num = int(lines[-3].split('=')[1])

    with Pool(10) as p:
        p.map(upload, list(range(num)))

    """
    for i in range(num):
        cfile = path+"/sensor_data/frame-"+"{:06d}".format(i)+".color.jpg"
        cimg = cv2.imread(cfile, cv2.IMREAD_UNCHANGED)
        dfile = path + "/sensor_data/frame-" + "{:06d}".format(i) + ".depth.png"
        dimg = cv2.imread(dfile, cv2.IMREAD_UNCHANGED)

        retval1, buf1 = cv2.imencode('.jpg', cimg, [cv2.IMWRITE_JPEG_QUALITY, 70])
        retval2, buf2 = cv2.imencode('.png', dimg, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        sess.post(FACADE_SERVER_ADDR + "/Store?keyword=" + keyword + "&id=" + str(i) + "&src=" + src1, buf1.tobytes())
        sess.post(FACADE_SERVER_ADDR + "/Store?keyword=" + keyword + "&id=" + str(i) + "&src=" + src2, buf2.tobytes())

        cv2.imshow("color", cimg)
        cv2.imshow("depth", dimg)
        cv2.waitKey(1)
    """
    sess.post(FACADE_SERVER_ADDR + "/Save?keyword=" + keyword + "&src=" + src1)
    sess.post(FACADE_SERVER_ADDR + "/Save?keyword=" + keyword + "&src=" + src2)
