from socket import *
import ujson
import argparse
import requests
import cv2
import numpy as np
from multiprocessing import Pool
import evaluate
import associate

import yaml

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
keyword = "KITTI"
scene_id = "07"
pid = 7

kw_intrinsic = keyword+str(pid)

path = "E:\\SLAM_DATASET\\KITTI\\sequences\\"
pfile = path+kw_intrinsic+".yaml"
path = path + scene_id
# keyword = "BONN"
# scene_id = "crowd3"
# path = "E:/SLAM_DATASET/bonn/rgbd_bonn_"+scene_id

src1 = keyword + (scene_id) + ".left"
src2 = keyword + (scene_id) + ".right"
src3 = keyword + (scene_id) + ".ts"
sess = requests.Session()

ts_file = path + "/times.txt"
lpath = path+"/image_2"
rpath = path+"/image_3"

file = open(ts_file)
lines = file.readlines()

asso = []
tid = 0
for line in lines:
    line = line.strip()
    ts = float(line)
    tmp = ("%f %s %d"%(ts, str(tid).zfill(6)+".png",tid))
    tid = tid + 1
    asso.append(tmp)

def upload(ss):
    tmp = ss.split(' ')
    ts = tmp[0]
    id = tmp[2]
    lfile = lpath + '/' + tmp[1]
    limg = cv2.imread(lfile, cv2.IMREAD_UNCHANGED)
    if limg is None:
        return ss
    rfile = rpath + '/' + tmp[1]
    rimg = cv2.imread(rfile, cv2.IMREAD_UNCHANGED)
    if rimg is None:
        return ss

    retval1, buf1 = cv2.imencode('.jpg', rimg, [cv2.IMWRITE_JPEG_QUALITY, 70])
    retval2, buf2 = cv2.imencode('.jpg', limg, [cv2.IMWRITE_JPEG_QUALITY, 70])

    sess.post(FACADE_SERVER_ADDR + "/Upload?keyword=" + keyword + "&id=" + id + "&src=" + src1, buf1.tobytes())
    sess.post(FACADE_SERVER_ADDR + "/Upload?keyword=" + keyword + "&id=" + id + "&src=" + src2, buf2.tobytes())
    sess.post(FACADE_SERVER_ADDR + "/Upload?keyword=" + keyword + "&id=" + id + "&src=" + src3, ts)

    return ss


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

    # save intrinsic
    src4 = kw_intrinsic
    with open(pfile, 'r') as stream:
        yfile = yaml.full_load(stream)
        print(src4, yfile)
        w = yfile['Image.width']
        h = yfile['Image.height']
        fx = yfile['Camera.fx']
        fy = yfile['Camera.fy']
        cx = yfile['Camera.cx']
        cy = yfile['Camera.cy']
        d0 = yfile['Camera.k1']
        d1 = yfile['Camera.k2']
        d2 = yfile['Camera.p1']
        d3 = yfile['Camera.p2']
        d4 = 0.0
        if 'Camera.k3' in yfile.keys():
            d4 = yfile['Camera.k3']
        bf = 0.0
        if 'Camera.bf' in yfile.keys():
            bf = yfile['Camera.bf']
        thDepth = 0.0
        if 'Camera.ThDepth' in yfile.keys():
            thDepth = yfile['ThDepth']
        fps = yfile['Camera.fps']

        intrinsic = [w, h, fx, fy, cx, cy, d0, d1, d2, d3, d4, fps, bf, thDepth]
        camdata = np.array(intrinsic, dtype=np.float32)
        print(camdata)
        sess.post(FACADE_SERVER_ADDR + "/Upload?keyword=" + keyword + "&id=" + "0" + "&src=" + src4, camdata.tobytes())
        sess.post(FACADE_SERVER_ADDR + "/Save?keyword=" + keyword + "&src=" + src4)
        print(w,h)
    """
    src4 = keyword + ".intrinsic"
    w = 1241
    h = 376
    fx = 718.856
    fy = 718.856
    cx = 607.1928
    cy = 185.2157
    #k1 k2 p1 p2 k3
    d0 = 0.0
    d1 = 0.0
    d2 = 0.0
    d3 = 0.0
    d4 = 0.0
    fps = 10.0
    bf = 386.1448
    thDepth = 35
    intrinsic = [w, h, fx, fy, cx, cy, d0,d1,d2,d3,d4, fps, bf, thDepth]
    camdata = np.array(intrinsic, dtype=np.float32)
    print(camdata)
    sess.post(FACADE_SERVER_ADDR + "/Upload?keyword=" + keyword + "&id=" + "0" + "&src=" + src4, camdata.tobytes())
    sess.post(FACADE_SERVER_ADDR + "/Save?keyword=" + keyword + "&src=" + src4)
    """
    # save intrinsic

    with Pool(10) as p:
        p.map(upload, asso)

    sess.post(FACADE_SERVER_ADDR + "/Save?keyword=" + keyword + "&src=" + src1)
    sess.post(FACADE_SERVER_ADDR + "/Save?keyword=" + keyword + "&src=" + src2)
    sess.post(FACADE_SERVER_ADDR + "/Save?keyword=" + keyword + "&src=" + src3)
    """
    file = "scene"+(scene_id)+"_00.txt"
    f = open(path+"/"+file, mode='r')
    lines = f.readlines()
    num = int(lines[-3].split('=')[1])


    sess.post(FACADE_SERVER_ADDR + "/Save?keyword=" + keyword + "&src=" + src1)
    sess.post(FACADE_SERVER_ADDR + "/Save?keyword=" + keyword + "&src=" + src2)
    """

