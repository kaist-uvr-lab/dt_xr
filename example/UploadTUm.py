from socket import *
import ujson
import argparse
import requests
import cv2
import numpy as np
from multiprocessing import Pool
import evaluate
import associate

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
keyword = "TUM"
cam_id =1
cam_id = str(cam_id)
scene_id = cam_id+"_xyz"
path = "E:\\SLAM_DATASET\\TUM\\rgbd_dataset_freiburg"+scene_id
#keyword = "BONN"
#scene_id = "crowd3"
#path = "E:/SLAM_DATASET/bonn/rgbd_bonn_"+scene_id
src1 = keyword+(scene_id)+".color"
src2 = keyword+(scene_id)+".depth"
src3 = keyword+(scene_id)+".ts"
sess = requests.Session()

rgb_file = path+"/rgb.txt"
depth_file = path+"/depth.txt"

rgb_list = associate.read_file_list(rgb_file)
depth_list = associate.read_file_list(depth_file)
matches = evaluate.match(rgb_list, depth_list)
asso = []
tid = 0
for a, b in matches:
    tid = tid+1
    tmp = ("%f %s %f %s %d" % (a, " ".join(rgb_list[a]), b, " ".join(depth_list[b]), tid))
    asso.append(tmp)

def upload(ss):
    tmp = ss.split(' ')
    ts = tmp[0]
    id = tmp[4]
    cfile = path+'/'+tmp[1]
    cimg = cv2.imread(cfile, cv2.IMREAD_UNCHANGED)
    if cimg is None:
        return ss
    dfile = path+'/'+tmp[3]
    dimg = cv2.imread(dfile, cv2.IMREAD_UNCHANGED)
    if dimg is None:
        return ss

    retval1, buf1 = cv2.imencode('.jpg', cimg, [cv2.IMWRITE_JPEG_QUALITY, 70])
    retval2, buf2 = cv2.imencode('.png', dimg, [cv2.IMWRITE_PNG_COMPRESSION, 1])

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

    #save intrinsic
    if True :
        import yaml
        pfile = "E:\\SLAM_DATASET\\TUM\\"+keyword+cam_id+".yaml"
        with open(pfile, 'r') as stream:
            yfile = yaml.full_load(stream)
        src4 = keyword+cam_id + ".intrinsic"
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
        if 'ThDepth' in yfile.keys():
            thDepth = yfile['ThDepth']
        fps = yfile['Camera.fps']

        intrinsic = [w, h, fx, fy, cx, cy, d0, d1, d2, d3, d4, fps, bf, thDepth]
        camdata = np.array(intrinsic, dtype=np.float32)
        print(camdata)
        print(w,h,bf, thDepth,d4)
        sess.post(FACADE_SERVER_ADDR + "/Upload?keyword=" + keyword + "&id=" + "0" + "&src=" + src4, camdata.tobytes())
        sess.post(FACADE_SERVER_ADDR + "/Save?keyword=" + keyword + "&src=" + src4)

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

