import threading
import ujson
import time
import numpy as np
import requests
import cv2
from socket import *

import argparse
import torch, torchvision

from module.ProcessingTime import ProcessingTime
import pickle

#Thread Pool
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import queue

##CSAILVision
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from mit_semseg.config import cfg

import os

def saveProcessingTime():
    Data["process"].update()
    Data["upload"].update()
    Data["download"].update()
    print("Process = ", Data["process"].print())
    print("upload = ", Data["upload"].print())
    print("download = ", Data["download"].print())
    pickle.dump(Data["process"], open('./evaluation/processing_time.bin', "wb"))
    pickle.dump(Data["upload"], open('./evaluation/upload_time.bin', "wb"))
    pickle.dump(Data["download"], open('./evaluation/download_time.bin', "wb"))

import signal
import sys
def signal_handler(signal, frame):
    try:
        saveProcessingTime()
        sys.exit(0)
    except:
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])

bufferSize = 1024
def udpthread():
    #executor = ThreadPoolExecutor(NUM_MODEL)
    while True:
        bytesAddressPair = ECHO_SOCKET.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        predict(message)

def predict(message):

    data = ujson.loads(message.decode())
    id = data['id']
    src = data['src']
    ss = src.split('.')
    if len(ss) > 1:
        src = ss[0]
        datatype = ss[1]
    tds = time.time()
    res = sess.post(FACADE_SERVER_ADDR + "/Download?keyword="+datatype+"&id=" + str(id)+"&src="+src, "")
    tde = time.time()

    img_array = np.frombuffer(res.content, dtype=np.uint8)
    img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    #resized = cv2.resize(img_cv, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    segSize = (img_cv.shape[0], img_cv.shape[1])
    img_data = pil_to_tensor(img_cv)

    tps = time.time()
    singleton_batch = {'img_data': img_data[None].cuda(), 'img_data': img_data[None].cuda()}
    with torch.no_grad():
        scores = model(singleton_batch, segSize=segSize)
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy().astype('int8')
    _, png = cv2.imencode('.png', pred)
    tpe = time.time()

    tus = time.time()
    sess.post(FACADE_SERVER_ADDR + "/Upload?keyword=Segmentation&id=" + str(id) + "&src=" + src, png.tobytes())
    tue = time.time()

    Data["process"].add(tpe - tps, len(png))
    Data["upload"].add(tue - tus, len(png))
    Data["download"].add(tde - tds, len(res.content))

    print('Segmentation = ', src, id, tpe - tps, tue - tus, tde - tds)


if __name__ == "__main__":

    ##################################################
    ##basic arguments
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
        '--ip', type=str,default='0.0.0.0',
        help='ip address')
    parser.add_argument(
        '--port', type=int, default=35006,
        help='port number')
    parser.add_argument(
        '--use_gpu', type=str, default='0',
        help='port number')
    parser.add_argument(
        '--prior', type=str, default='0',
        help='port number')
    parser.add_argument(
        '--ratio', type=str, default='1',
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

    ##segmentation arguments
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    opt = parser.parse_args()
    ##segmentation arguments

    ####segmentation module configuration
    Data = {}
    try:
        path = os.path.dirname(os.path.realpath(__file__))
        f = open('./evaluation/processing_time.bin', 'rb')
        Data["process"] = pickle.load(f)
        f.close()
        f = open('./evaluation/upload_time.bin', 'rb')
        Data["upload"] = pickle.load(f)
        f.close()
        f = open('./evaluation/download_time.bin', 'rb')
        Data["download"] = pickle.load(f)
        f.close()

        try:
            avg_time = Data["process"].avg + Data["upload"].avg + Data["download"].avg
            capacity = int(1 / avg_time) - 1
        except ZeroDivisionError:
            capacity = 0
            avg_time=0
        finally:
            print('capacity', capacity, avg_time)

    except FileNotFoundError:
        Data["process"] = ProcessingTime()
        Data["upload"] = ProcessingTime()
        Data["download"] = ProcessingTime()
    finally:
        print("Process = ", Data["process"].print())
        print("upload = ", Data["upload"].print())
        print("download = ", Data["download"].print())

    ##Echo server
    FACADE_SERVER_ADDR = opt.FACADE_SERVER_ADDR
    ReceivedKeywords = opt.RKeywords.split(',')
    SendKeywords = opt.SKeywords
    datatype = opt.DataType

    sess = requests.Session()
    sess.post(FACADE_SERVER_ADDR + "/Connect", ujson.dumps({
        # 'port':opt.port,'key': keyword, 'prior':opt.prior, 'ratio':opt.ratio
        'src': 'SegmentationServer', 'type1': 'server', 'type2': 'NONE', 'keyword': SendKeywords, 'capacity': capacity, 'Additional': None
    }))
    ECHO_SERVER_ADDR = (opt.ECHO_SERVER_IP, opt.ECHO_SERVER_PORT)
    ECHO_SOCKET = socket(AF_INET, SOCK_DGRAM)
    for keyword in ReceivedKeywords:
        temp = ujson.dumps({'type1': 'connect', 'keyword': keyword, 'src': 'SegmentationServer', 'type2': 'all'})
        ECHO_SOCKET.sendto(temp.encode(), ECHO_SERVER_ADDR)
        Data[keyword] = {}
    # Echo server connect

    ######GPU Initialization
    device = torch.device("cuda:" + opt.use_gpu) if torch.cuda.is_available() else torch.device("cpu")
    ####segmentation module configuration
    cfg.merge_from_file(opt.cfg)
    cfg.merge_from_list(opt.opts)
    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.dirname(os.path.realpath(__file__)) + '/model/encoder_' + cfg.TEST.checkpoint
    cfg.MODEL.weights_decoder = os.path.dirname(os.path.realpath(__file__)) + '/model/decoder_' + cfg.TEST.checkpoint

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
           os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)
    crit = torch.nn.NLLLoss(ignore_index=-1)
    model = SegmentationModule(net_encoder, net_decoder, crit).to(device)
    rgb = np.random.randint(255, size=(640, 480, 3), dtype=np.uint8)
    tempsize = (rgb.shape[0], rgb.shape[1])
    init_data = pil_to_tensor(rgb)
    init_singleton_batch = {'img_data': init_data[None].cuda()}
    with torch.no_grad():
        model(init_singleton_batch, segSize=tempsize)

    print("initialization!!")
    ##init

    msgpipe = queue.Queue(100)
    th1 = threading.Thread(target=udpthread)
    th1.start()



