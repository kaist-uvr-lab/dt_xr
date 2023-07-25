import threading
import ujson
import time
import numpy as np
import requests
import cv2
from socket import *
import argparse
import os

#Thread Pool
from concurrent.futures import ThreadPoolExecutor
import queue

from module.ProcessingTime import ProcessingTime
import pickle
import torch

import signal
import sys
def saveProcessingTime():
    try:
        Data["process"].update()
        Data["upload"].update()
        Data["download"].update()
        print("Process = ",Data["process"].print())
        print("upload = ", Data["upload"].print())
        print("download = ", Data["download"].print())
        pickle.dump(Data["process"], open('./evaluation/processing_time.bin', "wb"))
        pickle.dump(Data["upload"], open('./evaluation/upload_time.bin', "wb"))
        pickle.dump(Data["download"], open('./evaluation/download_time.bin', "wb"))
    except Exception as e:
        print(e)
def signal_handler(signal, frame):
    try:
        saveProcessingTime()
        sys.exit(0)
    except:
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

#keyboard.add_hotkey("ctrl+p",lambda: printProcessingTime())
#keyboard.add_hotkey("ctrl+s",lambda: saveProcessingTime())

def Display(res):
    for i, (im, pred) in enumerate(zip(res.imgs, res.pred)):
        n = len(pred)
        fdata = np.zeros(n * 6, dtype=np.float32)  # n*6
        idx = 0
        for *box, conf, cls in reversed(pred):
            # label = res.names[int(cls)]
            # print("%s %f"%(label, conf))
            #print("%f %f %f %f"%(box[0], box[1], box[2], box[3]))
            fdata[idx] = float(cls)
            fdata[idx + 1] = conf
            fdata[idx + 2] = box[0]
            fdata[idx + 3] = box[1]
            fdata[idx + 4] = box[2]
            fdata[idx + 5] = box[3]
            idx = idx + 6

        return fdata

def predict(message):

    data = ujson.loads(message.decode())
    id = data['id']
    src = data['src']

    tds = time.time()
    res = sess.post(FACADE_SERVER_ADDR + "/Load?keyword=" + datatype + "&id=" + str(id) + "&src=" + src, "")
    tde = time.time()

    img_array = np.frombuffer(res.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    tps = time.time()
    with torch.no_grad():
        results = model(img)
    fdata = Display(results)
    tpe = time.time()

    if fdata.size > 0 :
        tus = time.time()
        sess.post(FACADE_SERVER_ADDR + "/Store?keyword=ObjectDetection&id=" + str(id) + "&src=" + src, fdata.tobytes())
        tue = time.time()

        Data["process"].add(tpe - tps, len(res.content))
        Data["upload"].add(tue - tus, len(fdata))
        Data["download"].add(tde - tds, len(fdata))

        try:
            print('Detection = ', src, id, tpe-tps, tue-tus, tde-tds,
                  len(fdata.tobytes()))
        except Exception as e:
            print(e)

bufferSize = 1024

def udpthread():

    #executor = ThreadPoolExecutor(NUM_MODEL)

    while True:
        bytesAddressPair = ECHO_SOCKET.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        predict(message)
        #msgpipe.put(message)
        #pipe.put(message)
        #with ThreadPoolExecutor() as executor:
        #    executor.submit(predict, pipe)

if __name__ == '__main__':
    ##################################################
    ##arguments parsing
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
    ##arguments parsing

    ###CONNECT
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
            avg_time = 0
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
    capacity = 0
    ##Echo server

    FACADE_SERVER_ADDR = opt.FACADE_SERVER_ADDR
    ReceivedKeywords = opt.RKeywords.split(',')
    SendKeywords = opt.SKeywords
    datatype = opt.DataType

    sess = requests.Session()
    sess.post(FACADE_SERVER_ADDR + "/Connect", ujson.dumps({
        # 'port':opt.port,'key': keyword, 'prior':opt.prior, 'ratio':opt.ratio
        'src': 'ObjectDetectionServer', 'type1': 'server', 'type2': 'test', 'keyword': SendKeywords, 'capacity':capacity, 'Additional': None
    }))
    ECHO_SERVER_ADDR = (opt.ECHO_SERVER_IP, opt.ECHO_SERVER_PORT)
    ECHO_SOCKET = socket(AF_INET, SOCK_DGRAM)
    for keyword in ReceivedKeywords:
        temp = ujson.dumps({'type1': 'connect', 'keyword': keyword, 'src': 'ObjectDetectionServer', 'type2': 'all'})
        ECHO_SOCKET.sendto(temp.encode(), ECHO_SERVER_ADDR)
        Data[keyword] = {}
    # Echo server connect
    ######################

    ####LOAD MODEL
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #pipe = queue.Queue(100)
    with torch.no_grad():
        rgb = np.random.randint(255, size=(640, 480, 3), dtype=np.uint8)
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval().to(device)
        model(rgb)
    print("Load model")
    th1 = threading.Thread(target=udpthread)
    th1.start()
    print("thread start")