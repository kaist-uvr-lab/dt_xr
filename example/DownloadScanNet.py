from socket import *
import ujson
import argparse
import requests
import cv2
import numpy as np

if __name__ == '__main__':
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

    Data = {}
    capacity = 0
    FACADE_SERVER_ADDR = opt.FACADE_SERVER_ADDR
    ReceivedKeywords = opt.RKeywords.split(',')
    SendKeywords = opt.SKeywords
    datatype = opt.DataType

    #DATA COMMUNICATION SEVER 연결
    sess = requests.Session()
    #전송할 키워드 등록
    sess.post(FACADE_SERVER_ADDR + "/Connect", ujson.dumps({
        # 'port':opt.port,'key': keyword, 'prior':opt.prior, 'ratio':opt.ratio
        'src': 'TestServer', 'type1': 'server', 'type2': 'test', 'keyword': SendKeywords,
        'capacity': capacity, 'Additional': None
    }))
    #NOTIFICATION UDP SERVER 연결
    ECHO_SERVER_ADDR = (opt.ECHO_SERVER_IP, opt.ECHO_SERVER_PORT)
    ECHO_SOCKET = socket(AF_INET, SOCK_DGRAM)
    #수신할 키워드 등록
    for keyword in ReceivedKeywords:
        temp = ujson.dumps({'type1': 'connect', 'keyword': keyword, 'src': 'TestServer', 'type2': 'all'})
        ECHO_SOCKET.sendto(temp.encode(), ECHO_SERVER_ADDR)
        Data[keyword] = {}

    #서버에 미리 저장 된 데이터셋 준비 요청
    #keyword = "TUM"
    #src1 = "str_tex_far.depth"
    #src2 = "str_tex_far.color"
    keyword = "ScanNetv2"
    scene_id = "0720"
    src1 = "scene" + (scene_id) + ".color"
    src2 = "scene" + (scene_id) + ".depth"
    sess.post(FACADE_SERVER_ADDR + "/Load?keyword=" + keyword + "&src=" + src1)
    sess.post(FACADE_SERVER_ADDR + "/Load?keyword=" + keyword + "&src=" + src2)

    # 데이터셋의 id 리스트 획득
    res = sess.post(FACADE_SERVER_ADDR + "/Get?keyword=" + keyword + "&src=" + src1)
    ids = np.frombuffer(res.content, dtype=np.uint32)

    for id in ids:
        #데이터 다운로드
        res = sess.post(FACADE_SERVER_ADDR + "/Download?keyword=" + keyword + "&id=" + str(id) + "&src=" + src1, "")
        data_array = np.frombuffer(res.content, dtype=np.uint8)
        color = cv2.imdecode(data_array, cv2.IMREAD_UNCHANGED)
        res = sess.post(FACADE_SERVER_ADDR + "/Download?keyword=" + keyword + "&id=" + str(id) + "&src=" + src2, "")
        data_array = np.frombuffer(res.content, dtype=np.uint8)
        depth = cv2.imdecode(data_array, cv2.IMREAD_UNCHANGED)

        cv2.imshow("color", color)
        cv2.imshow("depth", depth)
        cv2.waitKey(10)
        #데이터 처리

        #데이터 업로드
        #res = sess.post(FACADE_SERVER_ADDR + "/Store?keyword=" + keyword + "&id=" + str(id) + "&src=" + src, "")s