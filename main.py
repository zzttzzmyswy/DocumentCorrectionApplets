# --coding: utf-8 --
import base64
import json
import logging
import math
import os
import shutil
from glob import glob
from pathlib import Path

import imutils
import numpy as np
import requests
from cv2 import cv2

# 获取access_token
# client_id 为官网获取的AK， client_secret 为官网获取的SK
appid = "25263167"
client_id = "8nu95YZxBaR7detNcNiB2xkz"
client_secret = "7fmy2jVyo2Ksatgfg7oYKZSLHhkyc7Fx"
token_url = "https://aip.baidubce.com/oauth/2.0/token"
host = f"{token_url}?grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}"
response = requests.get(host)
access_token = response.json().get("access_token")

# 配置程序运行报告参数
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # 日志等级总开关
# 设置日志文件相关
fileLog = logging.FileHandler('runlog.log', mode='a+', encoding="utf-8")
fileLog.setLevel(logging.DEBUG)  # 输出到file的日志等级
fileLog.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    ))
# 设置输出到控制台的日志参数
consoleLog = logging.StreamHandler()
consoleLog.setLevel(logging.INFO)  # 输出到console的日志等级
consoleLog.setFormatter(
    logging.Formatter(
        "%(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"))
# 添加日志控制块
logger.addHandler(fileLog)
logger.addHandler(consoleLog)


def baiduOCR(filPath, name):
    # 调用通用文字识别接口
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
    # 以二进制方式打开图文件
    # 参数image：图像base64编码
    # 下面图片路径请自行切换为自己环境的绝对路径
    with open(filPath, "rb") as f:
        image = base64.b64encode(f.read())
    body = {
        "image": image,  # 图像信息
        "language_type": "CHN_ENG",  # 检测方式为中英混合
        "detect_direction": "true",  # 自动检测90、180、270度的旋转
        "paragraph": "false",  # 检测不分段落
        "probability": "true",  # 回报检测预估准确率
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    request_url = f"{request_url}?access_token={access_token}"
    response = requests.post(request_url, headers=headers, data=body)
    content = response.content.decode("UTF-8")
    content = json.loads(content)
    ret = ''
    retProbabilityList = []
    if content['words_result_num'] > 0:
        for inf in content['words_result']:
            ret += (inf['words'] + '\r\n')
            retProbabilityList.append(inf['probability']['average'])
    else:
        ret += "没有识别结果\r\n"
        retProbabilityList.append(0)
    retProbability = np.mean(retProbabilityList)
    file_handle = open(os.path.dirname(filPath) + '/' +
                       os.path.basename(filPath).split('.')[0] + '.txt',
                       mode='w',
                       encoding="utf-8")
    file_handle.write(filPath + ' 文字识别内容:\r\n')
    file_handle.write("识别准确率（仅供参考）:" + str(round(retProbability * 100, 2)) +
                      '%\r\n')
    logging.info(name + "识别准确率（仅供参考）:" + str(round(retProbability * 100, 2)) +
                 '%')
    file_handle.write(ret)
    file_handle.close()


def findPoint(pts):  # 由轮廓检测结果获得轮廓的四点用于透视变换
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")
    # 按顺序找到对应坐标，左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def findPointTransform(image, pts):  # 透视变换
    # 获取输入坐标点
    if pts is None:
        return None
    rect = findPoint(pts)
    (tl, tr, br, bl) = rect
    # 计算输入的w和h值
    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    # 取水平最大值
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    # 取竖直最大值
    maxHeight = max(int(heightA), int(heightB))
    # 变换后对应坐标位置（-1只是为了防止有误差出现，不-1也可以。）
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]],
                   dtype="float32")
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warPed = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # 返回变换后结果
    return warPed


def fContours(img, edged):
    # 轮廓检测
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None
    # 遍历轮廓
    for c in cnts:
        # 计算轮廓近似
        peri = cv2.arcLength(c, True)
        # 多边拟合函数 c表示输入的点集，epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # img = cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
        # 当拟合函数拟合了一个矩形时结束遍历
        if len(approx) == 4:
            screenCnt = approx
            cont = cv2.contourArea(c)
            logging.debug("边缘检测预估文档面积：" + str(cont))
            if cont < 200000:
                logging.warning("边缘检测预估文档面积过小，轮廓识别梯形矫正结果可能不正确")
                # 当预估面积过小时抛弃此次运算
                # screenCnt = None
            # 之前使用sorted对轮廓面积进行了排序，所以screenCnt会自动取轮廓面积的最大值
            break
    if screenCnt is None:
        return img
    res = cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
    return res, screenCnt


def adaptreThre(img, size):
    # 自适应区域阈值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,
                                int((140 * size) / 2) * 2 + 1, 15)
    return ret

def equalHist(img):
    # 图像灰度化并均衡化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret = cv2.equalizeHist(gray)
    return ret

def getMad(s):  # 利用绝对中位差剔除异常值
    median = np.median(s)
    constant = 1.4826
    mad = constant * np.median(np.abs(s - median))
    lowerLimit = median - (3 * mad)
    upperLimit = median + (3 * mad)
    return lowerLimit, upperLimit


def rotate(image, angle):  # 返回旋转结果|输入图像，旋转角度
    # 获得旋转因子
    # rotationFactor = cv2.getRotationMatrix2D(center, angle, 1)
    # 进行仿射变换
    # return cv2.warpAffine(image, rotationFactor, (h, w), None, cv2.INTER_LANCZOS4, borderValue=(255, 255, 255))

    # 使用这种方式保证旋转后图像不会丢失信息
    return imutils.rotate_bound(image, angle)


def documentCorrection(file_path):  # 文档图像处理主函数
    # 新建存放结果需要的文件夹
    fileName = os.path.basename(file_path).split('.')[0]
    os.system("mkdir " + str(Path("output/" + fileName)))
    os.system("mkdir " + str(Path("temp/" + fileName)))
    # 读取图片，灰度化
    src = cv2.imread(file_path)
    saveIntermediateImage("1.原图像", src, fileName)
    heigh = src.shape[0] * float(1200) / src.shape[1]
    srcy = cv2.resize(src, (1200, int(heigh)),
                      fx=None,
                      fy=None,
                      interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(srcy, cv2.COLOR_BGR2GRAY)
    saveIntermediateImage("2.缩放后的原图像", src, fileName)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    saveIntermediateImage("3.灰度图像", gray, fileName)
    # 生成腐蚀和膨胀所需的核
    kernel = np.ones((5, 5), np.uint8)
    # 腐蚀：取结构元所指定的领域内值的最小值作为该位置的输出灰度值
    etchImage = cv2.erode(gray, kernel)
    saveIntermediateImage("4.腐蚀后图像", etchImage, fileName)
    # 膨胀：取结构元所指定的领域内值的最大值作为该位置的输出灰度值
    expandImage = cv2.dilate(etchImage, kernel)
    saveIntermediateImage("5.腐蚀并膨胀后图像", expandImage, fileName)
    # 边缘检测
    cannyImage = cv2.Canny(expandImage, 50, 150)
    saveIntermediateImage("6.Canny算子边缘检测下载", cannyImage, fileName)
    # 轮廓检测
    contours, contoursp = fContours(srcy, cannyImage)
    if contoursp is not None:
        saveIntermediateImage("7.轮廓检测图像", contours, fileName)
        findPointImg = findPointTransform(
            src,
            contoursp.reshape(4, 2) * (src.shape[1] / 1200.0))
        if findPointImg is not None:
            baiduOCRFile = saveFinalImage("轮廓识别梯形矫正处理结果", findPointImg, fileName)
            baiduOCR(baiduOCRFile, "轮廓识别梯形矫正处理结果")
            thFile = adaptreThre(findPointImg,
                                 (findPointImg.shape[1] / 1200.0) * 0.5 + 0.5)
            baiduOCRFile = saveFinalImage("梯形矫正自适应区域二值化处理结果", thFile, fileName)
            baiduOCR(baiduOCRFile, "梯形矫正自适应区域二值化处理结果")
            # 直方图均衡化对白底文档效果不佳，不使用
            # elFile = equalHist(findPointImg)
            # baiduOCRFile = saveFinalImage("梯形矫正直方图均衡化处理结果", elFile, fileName)
            # baiduOCR(baiduOCRFile, "梯形矫正直方图均衡化处理结果")
        else:
            logging.warning(file_path + "文件没有正确地轮廓识别")
    else:
        logging.warning(file_path + "文件没有正确地轮廓识别")
    # 霍夫变换得到线条
    lines = cv2.HoughLinesP(cannyImage,
                            0.9,
                            np.pi / 180,
                            90,
                            minLineLength=100,
                            maxLineGap=10)
    # 两组霍夫线结果，一组不进行处理，一组过滤调对偏斜角度计算产生不利影响的线
    HoughTransformImage1 = np.zeros(srcy.shape[:], dtype=np.uint8)
    HoughTransformImage2 = np.zeros(srcy.shape[:], dtype=np.uint8)
    # 定义角度数组
    theras = []
    # 画出线条
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 计算角度,x轴向右，y轴向下
        # 防止出现角度过大的线导致计算溢出
        k = float(y1 - y2) / (1 if abs(x1 - x2) < 1 else (x1 - x2))
        thera = np.degrees(math.atan(k))
        theras.append(thera)
        cv2.line(HoughTransformImage1, (x1, y1), (x2, y2), (255, 255, 255),
                 1,
                 lineType=cv2.LINE_AA)
    # 排除异常值
    lowerLimit, upperLimit = getMad(theras)
    # 当文字行数不足时，绝对中位差可能会出现错误，使用60度的阈值对其进行约束
    if lowerLimit < -30:
        lowerLimit = -30
        logging.warning("偏移角度分析低阈值超出预定范围-30度，旋转矫正结果可能不正确")
    if upperLimit > 30:
        upperLimit = 30
        logging.warning("偏移角度分析高阈值超出预定范围30度，旋转矫正结果可能不正确")
    theras.clear()
    logging.debug("自动确认正常角度范围: " + str(round(lowerLimit, 2)) + "度" + " - " +
                  str(round(upperLimit, 2)) + "度")

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 计算角度,x轴向右，y轴向下
        # 防止出现角度过大的线导致计算溢出
        k = float(y1 - y2) / (1 if abs(x1 - x2) < 1 else (x1 - x2))
        thera = np.degrees(math.atan(k))
        # 处理使用阈值过滤后的直线组合
        if thera > lowerLimit and thera < upperLimit:
            # 绘制计算需要的霍夫线
            cv2.line(HoughTransformImage2, (x1, y1), (x2, y2), (255, 255, 255),
                     1,
                     lineType=cv2.LINE_AA)
            theras.append(thera)
    saveIntermediateImage("8.霍夫变换得到线条图像", HoughTransformImage1, fileName)
    saveIntermediateImage("9.经过自动阈值筛选的霍夫变换得到线条图像", HoughTransformImage2,
                          fileName)
    # 对经过处理的线的角度进行求均值
    thera = np.mean(theras)
    logging.info("旋转角度：" + ("逆时针" if thera > 0 else "顺时针 ") +
                 str(round(thera, 2)) + " 度")
    # 使用求得的偏移角度均值来对图像进行旋转
    rotateImg = rotate(src, -thera)
    baiduOCRFile = saveFinalImage("旋转矫正处理结果", rotateImg, fileName)
    # 对旋转矫正的图像进行OCR文字识别
    baiduOCR(baiduOCRFile, "旋转矫正处理结果")


def saveIntermediateImage(winName, img, inName):  # 保存计算过程产生照片
    # cv2.imwrite(str(Path("temp/"+inName+"/"+winName+".jpg")), img) 不使用此方法防止中文名称编码混乱
    cv2.imencode('.jpg', img)[1].tofile(
        str(Path("temp/" + inName + "/" + inName + "-" + winName + ".jpg")))
    return str(Path("temp/" + inName + "/" + inName + "-" + winName + ".jpg"))


def saveFinalImage(winName, img, inName):  # 保存计算结果产生照片
    # cv2.imwrite(str(Path("output/"+inName+"/"+winName+".jpg")), img) 不使用此方法防止中文名称编码混乱
    cv2.imencode('.jpg', img)[1].tofile(
        str(Path("output/" + inName + "/" + inName + "-" + winName + ".jpg")))
    return str(Path("output/" + inName + "/" + inName + "-" + winName +
                    ".jpg"))


def getJpgImage(image_path):  # 读取目录下所有的图片
    file_name = glob(image_path + "/*jpg")
    file_name += glob(image_path + "/*png")
    file_name += glob(image_path + "/*bmp")
    file_name += glob(image_path + "/*jpeg")
    sample = []
    for file in file_name:
        sample.append(os.path.basename(file))
    return sample


def delDir(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        logging.warning("文件夹删除出现错误: %s - %s." % (e.filename, e.strerror))


if __name__ == "__main__":
    logging.info("文档自动矫正并可以进行简单识别程序-ZZT")
    # 待处理文件列表
    fileList = getJpgImage(os.path.dirname(os.path.abspath(__file__)))
    # 删除之前的处理结果
    delDir("temp")
    delDir("output")
    # 新建文件夹以存放处理结果
    os.system("mkdir temp")
    os.system("mkdir output")
    # 开始处理文档图像文件
    for file in fileList:
        logging.info("---------------------------------------------------")
        logging.info("开始处理文件" + file + ":")
        documentCorrection(file)
    logging.info("文档图片文件处理完毕，处理结果在optupt文件夹下，处理中间产物在temp文件夹下")
