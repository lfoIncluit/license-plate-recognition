import numpy as np
import cv2
import imutils
from openvino.inference_engine import IECore
from imutils import paths

bShowColor = False

TEST_PATH = "testImgs"

FRAME_WIDTH = 0
FRAME_HEIGHT = 0

vColor = (0, 255, 0)  # vehicle bounding-rect and information color
pColor = (0, 0, 255)  # plate bounding-rect and information color
rectThinkness = 2

items = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "<Anhui>",
    "<Beijing>",
    "<Chongqing>",
    "<Fujian>",
    "<Gansu>",
    "<Guangdong>",
    "<Guangxi>",
    "<Guizhou>",
    "<Hainan>",
    "<Hebei>",
    "<Heilongjiang>",
    "<Henan>",
    "<HongKong>",
    "<Hubei>",
    "<Hunan>",
    "<InnerMongolia>",
    "<Jiangsu>",
    "<Jiangxi>",
    "<Jilin>",
    "<Liaoning>",
    "<Macau>",
    "<Ningxia>",
    "<Qinghai>",
    "<Shaanxi>",
    "<Shandong>",
    "<Shanghai>",
    "<Shanxi>",
    "<Sichuan>",
    "<Tianjin>",
    "<Tibet>",
    "<Xinjiang>",
    "<Yunnan>",
    "<Zhejiang>",
    "<police>",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

lpr_model_xml = "./license-plate-recognition-barrier-0007/license-plate-recognition-barrier-0007.xml"
lpr_model_bin = "./license-plate-recognition-barrier-0007/license-plate-recognition-barrier-0007.bin"
device = "CPU"
ie = IECore()

neural_net = ie.read_network(model=lpr_model_xml, weights=lpr_model_bin)
if neural_net is not None:
    input_blob = next(iter(neural_net.input_info))
    output_blob = next(iter(neural_net.outputs))
    neural_net.batch_size = 1
    execution_net = ie.load_network(network=neural_net, device_name=device.upper())
n, c, h, w = neural_net.input_info[input_blob].input_data.shape

# https://docs.openvinotoolkit.org/2019_R1/_vehicle_license_plate_detection_barrier_0106_description_vehicle_license_plate_detection_barrier_0106.html
pd_net = cv2.dnn.readNet(
    "./vehicle-license-plate-detection-barrier-0106/vehicle-license-plate-detection-barrier-0106.xml",
    "./vehicle-license-plate-detection-barrier-0106/vehicle-license-plate-detection-barrier-0106.bin",
)

pd_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
pd_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def drawText(frame, scale, rectX, rectY, rectColor, text):

    textSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)

    top = max(rectY - rectThinkness, textSize[0])

    cv2.putText(
        frame, text, (rectX, top), cv2.FONT_HERSHEY_SIMPLEX, scale, rectColor, 3
    )


def plateRecognition(frame):

    # name: "input" , shape: [1x3x300x300] - An input image in the format [BxCxHxW], where:
    pd_blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
    pd_net.setInput(pd_blob)
    out_pb = pd_net.forward()
    global cur_request_id

    # An every detection is a vector [imageId, classId, conf, x, y, X, Y]
    for detection in out_pb.reshape(-1, 7):
        conf = detection[2]
        if conf < 0.4:
            continue

        classId = int(detection[1])
        
        if classId == 2:  # plate
            xmin = int(detection[3] * FRAME_WIDTH)
            ymin = int(detection[4] * FRAME_HEIGHT)
            xmax = int(detection[5] * FRAME_WIDTH)
            ymax = int(detection[6] * FRAME_HEIGHT)

            xmin = max(0, xmin - 10)
            ymin = max(0, ymin - 10)
            xmax = min(xmax + 10, FRAME_WIDTH - 1)
            ymax = min(ymax + 10, FRAME_HEIGHT - 1)

            rectW = xmax - xmin
            if rectW < 93:  # Minimal weight in plate-recognition-barrier-0001 is 94
                continue

            # Crop a license plate. Do some offsets to better fit a plate.
            lpImg = frame[ymin : ymax + 1, xmin : xmax + 1]
            blob = cv2.dnn.blobFromImage(lpImg, size=(94, 24), ddepth=cv2.CV_8U)
            results = execution_net.infer( inputs={input_blob: blob}).get("d_predictions.0")
        
            content = ""
            for _ in results[0]:
                if _ == -1:
                    break
                content += items[_]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), pColor, rectThinkness)
            drawText(frame, rectW * 0.008, xmin, ymin, pColor, content)
        else:
            continue

    showImg = imutils.resize(frame, height=600)
    cv2.imshow("showImg", showImg)


for imagePath in paths.list_images(TEST_PATH):
    print(imagePath)
    img = cv2.imread(imagePath)
    if img is None:
        continue
    FRAME_WIDTH = img.shape[1]
    FRAME_HEIGHT = img.shape[0]

    plateRecognition(img)
    cv2.waitKey(0)
