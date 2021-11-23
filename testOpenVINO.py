import cv2
import imutils
from openvino.inference_engine import IECore
from imutils import paths

TEST_PATH = "../AddressDetection/train"
CONF = 0.4

pColor = (0, 0, 255)  # plate bounding-rect and information color
rectThinkness = 2

lpr_model_xml = "./license-plate-recognition-barrier-0007/license-plate-recognition-barrier-0007.xml"
lpr_model_bin = "./license-plate-recognition-barrier-0007/license-plate-recognition-barrier-0007.bin"
device = "CPU"
ADDRESS = True

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


def drawText(frame, scale, rectX, rectY, rectColor, text):

    textSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)

    top = max(rectY - rectThinkness, textSize[0])

    cv2.putText(
        frame, text, (rectX, top), cv2.FONT_HERSHEY_SIMPLEX, scale, rectColor, 3
    )


def plateRecognition(frame, txtr_execution_net, txtr_input_blob):

    frame_width = frame.shape[1]

    if frame_width > 93:  # Minimal weight in plate-recognition-barrier-0001 is 94
        # Crop a license plate. Do some offsets to better fit a plate.
        blob = cv2.dnn.blobFromImage(frame, size=(94, 24), ddepth=cv2.CV_8U)
        txtr_results = txtr_execution_net.infer(inputs={txtr_input_blob: blob}).get(
            "d_predictions.0"
        )

        content = ""
        for _ in txtr_results[0]:
            if _ == -1:
                break
            elif ADDRESS and _ > 9:
                pass
            else:
                content += items[_]
        print("RESULTADO: ", content)
        drawText(frame, frame_width * 0.008, 0, 0, pColor, content)
    else:
        print("Image too small.")

    showImg = imutils.resize(frame, height=600)
    cv2.imshow("showImg", showImg)


def main():

    ie = IECore()

    txtr_neural_net = ie.read_network(model=lpr_model_xml, weights=lpr_model_bin)
    if txtr_neural_net is not None:
        txtr_input_blob = next(iter(txtr_neural_net.input_info))
        txtr_neural_net.batch_size = 1
        txtr_execution_net = ie.load_network(
            network=txtr_neural_net, device_name=device.upper()
        )

    for imagePath in paths.list_images(TEST_PATH):
        print(imagePath)
        img = cv2.imread(imagePath)
        if img is None:
            continue

        plateRecognition(img, txtr_execution_net, txtr_input_blob)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
