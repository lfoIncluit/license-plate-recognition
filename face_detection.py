import cv2
from openvino.inference_engine import IECore, StatusCode
from imutils import paths, resize
from time import sleep

TEST_PATH = "faceTest"
CONF = 0.4
MODEL_FRAME_SIZE = 640

pColor = (0, 0, 255)
rectThinkness = 2


face_model_xml = "./face-detection-model/face-detection-0206.xml"
face_model_bin = "./face-detection-model/face-detection-0206.bin"


device = "CPU"


def faceDetection(frame, face_execution_net, face_input_blob):

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    face_blob = cv2.dnn.blobFromImage(
        frame, size=(MODEL_FRAME_SIZE, MODEL_FRAME_SIZE), ddepth=cv2.CV_8U
    )

    face_execution_net.requests[0].async_infer({face_input_blob: face_blob})

    while face_execution_net.requests[0].wait(0) != StatusCode.OK:
        sleep(1)

    face_results = face_execution_net.requests[0].output_blobs["boxes"].buffer

    print("FACE: ", face_results)

    if face_results.any():
        for detection in face_results:
            if detection[0] == 0:
                break
            print(detection)
            conf = detection[4]
            if conf < CONF:
                continue
            xmin = int(detection[0] * frame_width / MODEL_FRAME_SIZE)
            ymin = int(detection[1] * frame_height / MODEL_FRAME_SIZE)
            xmax = int(detection[2] * frame_width / MODEL_FRAME_SIZE)
            ymax = int(detection[3] * frame_height / MODEL_FRAME_SIZE)
            xmin = max(0, xmin - 5)
            ymin = max(0, ymin - 5)
            xmax = min(xmax + 5, frame_width - 1)
            ymax = min(ymax + 5, frame_height - 1)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), pColor, rectThinkness)

    showImg = resize(frame, height=750)
    cv2.imshow("showImg", showImg)


def main():

    ie = IECore()

    face_neural_net = ie.read_network(model=face_model_xml, weights=face_model_bin)
    if face_neural_net is not None:
        face_input_blob = next(iter(face_neural_net.input_info))
        face_neural_net.batch_size = 1
        face_execution_net = ie.load_network(
            network=face_neural_net, device_name=device.upper(), num_requests=0
        )

    for name, info in face_neural_net.outputs.items():
        print("\tname: {}".format(name))
        print("\tshape: {}".format(info.shape))
        print("\tlayout: {}".format(info.layout))
        print("\tprecision: {}\n".format(info.precision))

    for imagePath in paths.list_images(TEST_PATH):
        print(imagePath)
        img = cv2.imread(imagePath)
        if img is None:
            continue

        faceDetection(img, face_execution_net, face_input_blob)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
