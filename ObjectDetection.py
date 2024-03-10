import cv2

cap = cv2.VideoCapture(1)


classNames = []
path = 'Object_Detection_Files/coco.names'

with open(path, 'rt') as f:
    classNames = f.read().splitlines()

configPath = 'Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'Object_Detection_Files/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:

    ret, frame = cap.read()
    index, accuracy, bbox = net.detect(frame, confThreshold = 0.5)
    print(index, bbox, accuracy)

    for id, score, box in zip(index.flatten(), accuracy.flatten(), bbox):
        cv2.rectangle(frame, box, color=(0,255,0),thickness=2)
        cv2.putText(frame, classNames[id-1], (box[0]+ 10, box[1]+ 30),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,thickness=2, color=(0,255,0))

    cv2.imshow("Result", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()