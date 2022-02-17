from collections import defaultdict
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
from pathlib import Path
import numpy as np


label_map = {   1:  "person",   2:  "bicycle",   3:  "car",   4:  "motorcycle",   5:  "airplane",   6:  "bus",   7:  "train",   8:  "truck",   9:  "boat",   10:  "traffic light",   11:  "fire hydrant",   13:  "stop sign",   14:  "parking meter",   15:  "bench",   16:  "bird",   17:  "cat",   18:  "dog",   19:  "horse",   20:  "sheep",   21:  "cow",   22:  "elephant",   23:  "bear",   24:  "zebra",   25:  "giraffe",   27:  "backpack",   28:  "umbrella",   31:  "handbag",   32:  "tie",   33:  "suitcase",   34:  "frisbee",   35:  "skis",   36:  "snowboard",   37:  "sports ball",   38:  "kite",   39:  "baseball bat",   40:  "baseball glove",   41:  "skateboard",   42:  "surfboard",   43:  "tennis racket",   44:  "bottle",   46:  "wine glass",   47:  "cup",   48:  "fork",   49:  "knife",   50:  "spoon",   51:  "bowl",   52:  "banana",   53:  "apple",   54:  "sandwich",   55:  "orange",   56:  "broccoli",   57:  "carrot",   58:  "hot dog",   59:  "pizza",   60:  "donut",   61:  "cake",   62:  "chair",   63:  "couch",   64:  "potted plant",   65:  "bed",   67:  "dining table",   70:  "toilet",   72:  "tv",   73:  "laptop",   74:  "mouse",   75:  "remote",   76:  "keyboard",   77:  "cell phone",   78:  "microwave",   79:  "oven",   80:  "toaster",   81:  "sink",   82:  "refrigerator",   84:  "book",   85:  "clock",   86:  "vase",   87:  "scissors",   88:  "teddy bear",   89:  "hair drier",   90:  "toothbrush",}


def intersection(self, other):
    a, b = self, other
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    x1 = max(min(ax1, ax2), min(bx1, bx2))
    y1 = max(min(ay1, ay2), min(by1, by2))
    x2 = min(max(ax1, ax2), max(bx1, bx2))
    y2 = min(max(ay1, ay2), max(by1, by2))
    if x1<x2 and y1<y2:
        return np.array([y1, x1, y2, x2])
    else:
        return np.array([0, 0, 0, 0])


def area(a):
    ay1, ax1, ay2, ax2 = a
    return (ay2 - ay1)*(ax2 - ax1)


# detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1")
# detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/d5/1")

detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite3/detection/1")

n_boxes = defaultdict(int)
for path in Path("24").glob("*.jpg"):
    if path.name.startswith("."):
        continue

    img = cv2.imread(str(path))
    height, width, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img)
    # img = cv2.resize(img, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
    img = np.array(img)
    img = np.expand_dims(img, 0)

    # results = detector(img)
    # results = {key:value.numpy() for key,value in results.items()}
    # boxes, scores, classes = results['detection_boxes'][0], results['detection_scores'][0], results['detection_classes'][0]
    # boxes = [np.array([b[0] * height, b[1] * width, b[2] * height, b[3] * width]) for b in boxes]

    results = detector(img)
    results = [x.numpy() for x in results]
    boxes, scores, classes, num_detections = results
    boxes, scores, classes = boxes[0], scores[0], classes[0]
    print(boxes)
    print(scores)
    print(classes)

    box_history = []
    for box, score, cl in zip(boxes, scores, classes):
        if score < 0.2:
            continue

        # dup = False
        # for h in box_history:
        #     intersect_ratio = area(intersection(h, box))/(max(area(box), area(h)))
        #     if intersect_ratio > 0.8:
        #         dup = True
        #         break
        
        # if dup:
        #     continue

        label = label_map[int(cl)]

        ymin, xmin, ymax, xmax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f"{label}", color="red")
        box_history.append(box)
        n_boxes[label] += 1
    plt.show()

print(n_boxes)
