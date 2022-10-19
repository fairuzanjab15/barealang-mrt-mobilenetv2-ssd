
"""
Bismillahirohmanirrohim
Ya Allah Mudahkanlah dalam menyusun program ini
Semoga Juara 1 AAMIIN

"""

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 

### Mengimport library
# For tensor
from array import array
from tokenize import cookie_re
# from sys import _asyncgen_hooks
from turtle import delay
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# For Load Model
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# For Detection
import cv2 
import numpy as np
import serial
import time
global cap

ymin = 0
xmin = 0
ymax = 0
xmax = 0
green_x = 0
red_x = 0
tengah = 0

arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.1)
def write_read(inp):
    arduino.write(bytes(inp, 'utf-8'))
"""
Masuk sesi program
Semua dipisah pisah tapi berurutan
"""

### Konfigurasi model
CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'
config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
# config


### Loading Model from Check Point

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

### Real-time Detection

category_index = label_map_util.create_category_index_from_labelmap('Tensorflow/workspace/annotations/label_map.pbtxt')

# Setup capture

cap = cv2.VideoCapture(0)

# Setup Arduino

arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.1)
def write_read(inp):
    arduino.write(bytes(inp, 'utf-8'))
    # time.sleep(0.01)
    data = arduino.readline()
    return data

# Looping camera

while True: 
    ret, frame = cap.read()
    image_np = np.array(frame)

    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = cv2.cvtColor(image_np.copy(), cv2.COLOR_BGR2RGB)



    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=10,
                min_score_thresh=.70,
                agnostic_mode=False)

    
    #pencet Q untuk close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
    
    # Deteksi objek menjadi nilai tugas
    # This is the way I'm getting my coordinates
    boxes = detections['detection_boxes']
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # get scores to get a threshold
    scores = detections['detection_scores']
    # this is set as a default but feel free to adjust it to your needs
    min_score_thresh=.5
    # # iterate over all objects found
    coordinates = []
    # class_id = int(detections['detection_classes'] + 1)



    # # ### PEMISAH ANTARA 1 DAN 2

    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores[i] > min_score_thresh:
            class_id = int(detections['detection_classes'][i] + 1)
            coordinates.append({
                "class_name": category_index[class_id]["name"],
                "box": boxes[i],
                "score": scores[i]
            })

            ymin, xmin, ymax, xmax = boxes[i]

            ## memisahkan nilai pada bola merah dan hijau
            if(class_id == 1):
                w1=int(xmax*640)
                h1=int(ymax*480)
                tL_x1=int(xmin*640)
                tL_y1=int(ymin*480)  

                red_x = int((tL_x1+w1)/2 ) # titik tengah x

                cv2.rectangle(frame, (tL_x1,tL_y1), (w1,h1), (255,0,0), 2)

            elif(class_id == 2):
                w2=int(xmax*640)
                h2=int(ymax*480)
                tL_x2=int(xmin*640)
                tL_y2=int(ymin*480)    
                
                green_x = int((tL_x2+w2)/2) # titik tengah x

                cv2.rectangle(frame, (tL_x2,tL_y2), (w2,h2), (255,255,0), 2)

            else:
                ymin,ymax,xmin,xmax=0

            tengah = int((red_x + green_x)/2)
            cv2.circle(frame, (tengah,240),4,(255,0,255),2)
            # print(tengah)

        #### 2 OBJ CHECK 
            
    if (green_x > 0 and red_x > 0):
                
        if (tengah>300 and tengah<340):
            # pendorong()
            num = "pendorong3"
            # value = write_read(num)
            print('maju3')
            arduino.write(bytes(num, 'utf-8'))
        elif(tengah<300): #belok kiri
            num = "kiri3"
            # value = write_read(num)
            print('kiri3')
            arduino.write(bytes(num, 'utf-8'))
        elif(tengah>340):
            num = "kanan3"
            # value = write_read(num)
            print('kanan3')
            arduino.write(bytes(num, 'utf-8'))


            
                        
            

            
    cv2.imshow("origin", frame)
    cv2.imshow('object detection', image_np_with_detections)
