
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
from multiprocessing.resource_sharer import stop
from pyexpat.model import XML_CQUANT_REP
# from termios import CR2, CR3
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
red_x=""
green_x=""
yllw_x=""
tengah= ""

def zero(): 
    global red_x, green_x, tengah, yllw_x, cx_red1, cx_green2, cx_yllw3, cy_yllw3, cy_red1, cy_green2
    red_x = -1
    green_x = -1
    yllw_x = -1

    cx_red1 = -1
    cx_green2 = -1 
    cx_yllw3 = -1
    tengah = -1


# Setup Arduino

# arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.1)
# def write_read(inp):
#     arduino.write(bytes(inp, 'utf-8'))

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
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

### Real-time Detection

category_index = label_map_util.create_category_index_from_labelmap('Tensorflow/workspace/annotations/label_map.pbtxt')

# Setup capture
Xn_red = [1]*20
Yn_red = [1]*20
Xm_red = [1]*20
Ym_red = [1]*20
tampung1 = [1]*20

Xn_green = [1]*20
Yn_green = [1]*20
Xm_green = [1]*20
Ym_green = [1]*20
tampung2 = [1]*20 

Xn_yllw = [1]*20
Yn_yllw = [1]*20
Xm_yllw = [1]*20
Ym_yllw = [1]*20
tampung3 = [1]*20 

cx_green2 = -1
cy_green2 = -1
cx_red1 = -1
cy_red1 = -1
cx_yllw3 = -1
cy_yllw3 = -1

# Looping camera
w1      = 0
h1      = 0
tL_x1   = 0
tL_y1   = 0 

cap = cv2.VideoCapture('bkk.mp4')
while True: 
    # global red_x
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



    # Deteksi objek menjadi nilai tugas
    # This is the way I'm getting my coordinates
    boxes = detections['detection_boxes']

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=10,
                min_score_thresh=.40,
                agnostic_mode=False)

    
    
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


    # TITIK TITIK


    # Xn_red = [1]*10
    # Yn_red = [1]*10
    # Xm_red = [1]*10
    # Ym_red = [1]*10
    # tampung1 = [1]*10

    # Xn_green = [1]*10
    # Yn_green = [1]*10
    # Xm_green = [1]*10
    # Ym_green = [1]*10
    # tampung2 = [1]*10 
    
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
                w1      =int(xmax*640)
                h1      =int(ymax*480)
                tL_x1   =int(xmin*640)
                tL_y1   =int(ymin*480)  

                
                red_x = int((tL_x1+w1)/2 ) # titik tengah red
                
                # cv2.rectangle(frame, (tL_x1,tL_y1), (w1,h1), (255,0,0), 2) #kotak red
            
                # print(tL_x1, tL_y1, luas_red)
                Xn_red[i] = tL_x1
                Yn_red[i] = tL_y1
                Xm_red[i] = w1
                Ym_red[i] = h1

                
                for z in range(20):
                    tampung1[z] = (Xm_red[z] - Xn_red[z])*(Ym_red[z] - Yn_red[z]) ##luas nilai   
                    # print(tampung1)
                ### mensortir
                s = np.array(tampung1) #masukin ke s
                sort_index = np.flip(np.argsort(s)) # s masukin ke sort_index
                # print(sort_index)

                cx_red1 = int((Xn_red[sort_index[0]] + Xm_red[sort_index[0]])/2) #xy hasil sort_index
                cy_red1 = int((Yn_red[sort_index[0]] + Ym_red[sort_index[0]])/2) #wh jasil sort_index
                print(tampung1)
                
            
            elif(class_id == 2): 
                w2=int(xmax*640)
                h2=int(ymax*480)
                tL_x2=int(xmin*640)
                tL_y2=int(ymin*480)    
                
                green_x = int((tL_x2+w2)/2) # titik tengah x
                
                # cv2.rectangle(frame, (tL_x2,tL_y2), (w2,h2), (255,255,0), 2)
          
                Xn_green[i] = tL_x2
                Yn_green[i] = tL_y2
                Xm_green[i] = w2
                Ym_green[i] = h2


                for z in range(20):
                    tampung2[z] = (Xm_green[z] - Xn_green[z])*(Ym_green[z] - Yn_green[z]) ##luas nilai
                    # print(tampung2)
                s = np.array(tampung2) #masukin ke s
                sort_index = np.flip(np.argsort(s)) # s masukin ke sort_index
                print(sort_index)
                cx_green2 = int((Xn_green[sort_index[0]] + Xm_green[sort_index[0]])/2) #xy hasil sort_index
                cy_green2 = int((Yn_green[sort_index[0]] + Ym_green[sort_index[0]])/2) #wh jasil sort_index
                print(tampung2)
            
            elif (class_id == 3):
                # mendapatkan nilai deteksi boxes
                w3=int(xmax*640)
                h3=int(ymax*480)
                tL_x3=int(xmin*640)
                tL_y3=int(ymin*480)

                yllw_x = int((tL_x3+w3)/2) # titik tengah x    

                # Mengalamatkan luas yang diatas kedalam Xn,Yn,Xm,Wm
                Xn_yllw[i] = tL_x3
                Yn_yllw[i] = tL_y3
                Xm_yllw[i] = w3 
                Ym_yllw[i] = h3

                # ayllw_x = ((Xn_green[0]+))

                for z in range(20):
                    tampung3[z] = (Xm_yllw[z] - Xn_yllw[z])*(Ym_yllw[z] - Yn_yllw[z]) ##luas nilai

                s = np.array(tampung3) #masukin ke s
                sort_index = np.flip(np.argsort(s)) # s masukin ke sort_index
                # print(sort_index)
                cx_yllw3 = int((Xn_yllw[sort_index[0]] + Xm_yllw[sort_index[0]])/2) #xy hasil sort_index
                cy_yllw3 = int((Yn_yllw[sort_index[0]] + Ym_yllw[sort_index[0]])/2) #wh jasil sort_index
                # print(tampung3[0])
                
                cv2.circle(image_np_with_detections,(cx_yllw3,cy_yllw3),4,(0,255,255),2) 
                cv2.rectangle(image_np_with_detections, (Xn_yllw[sort_index[0]],Yn_yllw[sort_index[0]]), (Xm_yllw[sort_index[0]],Ym_yllw[sort_index[0]]), (5,255,200), 2)
                print(tampung3)
                

            else:
                    ymin,ymax,xmin,xmax= 0
                    green_x = -1
                    red_x = -1
                    yllw_x = -1
                    
                    cx_green2 = -1
                    cy_green2 = -1
                    cx_red1 = -1
                    cy_red1 = -1
                    cx_yllw3 = -1
                    cy_yllw3 = -1
    
    
    """
    Perintah belok untuk kanan dan kiri tanpa sortir
    """
    ## Terdeteksi 2 objek
    # if (green_x != -1 and red_x != -1 and yllw_x == -1):   
        
        # print(tengah)
        # num = "pendorong15"
        # write_read(num)
        # print('MAJU')  
 
    ## Deteksi hijau
    # elif (green_x != -1 and red_x == -1 and yllw_x == -1): 
            # num = "R12"
            # write_read(num)
            # print('KANAN')

    ## Deteksi merah        
    # elif (green_x == -1 and red_x != -1 and yllw_x == -1 ): 
            # num = "L12"
            # write_read(num)
            # print('KIRI')

    ## Deteksi tiang Kuning
    # elif (green_x == -1 and red_x == -1 and yllw_x != -1):
        # if(tampung3[0] > 20000): ## PERINTAH TURUN
            # num = "Stay3"
            # write_read(num)
            # print('TURUN')

    
    # else:
            # num = "stopazimuth"
            # write_read(num)
            # print('STOP')  
    

    """
    PERINTAH KEDUA PAKAI (CX Center x dan Cy Center Y)
    digunakan untuk mensortir bola paling besar
    """

    # if (green_x != -1 and red_x != -1 and yllw_x == -1):  
    #     if(tampung1[0] > 1000 and tampung2[0] > 1000):
    #         num = "pendorong10"
    #         write_read(num)
    #         print('MAJU')  

        
    # # # ## Deteksi hijau
    # elif (cx_green2 != -1 and cx_red1 == -1): 
    #      if(tampung2[0] > 50000):
    #         num = "Kanan10"
    #         write_read(num)
    #         print('KANAN10')

    #      elif (tampung2[0] > 40000):
    #         num = "Kanan9"
    #         write_read(num)
    #         print('KANAN9')
 
    #      elif (tampung2[0] > 35000):
    #         num = "Kanan8"
    #         write_read(num)
    #         print('KANAN8')

    #      elif (tampung2[0] > 30000):
    #         num = "Kanan7"
    #         write_read(num)
    #         print('KANAN7')

    #      elif (tampung2[0] > 25000):
    #         num = "Kanan6"
    #         write_read(num)
    #         print('KANAN6')

    #      elif (tampung2[0] > 20000):
    #         num = "Kanan5"
    #         write_read(num)
    #         print('KANAN5')

    #      elif (tampung2[0] > 15000):
    #         num = "Kanan4"
    #         write_read(num)
    #         print('KANAN4')

    #      elif (tampung2[0] > 10000):
    #         num = "Kanan3"
    #         write_read(num)
    #         print('KANAN3')

    #      elif (tampung2[0] > 5000):
    #         num = "Kanan2"
    #         write_read(num)
    #         print('KANAN2')

    #      elif (tampung2[0] > 1000):
    #         num = "Kanan1"
    #         write_read(num)
    #         print('KANAN1')

    # # # ## Deteksi merah        
    # elif (cx_green2 == -1 and cx_red1 != -1): 
    #     if(tampung1[0] > 50000):
    #         num = "Kiri10"
    #         write_read(num)
    #         print('KIRI10')
    #     elif (tampung1[0] > 40000):
    #         num = "Kiri9"
    #         write_read(num)
    #         print('KIRI9')           
    #     elif (tampung1[0] > 35000):
    #         num = "Kiri8"
    #         write_read(num)
    #         print('KIRI8')
    #     elif (tampung1[0] > 30000):
    #         num = "Kiri7"
    #         write_read(num)
    #         print('KIRI7')
    #     elif (tampung1[0] > 25000):
    #         num = "Kiri6"
    #         write_read(num)
    #         print('KIRI6')
    #     elif (tampung1[0] > 20000):
    #         num = "Kiri5"
    #         write_read(num)
    #         print('KIRI5')
    #     elif (tampung1[0] > 15000):
    #         num = "Kiri4"
    #         write_read(num)
    #         print('KIRI4')
    #     elif (tampung1[0] > 10000):
    #         num = "Kiri3"
    #         write_read(num)
    #         print('KIRI3')
    #     elif (tampung1[0] > 5000):
    #         num = "Kiri2"
    #         write_read(num)
    #         print('KIRI2') 
    #     elif (tampung1[0] > 100):
    #         num = "Kiri1"
    #         write_read(num)
    #         print('KIRI1')     
    

    # # elif (green_x == -1 and red_x == -1 and yllw_x != -1):
    # #         if(tampung3[0] > 30000):
    # #             num = "Stay2"
    # #             write_read(num)
    # #             print('NYELAM AJA') 
    # # elif (green_x == -1 and red_x != -1 and yllw_x != -1):
    # #         if(tampung3[0] > 30000):
    # #             num = "Stay4"
    # #             write_read(num)
    # #             print('NYELAM KIRI')  
    # # elif (green_x != -1 and red_x == -1 and yllw_x != -1):
    # #         if(tampung3[0] > 30000):
    # #             num = "Stay3"
    # #             write_read(num)
    # #             print('NYELAM KANAN')  
    # # elif (green_x != -1 and red_x != -1 and yllw_x != -1):
    # #         if(tampung3[0] > 30000):
    # #             num = "Stay2"
    # #             write_read(num)
    # #             print('NYELAM AJA')    
    # else:
    #         num = "stopazimuth"
    #         write_read(num)
    #         print('STOP')  
    

    #pencet Q untuk close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

    print("hijau =", green_x)
    print("merah = ",red_x) 
    print("kuning = ",cx_yllw3) 
    zero()
    cv2.imshow('object detection', image_np_with_detections)
   






        
