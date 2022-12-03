import numpy as np
import cv2
import os
import pandas as pd
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
rtsp_link = os.environ['RTSP_LINK']
rtsp_name = os.environ['RTSP_NAME']
import tensorflow_hub as hub
import tensorflow as tf

def produce_frames(q):
    #get the first frame to calculate size of buffer
    cap = cv2.VideoCapture(rtsp_link)
    success, frame = cap.read()
    shm = SharedMemory(create=True, size=frame.nbytes)
    framebuffer = np.ndarray(frame.shape, frame.dtype, buffer=shm.buf) #could also maybe use array.array instead of numpy, but I'm familiar with numpy
    framebuffer[:] = frame #in case you need to send the first frame to the main process
    q.put(shm) #send the buffer back to main
    q.put(frame.shape) #send the array details
    q.put(frame.dtype)
    try:
        while True:
            cap.read(framebuffer)
    except KeyboardInterrupt:
        pass
    finally:
        shm.close() #call this in all processes where the shm exists
        shm.unlink() #call this in at least one process

def consume_frames(q):
    shm = q.get() #get the shared buffer
    shape = q.get()
    dtype = q.get()
    framebuffer = np.ndarray(shape, dtype, buffer=shm.buf) 
    try:
        while True:
            rgb_tensor = tf.convert_to_tensor(framebuffer, dtype=tf.float32)
            converted_img  = tf.image.convert_image_dtype(framebuffer, tf.float32)[tf.newaxis, ...]   
            rgb_tensor = tf.expand_dims(rgb_tensor , 0)
            height = rgb_tensor.shape[1] 
            width = rgb_tensor.shape[2]
            detector_output = detector(converted_img)
            detector_output = {key:value.numpy() for key,value in detector_output.items()}
            boxes = detector_output['detection_boxes']
            scores = detector_output['detection_scores']
            class_labels = detector_output['detection_class_entities']

            for i in range(scores.shape[0]):
                score = scores[i]
                if score < 0.4:
                    continue
                label = class_labels[i].decode()
                ymin = int(boxes[i][0]*height)
                xmin = int(boxes[i][1]*width)
                ymax = int(boxes[i][2]*height)
                xmax = int(boxes[i][3]*width)
                score_txt = f'{round(score*100,0)}%'
                image_text = label+"-"+score_txt
                framebuffer = cv2.rectangle(framebuffer,(int(xmin), int(ymax)),(int(xmax), int(ymin)),(0,255,0),1)      
                cv2.putText(framebuffer, image_text,(xmin, max(0,ymin-20)), font, 1, (255,255,255), 4, cv2.LINE_AA)
            cv2.imshow(rtsp_name, framebuffer)
            cv2.waitKey(100)
    except KeyboardInterrupt:
        pass
    finally:
        shm.close()

if __name__ == "__main__":
    font = cv2.FONT_HERSHEY_SIMPLEX
    detector = hub.load("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1").signatures['default']
    print("Model Loaded")
    q = Queue()
    producer = Process(target=produce_frames, args=(q,))
    producer.start()
    consume_frames(q)