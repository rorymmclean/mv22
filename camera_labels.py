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
            rgb_tensor = tf.convert_to_tensor(framebuffer, dtype=tf.uint8)
            rgb_tensor = tf.expand_dims(rgb_tensor , 0)
            response = detector(rgb_tensor)
            boxes, scores, classes, num_detections = detector(rgb_tensor)
            pred_labels = classes.numpy().astype('int')[0] 
            pred_labels = [labels[i] for i in pred_labels]
            pred_boxes = boxes.numpy()[0].astype('float32')
            pred_scores = scores.numpy()[0]
            for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
                if score < 0.4:
                    continue
                ymin = int(ymin)
                ymax = int(ymax)
                xmin = int(xmin)
                xmax = int(xmax)
                score_txt = f'{round(score*100,0)}%'
                image_text = label+"-"+score_txt
                framebuffer = cv2.rectangle(framebuffer,(int(xmin), int(ymax)),(int(xmax), int(ymin)),(0,255,0),2)      
                cv2.putText(framebuffer, image_text,(xmin, max(0,ymin-20)), font, 1.5, (255,255,255), 4, cv2.LINE_AA)
                # print(image_text)
            cv2.imshow(rtsp_name, framebuffer)
            cv2.waitKey(100)
    except KeyboardInterrupt:
        pass
    finally:
        shm.close()

if __name__ == "__main__":
    font = cv2.FONT_HERSHEY_SIMPLEX
    detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1")
    # detector = hub.load("/Users/rorymclean/Projects/mv22/efficientdet_lite1_detection_1/")
    labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
    labels = labels['OBJECT (2017 REL.)']
    print("Model Loaded")
    q = Queue()
    producer = Process(target=produce_frames, args=(q,))
    producer.start()
    consume_frames(q)