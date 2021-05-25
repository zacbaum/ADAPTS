import time
import cv2
import mss
import numpy as np

from tensorflow.keras.models import load_model

QA_THRESH = 0.5

with mss.mss() as sct:
    # Part of the screen to capture
    top = 320
    left = 350
    width = 260
    height = 390
    monitor = {"top": top, "left": left, "width": width, "height": height}

    qa_model = load_model("models/quality-cls-50to1-effnetb0.h5", compile=False)

    d_model = load_model("models/fold1-effnetb0.h5", compile=False)

    unet_model = load_model("models/unet.h5", compile=False)

    while True:
        start_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))

        # Grayscale it
        img_gr = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # Rescale to proper size
        img_scaled = cv2.resize(img_gr, (180, 270)) # Update this to be the size of network inputs
        img_scaled_seg = cv2.resize(img_gr, (256, 384)) # Update this to be the size of network inputs

        # Batch dim, channels dim
        img_processed = np.expand_dims(np.expand_dims(img_scaled, axis=0), axis=-1)
        img_processed_seg = np.expand_dims(np.expand_dims(img_scaled_seg, axis=0), axis=-1)
        img_normalized_seg = (img_processed_seg - np.min(img_processed_seg))/np.ptp(img_processed_seg)
        
        qa_pred = qa_model(img_processed.astype(np.float32), training=False)[0]

        if qa_pred.numpy()[0] >= QA_THRESH:
            d_pred = d_model(img_processed.astype(np.float32), training=False)[0]
            print("COVID Likelihood: {:.2f}".format(d_pred.numpy()[0]))

            unet_pred = unet_model(img_normalized_seg.astype(np.float32), training=False)[0]
            unet_pred_rescaled = cv2.resize(np.squeeze(unet_pred), (width, height))
            unet_pred_denormed = (255 * (unet_pred_rescaled - np.min(unet_pred_rescaled)) / np.ptp(unet_pred_rescaled)).astype(np.uint8)        

            added_img = cv2.addWeighted(np.expand_dims(img_gr, axis=-1), 1.0, np.expand_dims(unet_pred_denormed, axis=-1), 0.75, 0)
            cv2.imshow("Capture & Output", added_img)
        
        else:
            print("Poor Image Quality!")    
            cv2.imshow("Capture & Output", img_gr)
        
        #print("fps: {}".format(1 / (time.time() - start_time)))

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break