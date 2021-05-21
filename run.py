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

    qa_model = load_model("models/quality-cls-50to1.h5", compile=False)

    d_model = load_model("models/fold1.h5", compile=False)

    while True:
        #last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))

        # Grayscale it
        img_gr = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # Rescale to proper size
        img_scaled = cv2.resize(img_gr, (180, 270)) # Update this to be the size of network inputs

        # Batch dim, channels dim
        img_processed = np.expand_dims(np.expand_dims(img_scaled, axis=0), axis=-1)
        
        qa_pred = qa_model(img_processed.astype(np.float32), training=False)[0]

        if qa_pred.numpy()[0] >= QA_THRESH:
            d_pred = d_model(img_processed.astype(np.float32), training=False)[0]
            print("COVID Confidence: {:3f}".format(d_pred.numpy()[0]))
        else:
            print("Poor Image Quality!")

        # Display the picture
        cv2.imshow("Capture & Output", img)

        #print("fps: {}".format(1 / (time.time() - last_time)))


        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break