import time
import cv2
import mss
import numpy as np

from tensorflow.keras.models import load_model

with mss.mss() as sct:
    # Part of the screen to capture
    top = 0
    left = 100
    width = 720
    height = 1080
    monitor = {"top": top, "left": left, "width": width, "height": height}

    model = load_model("models/fold1.h5", compile=False)

    while True:
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))

        # Grayscale it
        img_gr = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # Rescale to proper size
        img_scaled = cv2.resize(img_gr, (width//4, height//4))

        # Batch dim, channels dim
        img_processed = np.expand_dims(np.expand_dims(img_scaled, axis=0), axis=-1)
        
        pred = model(img_processed, training=False)[0]

        print(pred)

        # Display the picture
        cv2.imshow("Capture & Output", img)

        print("fps: {}".format(1 / (time.time() - last_time)))


        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break