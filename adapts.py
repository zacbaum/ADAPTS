import time
import cv2
import mss
import numpy as np
import tensorflow as tf
import sys

from tensorflow.keras.models import load_model

QA_THRESH = 0.5
COVID_THRESH = 0.5

def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    # We compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), preds

def get_added_gradcam(img, heatmap, alpha=0.1):
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use colormap to the CAM
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Add the images together
    superimposed_img = cv2.addWeighted(img[:, :, :3], 1.0, heatmap, alpha, 0)

    return superimposed_img

use_unet = True if sys.argv[1] == '1' else False
show_fps = True if sys.argv[2] == '1' else False

# Part of the screen to capture
top = 320
left = 350
width = 260
height = 390
monitor = {"top": top, "left": left, "width": width, "height": height}

rescale_factor = 2

window = "ADAPTS"
cv2.namedWindow(window)
cv2.moveWindow(window, left + width + 500, top - 200)

with mss.mss() as sct:

    qa_model = load_model("models/quality-cls-50to1-effnetb0.h5", compile=False)
    qa_model.call = tf.function(qa_model.call)

    d_model = load_model("models/fold1-effnetb0.h5", compile=False)
    d_model.call = tf.function(d_model.call)

    if not use_unet:
	    # Create a model that maps the input image to the activations
	    # of the last conv layer as well as the output predictions
	    d_model_grads = tf.keras.models.Model(
	        [d_model.inputs], [d_model.get_layer('top_conv').output, d_model.output] # top_conv is specific to effnetb0
	    )
    if use_unet:
        unet_model = load_model("models/unet.h5", compile=False)
        unet_model.call = tf.function(unet_model.call)

    while True:
        start_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))

        # Grayscale it
        img_gr = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # Rescale to proper size
        img_scaled = cv2.resize(
        	img_gr,
        	(qa_model.input_shape[2], qa_model.input_shape[1]), # cv2 uses flipped h,w from tf
        	interpolation=cv2.INTER_NEAREST,
        )
        if use_unet:
            img_scaled_seg = cv2.resize(
            	img_gr,
            	(unet_model.input_shape[2], unet_model.input_shape[1]), # cv2 uses flipped h,w from tf
            	interpolation=cv2.INTER_NEAREST,
            )
        # Add batch dim, channels dim
        img_processed = img_scaled[np.newaxis, ..., np.newaxis]
        if use_unet:
            img_processed_seg = img_scaled_seg[np.newaxis, ..., np.newaxis]
            img_normalized_seg = (img_processed_seg - np.min(img_processed_seg)) / np.ptp(img_processed_seg)
                
        qa_pred = qa_model(img_processed, training=False)[0]
        qa_str = "QA: {:.2f}".format(np.squeeze(qa_pred))

        if qa_pred[0] >= QA_THRESH:
            if use_unet:
                unet_pred = unet_model(img_normalized_seg, training=False)[0]
                unet_pred_rescaled = cv2.resize(
                	np.squeeze(unet_pred),
                	(width, height),
                	interpolation=cv2.INTER_NEAREST,
                )
                unet_pred_denormed = (255 * (unet_pred_rescaled - np.min(unet_pred_rescaled)) / np.ptp(unet_pred_rescaled)).astype(np.uint8)        
                img = cv2.addWeighted(img_gr[..., np.newaxis], 1.0, unet_pred_denormed[..., np.newaxis], 0.25, 0)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                # If we use the unet, we also need to get the diagnosistic likelihood
                d_pred = d_model(img_processed, training=False)[0]
            if not use_unet:
                # If we don't use the unet, we get the diagnositic likelihood from the gradcam model
                heatmap, d_pred = make_gradcam_heatmap(img_processed, d_model_grads)
                img = get_added_gradcam(img, heatmap)
            covid_str = "COVID: {:.2f}".format(np.squeeze(d_pred))
        else:
            covid_str = ""  

        if rescale_factor != 1:
            img = cv2.resize(
            	img,
            	(img.shape[1] * rescale_factor, img.shape[0] * rescale_factor),
            	interpolation=cv2.INTER_NEAREST,
            )
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        text_color_white = (255, 255, 255)
        text_color_green = (0, 255, 127) # BGR
        text_color_red = (0, 8, 255) # BGR
        px_width = 2

        cv2.putText(
            img, 
            qa_str,
            (5, height * rescale_factor - 10),
            font,
            scale,
            text_color_green if np.squeeze(qa_pred) >= QA_THRESH else text_color_red,
            px_width,
        )

        if covid_str is not "":
	        cv2.putText(
	            img, 
	            covid_str,
	            (5, height * rescale_factor - 40),
	            font,
	            scale,
	            text_color_red if np.squeeze(d_pred) >= COVID_THRESH else text_color_green,
	            px_width,
	        )

        if show_fps: 
            fps_str = "FPS: {:.1f}".format(1 / (time.time() - start_time))
            cv2.putText(
                img, 
                fps_str,
                (width * rescale_factor - 170, height * rescale_factor - 10),
                font,
                scale,
                text_color_white,
                px_width,
            )

        cv2.imshow(window, img)
        
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break