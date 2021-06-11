import time
import cv2
import mss
import numpy as np
import tensorflow as tf
import sys

from tensorflow.keras.models import load_model

QA_THRESH = 0.5

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
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

def get_added_gradcam(img, heatmap, alpha=0.2):
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use colormap to the CAM
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    
    # Add the images together
    superimposed_img = cv2.addWeighted(img[:, :, :3], 1.0, heatmap, alpha, 0)

    return superimposed_img

use_unet = True if sys.argv[1] == '1' else False
show_fps = True if sys.argv[2] == '1' else False

print("\n")
print("\n")
print("#######################################")
print("\n")
print("ADAPTS is starting up... Please wait...")
print("\n")
print("#######################################")
print("\n")
print("\n")

with mss.mss() as sct:
    # Part of the screen to capture
    top = 320
    left = 350
    width = 260
    height = 390
    monitor = {"top": top, "left": left, "width": width, "height": height}

    qa_model = load_model("models/quality-cls-50to1-effnetb0.h5", compile=False)
    qa_model.call = tf.function(qa_model.call, experimental_relax_shapes=True)

    d_model = load_model("models/fold1-effnetb0.h5", compile=False)
    d_model.call = tf.function(d_model.call, experimental_relax_shapes=True)

    if use_unet:
        unet_model = load_model("models/unet.h5", compile=False)
        unet_model.call = tf.function(unet_model.call, experimental_relax_shapes=True)

    while True:
        start_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))

        # Grayscale it
        img_gr = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # Rescale to proper size
        img_scaled = cv2.resize(img_gr, (180, 270)) # Update this to be the size of network inputs
        if use_unet:
            img_scaled_seg = cv2.resize(img_gr, (256, 384)) # Update this to be the size of network inputs

        # Batch dim, channels dim
        img_processed = np.expand_dims(np.expand_dims(img_scaled, axis=0), axis=-1)
        if use_unet:
            img_processed_seg = np.expand_dims(np.expand_dims(img_scaled_seg, axis=0), axis=-1)
            img_normalized_seg = (img_processed_seg - np.min(img_processed_seg))/np.ptp(img_processed_seg)
                
        qa_pred = qa_model(img_processed.astype(np.float32), training=False)[0]

        print_str = ""

        if qa_pred[0] >= QA_THRESH:
            
            if use_unet:
                unet_pred = unet_model(img_normalized_seg.astype(np.float32), training=False)[0]
                unet_pred_rescaled = cv2.resize(np.squeeze(unet_pred), (width, height))
                unet_pred_denormed = (255 * (unet_pred_rescaled - np.min(unet_pred_rescaled)) / np.ptp(unet_pred_rescaled)).astype(np.uint8)        
                added_img = cv2.addWeighted(np.expand_dims(img_gr, axis=-1), 1.0, np.expand_dims(unet_pred_denormed, axis=-1), 0.75, 0)
                # If we use the unet, we also need to get the diagnosistic likelihood
                d_pred = d_model(img_processed.astype(np.float32), training=False)[0]
            
            if not use_unet:
                # If we don't use the unet, we get the diagnositic likelihood from the gradcam outputs
                heatmap, d_pred = make_gradcam_heatmap(img_processed, d_model, 'top_conv')
                added_img = get_added_gradcam(img, heatmap)

            print_str += "COVID Likelihood: {:.2f}".format(np.squeeze(d_pred))
            cv2.imshow("Capture & Output", added_img)
        
        else:
            print_str += "Poor Image Quality!"  
            cv2.imshow("Capture & Output", img)
        
        if show_fps: print_str += " --- FPS: {:.2f}".format(1 / (time.time() - start_time))

        print(print_str)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break