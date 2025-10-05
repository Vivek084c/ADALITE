# import numpy as np
# import cv2
# import time
# import sys
# import signal
# import os
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from tensorflow.lite.python.interpreter import Interpreter as tflite

# # ---------------------------------------------------------------------
# # Configuration
# # ---------------------------------------------------------------------
# MODEL_PATH = "Raspberry_Pi_5/ADALITE_TFLITE.tflite"
# MODEL_INPUT_HEIGHT = 256
# MODEL_INPUT_WIDTH = 256
# DISPLAY_WIDTH_SINGLE = 256
# DISPLAY_HEIGHT_SINGLE = 256

# # ---------------------------------------------------------------------
# # Depth-map helpers
# # ---------------------------------------------------------------------
# def preprocess_frame(frame, inputWidth, inputHeight):
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img_resized = cv2.resize(img_rgb, (inputWidth, inputHeight),
#                              interpolation=cv2.INTER_CUBIC).astype(np.float32)
#     mean = np.array([0.485, 0.456, 0.406])
#     std  = np.array([0.229, 0.224, 0.225])
#     normalized_img = ((img_resized / 255.0 - mean) / std).astype(np.float32)
#     return normalized_img, img_rgb

# def plot_depth_with_points_fixed(depth_map, points, title, cmap='plasma', H=256, W=256):
#     dpi = 100
#     fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
#     ax.imshow(depth_map, cmap=cmap)
#     for (x, y) in points:
#         value = f"{depth_map[y, x]:.2f}"
#         ax.add_patch(plt.Circle((x, y), radius=4, color='white', alpha=0.8))
#         ax.text(x, y, value, color='black', fontsize=6, ha='center', va='center')
#     ax.set_title(title)
#     ax.axis('off')
#     fig.tight_layout(pad=0)
#     canvas = FigureCanvas(fig)
#     canvas.draw()
#     img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
#     img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
#     plt.close(fig)
#     return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

# # ---------------------------------------------------------------------
# # Main loop (Mac version: uses cv2.VideoCapture)
# # ---------------------------------------------------------------------
# def main():
#     # graceful shutdown
#     running = True
#     def cleanup_handler(signum=None, frame=None):
#         nonlocal running
#         running = False
#     signal.signal(signal.SIGINT, cleanup_handler)
#     signal.signal(signal.SIGTERM, cleanup_handler)

#     # depth-map overlay points
#     m, c = (-0.37965089082717896, 14.945058822631836)
#     xs = np.linspace(0, MODEL_INPUT_WIDTH - 1, 6, dtype=int)
#     ys = np.linspace(0, MODEL_INPUT_HEIGHT - 1, 6, dtype=int)
#     points = [(x, y) for y in ys for x in xs]

#     print("Loading TFLite model…")
#     interpreter = tflite(model_path=MODEL_PATH, num_threads=4)
#     interpreter.allocate_tensors()
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     print("Model loaded.")

#     # ---- Webcam setup ----
#     cap = cv2.VideoCapture(0)  # default Mac camera
#     if not cap.isOpened():
#         print("Could not open webcam")
#         return

#     print("Running real-time depth estimation. Press Ctrl+C to quit.")
#     frame_count, start_time = 0, time.time()

#     while running:
#         ret, frame_original_bgr = cap.read()
#         if not ret:
#             print("Frame capture failed.")
#             break

#         # crop to square for model
#         h_orig, w_orig, _ = frame_original_bgr.shape
#         crop_size = min(h_orig, w_orig)
#         sx = (w_orig - crop_size) // 2
#         sy = (h_orig - crop_size) // 2
#         frame_cropped_bgr = frame_original_bgr[sy:sy+crop_size, sx:sx+crop_size]

#         x_norm, raw_img_rgb = preprocess_frame(frame_cropped_bgr,
#                                                MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)

#         interpreter.set_tensor(input_details[0]['index'], np.expand_dims(x_norm, axis=0))
#         interpreter.invoke()
#         pred_depth = interpreter.get_tensor(output_details[0]['index'])
#         pred_depth = np.squeeze(pred_depth, axis=0).squeeze(-1)

#         if pred_depth.shape != (MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH):
#             pred_depth = cv2.resize(pred_depth, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT),
#                                     interpolation=cv2.INTER_CUBIC)
#         aligned_depth = m * pred_depth + c

#         aligned_vis_rgb = plot_depth_with_points_fixed(aligned_depth, points,
#                                                        "Aligned Depth", cmap='magma',
#                                                        H=MODEL_INPUT_HEIGHT, W=MODEL_INPUT_WIDTH)
#         original_vis_rgb = cv2.resize(raw_img_rgb, (DISPLAY_WIDTH_SINGLE, DISPLAY_HEIGHT_SINGLE))
#         combined_display_rgb = np.hstack([original_vis_rgb, aligned_vis_rgb])

#         # Show in an OpenCV window (Mac-friendly)
#         cv2.imshow("Depth Estimation (press q to quit)", cv2.cvtColor(combined_display_rgb, cv2.COLOR_RGB2BGR))
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         frame_count += 1
#         if time.time() - start_time > 1.5:
#             fps = frame_count / (time.time() - start_time)
#             print(f"FPS: {fps:.2f}")
#             frame_count, start_time = 0, time.time()

#     cap.release()
#     cv2.destroyAllWindows()
#     print("Cleanup complete.")

# if __name__ == "__main__":
#     main()

