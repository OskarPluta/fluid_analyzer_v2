import dearpygui.dearpygui as dpg
import cv2
import numpy as np

IMG_WIDTH = 640
IMG_HEIGHT = 480

# Initialize webcam
cap = cv2.VideoCapture("xd.MP4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Try to set webcam resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

# Read and process the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()

# Resize frame and convert to RGB
frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Initialize Dear PyGui
dpg.create_context()
dpg.create_viewport(title='Fluid Analyzer', width=1000, height=1000, decorated=True)  # Remove viewport decorations
dpg.setup_dearpygui()

# Mouse callback function to display click position
def mouse_callback(sender, app_data):
    mouse_pos = dpg.get_mouse_pos()
    dpg.set_value("mouse_pos_text", f"Last Click: ({int(mouse_pos[0])}, {int(mouse_pos[1])})")

# Create texture with normalized data
texture_data = (frame.astype(np.float32) / 255.0).ravel()
with dpg.texture_registry():
    dpg.add_raw_texture(
        width=IMG_WIDTH,
        height=IMG_HEIGHT,
        default_value=texture_data,
        format=dpg.mvFormat_Float_rgb,
        tag="video_texture"
    )

# Create a theme to remove window padding
with dpg.theme() as no_padding_theme:
    with dpg.theme_component(dpg.mvWindowAppItem):
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0, 0, category=dpg.mvThemeCat_Core)

# Video window
with dpg.window(tag="video_window") as window:
    dpg.set_item_width(window, 640)
    dpg.set_item_height(window, 480)
    dpg.set_item_pos(window, (0, 0))
    dpg.configure_item(window, no_title_bar=True, no_scrollbar=True, no_resize=True, no_move=True, no_close=True)
    dpg.add_image("video_texture", width=640, height=480, pos=(0, 0))  # Explicitly size and position image
    dpg.bind_item_theme(window, no_padding_theme)  # Apply no-padding theme

# Window2 with click position display
with dpg.window() as window2:
    dpg.add_text("Last Click: None", tag="mouse_pos_text")
    dpg.set_item_width(window2, 1000-640)
    dpg.set_item_height(window2, 480)
    dpg.set_item_pos(window2, (640, 0))
    dpg.configure_item(window2, no_title_bar=True, no_move=True, no_close=True)
    dpg.bind_item_theme(window2, no_padding_theme)

# Other windows
with dpg.window() as window3:
    dpg.add_text("Hello, World!")
    dpg.set_item_width(window3, 640)
    dpg.set_item_height(window3, 1000-480)
    dpg.set_item_pos(window3, (0, 480))
    dpg.configure_item(window3, no_title_bar=True, no_move=True, no_close=True)
    dpg.bind_item_theme(window3, no_padding_theme)

with dpg.window() as window4:
    dpg.add_text("Hello, World!")
    dpg.set_item_width(window4, 1000-640)
    dpg.set_item_height(window4, 1000-480)
    dpg.set_item_pos(window4, (640, 480))
    dpg.configure_item(window4, no_title_bar=True, no_move=True, no_close=True)
    dpg.bind_item_theme(window4, no_padding_theme)

# Bind mouse click callback (left-click)
with dpg.handler_registry():
    dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=mouse_callback)

# Show viewport

dpg.show_viewport()

def mockupfunction(frame):
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv2.putText(frame, "Contours", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame

# Main loop
while dpg.is_dearpygui_running():
    ret, frame = cap.read()

    frame = mockupfunction(frame)

    if not ret:
        print("End of video or error reading frame.")
        break
    
    # Resize frame and convert to RGB
    # frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    texture_data = (frame.astype(np.float32) / 255.0).ravel()
    dpg.set_value("video_texture", texture_data)
    dpg.render_dearpygui_frame()

# Cleanup
cap.release()
dpg.destroy_context()