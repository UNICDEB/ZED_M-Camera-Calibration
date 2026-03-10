# import pyzed.sl as sl
# import cv2
# import numpy as np
# import time
# import math

# click_x = 0
# click_y = 0

# def mouse_callback(event, x, y, flags, param):
#     global click_x, click_y
#     if event == cv2.EVENT_LBUTTONDOWN:
#         click_x = x
#         click_y = y


# def main():

#     global click_x, click_y

#     zed = sl.Camera()

#     init_params = sl.InitParameters()
#     init_params.camera_resolution = sl.RESOLUTION.HD720
#     init_params.camera_fps = 30
#     init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
#     init_params.coordinate_units = sl.UNIT.METER

#     if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
#         print("Failed to open ZED camera")
#         exit()

#     runtime = sl.RuntimeParameters()

#     image = sl.Mat()
#     depth = sl.Mat()
#     point_cloud = sl.Mat()

#     cv2.namedWindow("RGB")
#     cv2.setMouseCallback("RGB", mouse_callback)

#     prev_time = time.time()

#     while True:

#         if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

#             # Retrieve data
#             zed.retrieve_image(image, sl.VIEW.LEFT)
#             zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
#             zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

#             frame = image.get_data()
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

#             # Get clicked point 3D
#             err, point = point_cloud.get_value(click_x, click_y)

#             if err == sl.ERROR_CODE.SUCCESS:

#                 X = point[0]
#                 Y = point[1]
#                 Z = point[2]

#                 if np.isfinite(Z):

#                     distance = math.sqrt(X**2 + Y**2 + Z**2)

#                     text = f"X:{X:.2f}  Y:{Y:.2f}  Z:{Z:.2f}  Dist:{distance:.2f}m"

#                     cv2.putText(frame, text,
#                                 (20,40),
#                                 cv2.FONT_HERSHEY_SIMPLEX,
#                                 0.7,(0,255,0),2)

#                     cv2.circle(frame,(click_x,click_y),5,(0,0,255),-1)

#             # FPS
#             curr_time = time.time()
#             fps = 1/(curr_time-prev_time)
#             prev_time = curr_time

#             cv2.putText(frame,
#                         f"FPS: {int(fps)}",
#                         (20,70),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.7,(255,0,0),2)

#             # Depth visualization
#             depth_map = depth.get_data()

#             depth_map = np.nan_to_num(depth_map)

#             depth_map = cv2.normalize(depth_map,None,0,255,cv2.NORM_MINMAX)

#             depth_map = depth_map.astype(np.uint8)

#             depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

#             # Display
#             cv2.imshow("RGB", frame)
#             cv2.imshow("Depth", depth_map)

#         key = cv2.waitKey(1)

#         if key == 27:
#             break

#     zed.close()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()

####################

import pyzed.sl as sl
import cv2
import numpy as np
import time

click_x = 0
click_y = 0
current_view = "left"


def mouse_callback(event, x, y, flags, param):
    global click_x, click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x = x
        click_y = y


def get_stable_depth(depth, x, y):

    values = []

    width = depth.get_width()
    height = depth.get_height()

    for i in range(-3, 4):
        for j in range(-3, 4):

            px = x + i
            py = y + j

            if px < 0 or py < 0 or px >= width or py >= height:
                continue

            err, val = depth.get_value(px, py)

            if err == sl.ERROR_CODE.SUCCESS and np.isfinite(val):
                values.append(val)

    if len(values) > 0:
        return np.median(values)
    else:
        return None


def main():

    global current_view

    zed = sl.Camera()

    init_params = sl.InitParameters()
    # init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Camera open failed")
        exit()

    runtime = sl.RuntimeParameters()

    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()

    cv2.namedWindow("ZED View")
    cv2.setMouseCallback("ZED View", mouse_callback)

    prev_time = time.time()

    while True:

        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

            if current_view == "left":
                zed.retrieve_image(image, sl.VIEW.LEFT)
            else:
                zed.retrieve_image(image, sl.VIEW.RIGHT)

            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # FIX: ensure OpenCV compatible image
            frame = np.ascontiguousarray(image.get_data()[:, :, :3])

            depth_value = get_stable_depth(depth, click_x, click_y)

            err, point = point_cloud.get_value(click_x, click_y)

            if err == sl.ERROR_CODE.SUCCESS and depth_value is not None:

                X = point[0]
                Y = point[1]
                Z = depth_value

                text = f"X:{X:.2f}m  Y:{Y:.2f}m  Z:{Z:.2f}m"

                cv2.putText(frame,
                            text,
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2)

                cv2.circle(frame, (click_x, click_y), 5, (0, 0, 255), -1)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            cv2.putText(frame,
                        f"FPS: {int(fps)}",
                        (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2)

            cv2.putText(frame,
                        f"View: {current_view.upper()}",
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2)

            depth_map = depth.get_data()

            depth_map = np.nan_to_num(depth_map)

            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

            depth_map = depth_map.astype(np.uint8)

            depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

            cv2.imshow("ZED View", frame)
            cv2.imshow("Depth Map", depth_map)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == ord('l'):
            current_view = "left"

        elif key == ord('r'):
            current_view = "right"

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()