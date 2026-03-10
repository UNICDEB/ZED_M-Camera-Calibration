import pyzed.sl as sl
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("E:/Project_Work/2025/Saffron_Project/Github_Code/Saffron_Detection/YoloV8/yolo11x.pt")   # change to your model

# ZED camera setup
zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER

status = zed.open(init_params)

if status != sl.ERROR_CODE.SUCCESS:
    print("Camera Open Error")
    exit()

runtime = sl.RuntimeParameters()

image = sl.Mat()
point_cloud = sl.Mat()

prev_time = time.time()

print("Starting detection...")

while True:

    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        frame = np.ascontiguousarray(image.get_data()[:, :, :3])

        # YOLO detection
        results = model(frame, verbose=False)

        for result in results:

            boxes = result.boxes

            for box in boxes:

                conf = float(box.conf[0])

                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                err, point = point_cloud.get_value(cx, cy)

                if err == sl.ERROR_CODE.SUCCESS:

                    X = point[0]
                    Y = point[1]
                    Z = point[2]

                    if np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z):

                        distance = np.sqrt(X**2 + Y**2 + Z**2)

                        label = f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f} D:{distance:.2f}m"

                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                        cv2.circle(frame,(cx,cy),5,(0,0,255),-1)

                        cv2.putText(frame,
                                    label,
                                    (x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0,255,0),
                                    2)

                        print(label)

        # FPS
        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time = curr_time

        cv2.putText(frame,
                    f"FPS: {int(fps)}",
                    (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2)

        cv2.imshow("ZED YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

zed.close()
cv2.destroyAllWindows()