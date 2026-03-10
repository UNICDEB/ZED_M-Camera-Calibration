import pyzed.sl as sl
import cv2
import numpy as np

def main():

    # Create ZED Camera object
    zed = sl.Camera()

    # Camera initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    # Open the camera
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open Failed:", status)
        exit(1)

    print("ZED-M Camera Opened Successfully")

    # Create image container
    image = sl.Mat()

    runtime_parameters = sl.RuntimeParameters()

    while True:

        # Grab frame
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)

            # Convert to numpy
            frame = image.get_data()

            # Convert RGBA to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            # Show live feed
            cv2.imshow("ZED-M Live View", frame)

        key = cv2.waitKey(1)

        if key == 27:  # Press ESC to exit
            break

    # Release resources
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()