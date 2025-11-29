
from picamera2 import Picamera2, Preview
import time
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)
picam2.capture_file("test.jpg")
still_config = picam2.create_still_configuration(main={"size": (2000, 2000)})
picam2.configure(still_config)
time.sleep(0.1)
picam2.capture_file("test_2000.jpg")
