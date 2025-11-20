from picamera2 import Picamera2

picam2 = Picamera2()

config = picam2.create_still_configuration(
    main={"size": (2200, 2200)}
)
picam2.configure(config)

picam2.start()
picam2.capture_file("photo_2200.jpg")
picam2.stop()
















