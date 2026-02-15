# How to Run PiCam on Raspberry Pi (python version)

## Running the Main Application
``` shell
source .venv/bin/activate
cd raspberry
python3 -m vision_python.src.main
```

## Capturing Images for Calibration
``` shell
cd raspberry
python3 -m tools.capture_for_calibration
```

## Calibrating the Camera
``` shell
cd raspberry
python3 -m tools.calibrate_camera
```

## Testing the Camera
``` shell
cd raspberry
python3 -m tools.test_camera
```
