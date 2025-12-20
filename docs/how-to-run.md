# How to Run PiCam on Raspberry Pi

``` shell
source /home/roboteseo/dev/picam/.venv/bin/activate
``` 

## Running the Main Application
``` shell
cd /home/roboteseo/dev/picam
python3 -m raspberry.src.main
```

## Capturing Images for Calibration
``` shell
cd /home/roboteseo/dev/picam
python3 -m raspberry.tools.capture_for_calibration
```

## Calibrating the Camera
``` shell
cd /home/roboteseo/dev/picam
python3 -m raspberry.tools.calibrate_camera
```

## Testing the Camera
``` shell
cd /home/roboteseo/dev/picam
python3 -m raspberry.tools.test_camera
```
