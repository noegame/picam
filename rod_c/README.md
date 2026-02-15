# ROD - Remote Observation Device

## What is ROD

As part of the [Eurobot](https://www.eurobot.org/) competition and the [French Robotics Cup](https://www.coupederobotique.fr/), robots compete against each other in matches on a game table. The competition rules allow the use of a Remote Observation Device (ROD): a system equipped with a camera that captures images of the game table and transmits information to the robot.

The objectives of this ROD project are:
- Recognition of ArUco TAGs on game elements using a camera mounted above the table
- Ability to precisely locate game elements on the playing field
- Transmission of game element positions to a robot

## How it works

ROD is working on two threads that communicate with each other through socket communication. The first thread is responsible for the computer vision part, while the second thread is responsible for the IPC (Inter-Process Communication) part.

The computer vision thread send to the IPC thread via socket an array that contain the list of detected objects with their coordinates [[id, x,y,angle], [id, x,y,angle], ...]. The IPC thread then send this array to the main process of the robot via shared memory.

```plantuml
rectangle "File structure"{
    folder "rod" {
        file "rod_com.c"
        file "rod_detection.c"
        folder "rod_cv" {
            file "rod_cv.c"
            file "opencv-wrapper.cpp"
            file "opencv-wrapper.h"
        }
        folder "rod_camera" {
            file "rod_camera.c"
            file "libcamera-wrapper.cpp"
            file "libcamera-wrapper.h"
        }
        folder "rod_communication" {
            file "rod_communication.c"
        }
    }
}
```

```plantuml
rectangle "Component diagram"{

    component "com" as rod{
        file "rod_com.c"
    }

    component "detection" as detection{
        file "rod_detection.c"

    }

    rod <-- detection : "socket communication"
}
```

## How to use
- [How to build](docs/how-to-build.md)
- [How to run](docs/how-to-run.md)
