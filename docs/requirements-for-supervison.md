# Specifications for improvement of supervision extension

This involves adding a camera tab to the VSC "robot eseo" extension toolbox.
The implementation of this new feature should reuse as much as possible the existing code of the supervision extension.

## General

The tab offers 2 displays: "camera" or "field". 
The displays can be hidden, shown independently, or simultaneously.

The tab displays the camera status: "connected" or "disconnected"
The tab displays the transmission mode: "image" and/or "coordinates"
The tab displays the latency (only possible if the message contains the sending time)

The tab displays a "restart" button
The tab displays a "reconnect" button


### Camera Display

The tab displays what the camera sees.
The tab displays the number of frames per second.

### Field Display

The tab receives the positions of game elements. 
The tab displays the game elements on the field.

## Note

- box game element: 150 mm x 50 mm

