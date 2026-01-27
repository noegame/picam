## My functions

**Markers detection**
Find the markers in the input image and return their corners coordinates in the image and IDs.
```plantuml
@startuml
left to right direction
rectangle {
    [aruco detector] --> [detect_markers]
    [Img] --> [detect_markers]
    [detect_markers] --> [corners (matrice Nx4)]
    [detect_markers] --> [IDs (matrice Nx1)]
    [detect_markers] --> [rejected]
}
@enduml
```

**Coordinate conversion**
convert the coordinates of the image coordinate system to the terrain coordinate system
```plantuml
@startuml
left to right direction
rectangle {
    [corners (matrice Nx4)] --> [convert_coord]
    [convert_coord] --> [new_corners (matrice Nx4)]
}
@enduml
```

**Centers calculation**
Find the center coordinates of each detected marker from their corners coordinates
```plantuml
@startuml
left to right direction
rectangle {
    [corners (matrice Nx4)] --> [find_center_coord]
    [find_center_coord] --> [Centers (matrice Nx1)]
}
@enduml
```

**Detection statistics**
Print statistics about the detected markers
```plantuml
@startuml
left to right direction
rectangle {
    [IDs (matrice Nx1)] --> [print_detection_stats]
    [Centers (matrice Nx1)] --> [print_detection_stats]
    [print_detection_stats] --> [None]
}
@enduml
```