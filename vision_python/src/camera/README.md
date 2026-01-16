# Module Camera

Ce module fournit une architecture orientée objet pour la gestion de différents types de caméras. Il utilise les principes de la programmation orientée objet (POO) avec une classe abstraite de base et plusieurs implémentations concrètes.

La classe `Camera` définit l'interface commune pour tous les types de caméras. Elle contient les méthodes abstraites suivantes :

#### Méthodes abstraites

- **`init()`** : Initialise le matériel de la caméra
- **`start()`** : Démarre le flux vidéo
- **`stop()`** : Arrête le flux et libère les ressources
- **`set_parameters(parameters: Dict[str, Any])`** : Configure les paramètres de la caméra
- **`capture_photo()`** : Capture une photo et la retourne sous forme de `np.ndarray`



```plantuml
@startuml Camera Architecture

' Configuration du style
skinparam classAttributeIconSize 0
skinparam shadowing false
skinparam backgroundColor white


abstract class Camera {
    ' Classe abstraite de base pour toutes les caméras
    {abstract} +init() : void
    {abstract} +start() : void
    {abstract} +stop() : void
    {abstract} +set_parameters(parameters: Dict[str, Any]) : void
    {abstract} +capture_photo() : np.ndarray
}

class PiCamera {
    ' Implémentation pour Raspberry Pi
}

class Webcam {
    ' Implémentation pour webcams USB
}

class EmulatedCamera {
    ' Implémentation pour simulation/test
}

' Relations d'héritage
Camera <|-- PiCamera
Camera <|-- Webcam
Camera <|-- EmulatedCamera

' Note pour la classe Camera
note right of Camera
    Classe abstraite définissant
    l'interface commune pour
    tous les types de caméras
end note

@enduml
```