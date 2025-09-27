#!/usr/bin/env python3
"""
Test rapide de caméra - Prend une photo et l'envoie via SSH
Usage: python3 quick_test_camera.py [IP_PC] [USERNAME]
"""

import sys
import time
import subprocess
from datetime import datetime
from picamera2 import Picamera2
import os

def take_photo():
    """Prend une photo avec PiCamera2"""
    print("🎥 Test de la caméra...")
    
    try:
        # Initialiser la caméra
        picam2 = Picamera2()
        
        # Vérifier les caméras disponibles
        cameras = Picamera2.global_camera_info()
        print(f"📷 Caméras détectées: {len(cameras)}")
        
        if not cameras:
            print("❌ Aucune caméra trouvée!")
            return None
            
        for i, cam in enumerate(cameras):
            print(f"  Camera {i}: {cam}")
        
        # Configuration haute résolution pour le test
        config = picam2.create_still_configuration(
            main={"size": (1920, 1080)}
        )
        picam2.configure(config)
        
        print("✅ Configuration appliquée")
        
        # Démarrer la caméra
        picam2.start()
        print("✅ Caméra démarrée")
        
        # Attendre stabilisation
        print("⏳ Stabilisation (3 secondes)...")
        time.sleep(3)
        
        # Générer nom de fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"camera_test_{timestamp}.jpg"
        
        # Prendre la photo
        print(f"📸 Capture en cours...")
        picam2.capture_file(filename)
        
        # Arrêter la caméra
        picam2.stop()
        
        # Vérifier que le fichier existe
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"✅ Photo prise: {filename} ({file_size} bytes)")
            return filename
        else:
            print("❌ Erreur: fichier non créé")
            return None
            
    except Exception as e:
        print(f"❌ Erreur caméra: {e}")
        import traceback
        traceback.print_exc()
        return None

def send_photo_ssh(filename, pc_ip, username):
    """Envoie la photo via SCP"""
    if not filename or not os.path.exists(filename):
        print("❌ Pas de fichier à envoyer")
        return False
    
    try:
        print(f"📡 Envoi vers {username}@{pc_ip}...")
        
        # Commande SCP pour envoyer le fichier
        scp_command = [
            "scp", 
            filename, 
            f"{username}@{pc_ip}:~/Downloads/"
        ]
        
        # Exécuter la commande
        result = subprocess.run(scp_command, 
                              capture_output=True, 
                              text=True, 
                              timeout=30)
        
        if result.returncode == 0:
            print(f"✅ Photo envoyée dans ~/Downloads/ sur {pc_ip}")
            print(f"🖼️  Fichier: {filename}")
            return True
        else:
            print(f"❌ Erreur SCP: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout lors de l'envoi")
        return False
    except Exception as e:
        print(f"❌ Erreur envoi: {e}")
        return False

def send_photo_local_network(filename):
    """Alternative: copie via partage réseau local"""
    print("💡 Alternative: Vous pouvez aussi:")
    print(f"   1. Télécharger directement: scp pi@IP_PI:~/{filename} ~/Downloads/")
    print(f"   2. Ou utiliser FileZilla/WinSCP pour récupérer: {filename}")

def main():
    print("=" * 60)
    print("🎯 Test Rapide Caméra Raspberry Pi")
    print("=" * 60)
    
    # Prendre la photo
    photo_file = take_photo()
    
    if not photo_file:
        print("\n❌ Test échoué - Problème avec la caméra")
        return 1
    
    print(f"\n✅ Test caméra réussi!")
    print(f"📁 Fichier local: {photo_file}")
    
    # Vérifier les arguments pour l'envoi SSH
    if len(sys.argv) >= 3:
        pc_ip = sys.argv[1]
        username = sys.argv[2]
        
        print(f"\n🔄 Tentative d'envoi vers {username}@{pc_ip}...")
        if send_photo_ssh(photo_file, pc_ip, username):
            print("\n🎉 Test complet réussi!")
            
            # Nettoyer le fichier local après envoi réussi
            try:
                os.remove(photo_file)
                print(f"🗑️  Fichier local supprimé: {photo_file}")
            except:
                pass
        else:
            print(f"\n⚠️  Envoi échoué, mais photo disponible localement")
            send_photo_local_network(photo_file)
    else:
        print(f"\n💡 Pour envoyer automatiquement sur votre PC:")
        print(f"   python3 {sys.argv[0]} IP_DE_VOTRE_PC VOTRE_USERNAME")
        print(f"   Exemple: python3 {sys.argv[0]} 192.168.1.10 votrenom")
        send_photo_local_network(photo_file)
    
    return 0

if __name__ == "__main__":
    exit(main())