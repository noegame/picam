# Script pour lister les fichiers d'un dossier et les enregistrer dans un fichier CSV
# Le script exclut lui-même et les fichiers .py
# Le fichier CSV contiendra une seule colonne avec les noms des fichiers

import os
import csv
import tkinter as tk
from tkinter import filedialog

# Nom du script lui-même pour ne pas l'inclure
current_script = os.path.basename(__file__)

# Choix du dossier à analyser
root = tk.Tk()
root.withdraw()  # Hide the main window

input_folder = filedialog.askdirectory(title="Select folder to analyze",
                                 initialdir=os.path.dirname(os.path.abspath(__file__)))

output_folder = filedialog.askdirectory(title="Select output folder",
                                 initialdir=os.path.dirname(os.path.abspath(__file__)))

# Nom du fichier CSV de sortie
output_csv = input_folder + ".csv"

# Récupération de la liste des fichiers du dossier
files = [
    f for f in os.listdir(input_folder)
    if os.path.isfile(os.path.join(input_folder, f)) and f != current_script and not f.endswith(".py")
]

# Écriture dans le fichier CSV
output_csv_path = os.path.join(output_folder, os.path.basename(input_folder) + ".csv")
with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["File Name"])  # en-tête
    for f in files:
        writer.writerow([f])

print(f"{len(files)} fichiers ajoutés dans {output_csv}")
