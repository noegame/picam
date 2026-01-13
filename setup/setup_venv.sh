#!/bin/bash

# Script pour créer l'environnement virtuel Python et installer les dépendances
# Usage: ./setup_venv.sh

set -e  # Arrêter le script en cas d'erreur

echo "🔧 Configuration de l'environnement virtuel Python..."

# Vérifier si Python3 est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Erreur: Python3 n'est pas installé"
    echo "Installez Python3 d'abord: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

# Afficher la version de Python
PYTHON_VERSION=$(python3 --version)
echo "✅ $PYTHON_VERSION détecté"

# Créer l'environnement virtuel s'il n'existe pas déjà
if [ -d ".venv" ]; then
    echo "⚠️  Le dossier .venv existe déjà"
    read -p "Voulez-vous le supprimer et le recréer? (o/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        echo "🗑️  Suppression de l'ancien .venv..."
        rm -rf .venv
    else
        echo "ℹ️  Conservation du .venv existant"
    fi
fi

if [ ! -d ".venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv .venv
    echo "✅ Environnement virtuel créé"
else
    echo "ℹ️  Utilisation du .venv existant"
fi

# Activer l'environnement virtuel
echo "🔌 Activation de l'environnement virtuel..."
source .venv/bin/activate

# Mettre à jour pip
echo "⬆️  Mise à jour de pip..."
pip install --upgrade pip

# Installer les dépendances depuis requirements.txt
echo "📥 Installation des dépendances..."
if [ -f "vision_python/requirements.txt" ]; then
    pip install -r vision_python/requirements.txt
    echo "✅ Dépendances installées avec succès"
else
    echo "❌ Erreur: fichier vision_python/requirements.txt non trouvé"
    exit 1
fi

echo ""
echo "✨ Configuration terminée!"
echo ""
echo "Pour activer l'environnement virtuel:"
echo "  source .venv/bin/activate"
echo ""
echo "Pour désactiver l'environnement virtuel:"
echo "  deactivate"
