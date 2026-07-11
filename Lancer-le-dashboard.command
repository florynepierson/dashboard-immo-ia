#!/bin/bash
cd "$(dirname "$0")"
echo "════════════════════════════════════════"
echo "  🏡  Dashboard immo — démarrage"
echo "════════════════════════════════════════"
if [ ! -d venv ]; then echo "· Création de l'environnement (1 seule fois)…"; python3 -m venv venv; fi
source venv/bin/activate
echo "· Vérification des librairies (1re fois : 1-2 min, laisse tourner)…"
pip install -r requirements.txt
echo ""
echo "✅  Prêt ! L'appli va s'ouvrir dans ton navigateur."
echo "    Si rien ne s'ouvre tout seul, ouvre toi-même :  http://localhost:8501"
echo "    (Pour arrêter l'appli plus tard : ferme cette fenêtre Terminal.)"
echo ""
streamlit run app.py
