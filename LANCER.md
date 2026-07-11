# Lancer l'appli sur ton Mac (méthode propre)

Ouvre le Terminal et copie-colle bloc par bloc :

## 1. Aller dans le dossier
cd ~/Desktop/freelance-portfolio/dashboard-immo-predictif

## 2. Créer un environnement virtuel (une seule fois)
python3 -m venv venv
source venv/bin/activate

## 3. Installer les libs (une seule fois)
pip install -r requirements.txt

## 4. Lancer l'appli
streamlit run app.py

→ Ça ouvre l'appli dans ton navigateur (http://localhost:8501). Interactif, ML en direct 🎉
(Pour relancer plus tard : refais l'étape 1, puis `source venv/bin/activate`, puis l'étape 4.)

---

## Si "command not found: python3"
Installe Python : va sur https://www.python.org/downloads/ → télécharge la dernière version Mac → installe. Rouvre le Terminal.

## Mettre en ligne = lien partageable pour ton portfolio (gratuit)
1. Mets ces fichiers sur un repo GitHub (app.py, requirements.txt, dossier donnees/).
2. share.streamlit.io → "New app" → choisis ton repo → Deploy.
3. Tu obtiens un LIEN PUBLIC → à mettre sur florynepierson.com et dans tes messages de démarchage.
   (« Voici une démo interactive que j'ai construite : [lien] »)
