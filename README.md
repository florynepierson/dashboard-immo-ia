# Tableau de bord Agence Immobilière + IA

Application data interactive pour une agence immobilière : pilotage des leads et des ventes,
plus **deux modèles de machine learning** —

- **Scoring des leads** (régression logistique) — quels prospects rappeler en priorité ?
- **Estimation du délai de vente** (régression linéaire sur la sur-évaluation) — aide à fixer le prix.
- **Prévision du chiffre d'affaires** (lissage exponentiel) + **recommandations automatiques**.

Construit en **Python** · scikit-learn · pandas · Plotly · **Streamlit**.
Données fictives réalistes ; modèles évalués sur jeu de test (scores honnêtes).

Par **Floryne Pierson**.

## Lancer en local
```bash
pip install -r requirements.txt
streamlit run app.py
```
