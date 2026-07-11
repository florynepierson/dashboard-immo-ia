# Dashboard Agence Immobilière + Prédictions — guide de construction (Power BI)

**Ta pièce portfolio n°1 (niche immo).** À la fin : un dashboard pro qui montre les leads, les ventes, les agents — **avec 3 briques prédictives** (prévision de CA, facteurs de conversion, scoring de leads / délai de vente). Compte ~1–2 h la 1ʳᵉ fois.

**Données fournies** (dans `donnees/`) :
- `biens.csv` — 260 mandats (type, ville, prix, source, statut, délai de vente, vues, contacts, commission)
- `leads.csv` — 620 prospects (source, budget, temps de réponse, relances, converti oui/non)

*(Données fictives mais réalistes — corrélées pour que les modèles apprennent quelque chose de vrai.)*

---

## Étape 1 — Importer
Power BI Desktop → **Accueil → Obtenir les données → Texte/CSV** → importe `biens.csv` puis `leads.csv`.
Dans **Transformer les données** (Power Query) : vérifie que `date_mandat`, `date_vente`, `date` sont en type **Date**, et `prix / commission / budget / delai_vente_jours` en **Nombre**. → **Fermer & appliquer**.

## Étape 2 — Table de dates (indispensable pour la prévision)
Onglet **Modélisation → Nouvelle table** :
```DAX
Calendrier = CALENDAR(DATE(2024,1,1), DATE(2025,12,31))
```
Puis relie `Calendrier[Date]` à `biens[date_vente]` (glisse l'un sur l'autre dans la vue **Modèle**). C'est ce qui débloque la prévision temporelle.

## Étape 3 — Les mesures (DAX) — copie-colle
Clic droit sur `biens` → **Nouvelle mesure**, une par une :
```DAX
Commissions = CALCULATE(SUM(biens[commission]), biens[statut] = "Vendu")
Biens vendus = CALCULATE(COUNTROWS(biens), biens[statut] = "Vendu")
Prix moyen = AVERAGE(biens[prix])
Délai moyen de vente = CALCULATE(AVERAGE(biens[delai_vente_jours]), biens[statut] = "Vendu")
```
Sur `leads` :
```DAX
Leads reçus = COUNTROWS(leads)
Leads gagnés = SUM(leads[converti])
Taux de conversion = DIVIDE([Leads gagnés], [Leads reçus])
Temps de réponse moyen (h) = AVERAGE(leads[temps_reponse_h])
```

## Étape 4 — Les visuels (dispo conseillée)
- **En haut : 5 cartes KPI** → `Commissions`, `Biens vendus`, `Taux de conversion`, `Délai moyen de vente`, `Temps de réponse moyen`.
- **Courbe** : Commissions par mois (axe = `Calendrier[Date]` en hiérarchie Mois).
- **Barres** : `Leads reçus` et `Leads gagnés` par `source` → montre quelle source convertit le mieux.
- **Barres** : `Commissions` par `agent`.
- **Carte/table** : biens par `ville` et `statut`.
- **Segments (filtres)** en haut : `ville`, `agent`, `type`.

👉 Design pro : fond clair, 1 couleur d'accent, titres courts. (Inspire-toi du fichier `../dashboard-commercial/index.html` que je t'ai fait.)

---

## Étape 5 — LES PRÉDICTIONS (le truc qui te démarque) ⭐

### 5a. Prévision du CA — **sans code** (fais ça en premier)
Sur ta **courbe des Commissions par mois** → volet **Visualisations → Analyses (la loupe) → Prévision → Ajouter**.
Règle : *Longueur de la prévision = 3 mois*, *Intervalle de confiance = 90 %*.
→ Power BI trace automatiquement le **CA prévu des 3 prochains mois** avec une zone d'incertitude. **C'est déjà de la prédiction, en 3 clics.**

### 5b. Qu'est-ce qui fait convertir un lead — **visuel IA, sans code**
Ajoute le visuel **« Facteurs clés d'influence »** (Key Influencers).
- *Analyser* : `converti`
- *Expliquer par* : `source`, `temps_reponse_h`, `nb_relances`, `budget`
→ Il te sort en clair : *« Quand le temps de réponse est < 2h, la conversion est 3× plus probable »*. **Insight en or pour le client** (« répondez plus vite »).

### 5c. Scoring de leads + délai de vente estimé — **Python (le niveau premium)**
Active Python : *Fichier → Options → Scripts Python* (pointe vers ton install + `pip install scikit-learn pandas matplotlib`).
Ajoute un **visuel Python**, glisse dans *Valeurs* : `temps_reponse_h`, `nb_relances`, `budget`, `source`, `converti`. Colle :
```python
# dataset = les colonnes glissées dans le visuel
import pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
df = dataset.drop_duplicates().copy()
df = pd.get_dummies(df, columns=['source'], drop_first=True)
feat = [c for c in df.columns if c!='converti']
model = LogisticRegression(max_iter=1000).fit(df[feat], df['converti'])
df['score'] = (model.predict_proba(df[feat])[:,1]*100).round()
# Poids des facteurs (ce qui pousse la conversion)
imp = pd.Series(model.coef_[0], index=feat).sort_values()
imp.plot(kind='barh', color=['#e35d6a' if v<0 else '#00b487' for v in imp])
plt.title("Ce qui fait (vert) ou empêche (rouge) la conversion d'un lead")
plt.tight_layout(); plt.show()
```
→ Tu obtiens un **modèle de scoring** (probabilité de conversion par lead) + un graphe des facteurs. Pour le **délai de vente estimé**, même principe avec `LinearRegression` sur `biens` (features : `prix`, `surface_m2`, `nb_contacts`, `type`, `ville` → cible : `delai_vente_jours` des vendus).

*(La version « client » propre : le score se calcule côté données et remonte comme une colonne — mais pour le portfolio, le visuel Python suffit à prouver que tu sais faire du ML.)*

---

## Étape 6 — Comment le présenter en portfolio (le plus important)
Une page (site ou PDF), structure **Problème → Solution → Impact** :
> **Problème** — Une agence immobilière suit ses leads et ses ventes dans des fichiers Excel épars : aucune vision de ce qui marche, pas de prévision.
> **Solution** — Un tableau de bord Power BI mis à jour automatiquement : leads par source, performance des annonces, ventes par agent, **+ prévision du CA à 3 mois et scoring des leads (ML)**.
> **Impact** — Fini le reporting manuel du lundi ; l'agence sait **où mettre son budget**, **quels leads rappeler en priorité**, et **anticipe son chiffre**.

**Étiquette-le honnêtement** : *« Projet de démonstration »* (ou le nom réel si l'agence de ton oncle te laisse l'utiliser).

---

### Ton angle de vente (à réutiliser en démarchage)
> *« Je fais pour les agences immobilières un tableau de bord qui montre d'où viennent les leads, quelles annonces marchent, et qui PRÉDIT le chiffre du trimestre + les leads les plus chauds à rappeler. »*

C'est ça, ta niche. Peu de gens la couvrent — surtout pas ta pote chez Rubis. 🐑
