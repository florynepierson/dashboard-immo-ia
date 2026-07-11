# -*- coding: utf-8 -*-
"""
ml_explique.py — LE MACHINE LEARNING DE L'APPLI, À ÉTUDIER
============================================================
Fichier pédagogique : chaque étape est commentée (le QUOI et le POURQUOI).
Pas d'interface ici — juste le ML, pour comprendre.

Pour le lancer :   python3 ml_explique.py
(depuis le dossier, avec l'environnement activé : source venv/bin/activate)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score

# ─────────────────────────────────────────────────────────────
# ÉTAPE 1 — CHARGER LES DONNÉES
# On lit les 2 fichiers CSV dans des "DataFrame" (des tableaux pandas).
# ─────────────────────────────────────────────────────────────
leads = pd.read_csv("donnees/leads.csv")
biens = pd.read_csv("donnees/biens.csv")
# On force les colonnes chiffrées à être des NOMBRES (parfois lues comme du texte).
for c in ["prix", "surface_m2", "delai_vente_jours"]:
    biens[c] = pd.to_numeric(biens[c], errors="coerce")

print("Leads :", len(leads), "lignes | Biens :", len(biens), "lignes\n")


# ═════════════════════════════════════════════════════════════
# MODÈLE 1 — SCORING DE LEADS  (CLASSIFICATION : oui / non)
# Question : ce prospect va-t-il devenir client (converti = 1) ou non (0) ?
# ═════════════════════════════════════════════════════════════
print("="*55, "\nMODÈLE 1 · Scoring de leads (régression logistique)\n", "="*55)

# --- 1a) Choisir les variables (X = les causes) et la cible (y = le résultat) ---
# On NE met PAS de variable "du futur" (fuite de données).
features_lead = ["temps_reponse_h", "nb_relances", "budget", "source"]
X = leads[features_lead]
y = leads["converti"]                       # 0 ou 1

# --- 1b) Traduire le texte en chiffres ---
# Un modèle ne comprend pas le mot "SeLoger". get_dummies crée une colonne 0/1 par source.
X = pd.get_dummies(X, columns=["source"], drop_first=True)

# --- 1c) Séparer entraînement / test ---
# On entraîne sur 75 %, on NOTE sur les 25 % restants (jamais vus) → score honnête.
# stratify=y garde la même proportion de oui/non dans les 2 paquets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# --- 1d) Le modèle : régression logistique, avec normalisation ---
# StandardScaler met toutes les variables à la même échelle (le budget en 100 000
# ne doit pas écraser les relances en 0-4). make_pipeline enchaîne scaler puis modèle.
# On choisit la logistique car elle est MONOTONE : plus de relances = toujours plus de proba.
modele_lead = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
modele_lead.fit(X_train, y_train)           # ← il APPREND ici

# --- 1e) Évaluer sur le jeu de test ---
pred = modele_lead.predict(X_test)                          # oui/non prédits
proba = modele_lead.predict_proba(X_test)[:, 1]             # probabilité d'être "1"
print(f"Accuracy (test) : {accuracy_score(y_test, pred):.2f}   "
      f"AUC (test) : {roc_auc_score(y_test, proba):.2f}")
# Bonus : validation croisée (5 découpages) → une AUC plus robuste
auc_cv = cross_val_score(modele_lead, X, y, cv=5, scoring="roc_auc")
print(f"AUC en validation croisée (5 plis) : {auc_cv.mean():.2f} ± {auc_cv.std():.2f}")

# --- 1f) Prédire pour un NOUVEAU prospect ---
def scorer_lead(source, budget, temps_reponse_h, nb_relances):
    ligne = pd.DataFrame([{"temps_reponse_h": temps_reponse_h, "nb_relances": nb_relances,
                           "budget": budget, "source": source}])
    ligne = pd.get_dummies(ligne, columns=["source"])
    ligne = ligne.reindex(columns=X.columns, fill_value=0)  # mêmes colonnes que l'entraînement
    return modele_lead.predict_proba(ligne)[0, 1]

print("Exemple — bon prospect (Référence, 0.5h, 4 relances) :",
      f"{scorer_lead('Référence', 300000, 0.5, 4)*100:.0f}%")
print("Exemple — mauvais (LeBonCoin, 72h, 0 relance)       :",
      f"{scorer_lead('LeBonCoin', 120000, 72, 0)*100:.0f}%\n")


# ═════════════════════════════════════════════════════════════
# MODÈLE 2 — DÉLAI DE VENTE  (RÉGRESSION : un nombre de jours)
# Question : combien de jours ce bien mettra-t-il à se vendre ?
# ═════════════════════════════════════════════════════════════
print("="*55, "\nMODÈLE 2 · Délai de vente (régression linéaire)\n", "="*55)

# On n'apprend que sur les biens DÉJÀ vendus (on connaît leur délai réel).
vendus = biens[biens["statut"] == "Vendu"].dropna(subset=["delai_vente_jours"]).copy()

# --- 2a) FEATURE ENGINEERING : la variable la plus importante ---
# Le prix seul ne suffit pas. Ce qui compte, c'est le prix par rapport au MARCHÉ.
# prix_m2 = prix par m². sur_eval = prix_m2 / prix médian au m² de la ville.
#   > 1  = surévalué (se vend lentement)   |   < 1 = bien placé (se vend vite)
vendus["prix_m2"] = vendus["prix"] / vendus["surface_m2"]
ref_ville = vendus.groupby("ville")["prix_m2"].median()      # prix médian au m² par ville
vendus["sur_eval"] = vendus["prix_m2"] / vendus["ville"].map(ref_ville)

features_delai = ["sur_eval", "surface_m2", "type", "ville"]
Xd = pd.get_dummies(vendus[features_delai], columns=["type", "ville"], drop_first=True)
yd = vendus["delai_vente_jours"]

# --- 2b) Séparer + entraîner ---
Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xd, yd, test_size=0.25, random_state=42)
modele_delai = LinearRegression().fit(Xd_train, yd_train)   # linéaire = monotone & lisible

# --- 2c) Évaluer ---
pred_d = modele_delai.predict(Xd_test)
print(f"Erreur moyenne MAE (test) : ± {mean_absolute_error(yd_test, pred_d):.0f} jours   "
      f"R² (test) : {r2_score(yd_test, pred_d):.2f}")

# --- 2d) Prédire pour un NOUVEAU bien ---
def estimer_delai(prix, surface, type_bien, ville):
    se = (prix / surface) / ref_ville.get(ville, ref_ville.median())
    ligne = pd.DataFrame([{"sur_eval": se, "surface_m2": surface, "type": type_bien, "ville": ville}])
    ligne = pd.get_dummies(ligne, columns=["type", "ville"])
    ligne = ligne.reindex(columns=Xd.columns, fill_value=0)
    return max(1, int(modele_delai.predict(ligne)[0]))

print("Exemple — Appart 60m² Metz bien placé (~156k) :", estimer_delai(156000, 60, "Appartement", "Metz"), "jours")
print("Exemple — le même surévalué (260k)            :", estimer_delai(260000, 60, "Appartement", "Metz"), "jours")

print("\n✅ Fini. Relis les commentaires : c'est ÇA, un pipeline de machine learning.")
