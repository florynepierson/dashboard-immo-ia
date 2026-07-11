# Guide ML — quel modèle choisir, comment le régler (pour tes futurs projets)

> Ta fiche de référence. À chaque nouveau projet : quelle question ? quelles données ? → quel modèle → quels réglages → comment mesurer.

---

## 1. LA DÉMARCHE (toujours la même, dans l'ordre)
1. **Cadrer la question** : on prédit **quoi** ? (un oui/non ? un nombre ? une évolution dans le temps ? des groupes ?)
2. **Regarder les données** : combien de lignes d'historique ? quelles colonnes ? propres ? → *ça décide de ce qui est possible.*
3. **Nettoyer + feature engineering** : corriger les formats, créer des variables utiles (ex. « prix au m² », « sur-évaluation »), **retirer les fuites** (variables non dispo au moment de prédire).
4. **Séparer train / test** (75 / 25) : on entraîne sur une partie, on **note sur l'autre** (jamais vue) → scores honnêtes.
5. **Choisir + entraîner un modèle** (voir §2-3).
6. **Évaluer + vérifier le COMPORTEMENT** : bons scores ? ET logique ? (monotone, sensible aux bons paramètres). Si illogique → changer de modèle.

---

## 2. QUEL MODÈLE POUR QUELLE QUESTION

| La question | Type de problème | Modèles candidats |
|---|---|---|
| « Est-ce que OUI ou NON ? » (converti, va churner, fraude…) | **Classification** | Régression **logistique**, Random Forest, Gradient Boosting |
| « Combien ? » (prix, délai, nb de ventes…) | **Régression** | Régression **linéaire**, Random Forest, Gradient Boosting |
| « Comment ça évolue dans le temps ? » (CA du trimestre) | **Série temporelle** | Tendance, **lissage exponentiel (Holt/Holt-Winters)**, ARIMA, Prophet |
| « Quels groupes naturels ? » (segments clients) | **Clustering** | **K-means**, clustering hiérarchique |

---

## 3. LES MODÈLES PRINCIPAUX (quand / forces / faiblesses / réglages)

### Régression logistique (classification)
- **Quand** : oui/non, quand tu veux de la **lisibilité** et un comportement **monotone** (plus de X = toujours plus de proba).
- **Forces** : simple, rapide, interprétable (chaque variable a un « poids »), monotone.
- **Faiblesses** : ne capte pas les relations **non-linéaires** ni les interactions complexes.
- **Réglages clés** : `C` (force de régularisation, plus petit = plus « lissé »), et **normaliser les variables** (`StandardScaler`) si elles ont des échelles très différentes (ex. budget en 100 000 vs relances 0-4).

### Régression linéaire (régression)
- **Quand** : prédire un nombre, relation ~ proportionnelle, besoin de lisibilité/monotonie.
- **Forces** : simple, interprétable, monotone, très bon **si tu lui donnes les bonnes variables** (feature engineering).
- **Faiblesses** : ne capte pas les non-linéarités/interactions → si R² mauvais, souvent = variable manquante (ex. il fallait « sur-évaluation », pas juste « prix »).
- **Réglages** : peu. Les variantes **Ridge / Lasso** ajoutent une régularisation (`alpha`) contre le sur-apprentissage.

### Random Forest (classif ou régression)
- **Quand** : relations **non-linéaires**, interactions, sans se prendre la tête. Bon « par défaut » performant.
- **Forces** : robuste, capte les interactions, peu de préparation.
- **Faiblesses** : **« par morceaux »** (prédictions en escalier → peut être **non-monotone / peu sensible** à une variable). Moins lisible. **N'extrapole pas** hors des valeurs vues.
- **Réglages clés** : `n_estimators` (nb d'arbres, 100-500), `max_depth` (profondeur, limite le sur-apprentissage), `min_samples_leaf`.

### Gradient Boosting (classif ou régression)
- **Quand** : chercher la **performance max** sur données tabulaires (souvent le meilleur score brut).
- **Forces** : très performant.
- **Faiblesses** : **non-monotone** (peut donner des trucs illogiques pour un utilisateur), plus lent, à régler soigneusement. *(Dans ce projet, on l'a écarté pour les leads car « 2 relances < 1 relance » = inacceptable côté client.)*
- **Réglages clés** : `n_estimators`, `learning_rate` (petit = plus fin mais plus lent), `max_depth`. Versions modernes : **XGBoost, LightGBM**.

### Lissage exponentiel / Holt-Winters (série temporelle)
- **Quand** : prévoir une **suite dans le temps** (ventes par mois).
- **Forces** : pondère les mois récents ; Holt-**Winters** gère la **saisonnalité** (besoin de 2+ cycles = 2 ans pour du mensuel).
- **Réglages** : `trend` (add/mul), `seasonal`, `seasonal_periods`.

---

## 4. LE TUNING (régler les hyperparamètres)
- **Hyperparamètres** = les « boutons » du modèle (ex. `C`, `n_estimators`, `max_depth`). On ne les apprend pas, on les **choisit/teste**.
- **Comment tester proprement** : **validation croisée** (`cross_val_score`) + **grille de recherche** (`GridSearchCV`) → il essaie plusieurs combinaisons et garde la meilleure **sur données de validation**.
- **Sur-apprentissage (overfitting)** : le modèle « récite » l'entraînement mais rate le test. **Signe** : score train excellent, score test mauvais. **Remèdes** : plus de régularisation, arbres moins profonds, plus de données, moins de variables.
- **Sous-apprentissage** : mauvais partout → modèle trop simple ou variable clé manquante (feature engineering !).

---

## 5. LES CRITÈRES DE CHOIX (le vrai jugement)
1. **Interprétabilité vs performance** : un client/recruteur préfère souvent un modèle **compréhensible** (logistique/linéaire) à une boîte noire 2 % meilleure.
2. **Monotonie / logique** : si l'outil est utilisé par un humain qui décide, il DOIT se comporter logiquement (→ logistique/linéaire).
3. **Taille des données** : peu de données → modèle simple (les complexes sur-apprennent).
4. **Linéaire ou pas** : relation ~ proportionnelle → linéaire ; relations tordues/interactions → forêt/boosting.
5. **Besoin d'extrapoler** : hors des valeurs connues → éviter Random Forest (il plafonne).

---

## 6. LES LEÇONS DE CE PROJET (concret)
- **Leads : logistique et pas Gradient Boosting** → parce qu'il faut du **monotone/lisible** (un client ne doit jamais voir « plus je relance, moins ça marche »). On a sacrifié ~1 % d'AUC pour de la **logique** : bon deal.
- **Délai : régression linéaire + feature « sur-évaluation »** → le prix seul ne suffisait pas (R² négatif !) ; créer *prix ÷ prix médian de la ville* a tout changé (R² 0.94). **La feature intelligente > le modèle compliqué.**
- **Train/test obligatoire** → sinon scores gonflés, indéfendables.
- **Retirer les fuites** (`nb_contacts` arrive après la vente) → sinon le modèle « triche » avec le futur.
- **Normaliser** (scaler) quand les échelles diffèrent (budget vs relances).
- **Toujours vérifier le comportement**, pas juste le score : c'est ce que TU as fait en repérant les incohérences. 👏

→ *Retiens ça : 80 % de la qualité vient du **cadrage + des bonnes variables + la vérification du comportement**, pas de l'algo à la mode.*
