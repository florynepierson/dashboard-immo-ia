# Comprendre ton appli de A à Z (pour savoir la présenter)

> Objectif : que tu puisses **expliquer chaque partie** à un client ou un recruteur, sans stress. Lis-la une fois tranquillement.

---

## 1. C'est quoi Streamlit ?
Un outil Python qui transforme un **script** en **site web interactif** (boutons, curseurs, graphiques), **sans coder le web**. Tu écris du Python normal + des `st.quelquechose(...)` → Streamlit fabrique la page. C'est pour ça que ton fichier s'appelle `app.py` : c'est **l'appli**.

## 2. D'où viennent les données ? (ce qui ENTRE)
Deux fichiers **fictifs** que j'ai fabriqués (dossier `donnees/`) :
- **`biens.csv`** — 260 biens : type, ville, surface, prix, agent, source de l'annonce, statut (vendu / en cours…), délai de vente, vues, contacts, commission.
- **`leads.csv`** — 620 prospects : source, budget, temps de réponse, nb de relances, et surtout **`converti`** (1 = est devenu client, 0 = non).

👉 Pour un **vrai client**, tu remplaces juste ces 2 fichiers par SES données (export de son logiciel / CRM / Excel). Le reste de l'appli marche pareil.

## 3. Comment l'appli est construite (partie par partie)

**a) Charger les données** — `pd.read_csv(...)` lit les fichiers dans des **tableaux** (des « DataFrame » pandas). On convertit les dates et les nombres au bon format. Le `@st.cache_data` = « garde ça en mémoire pour ne pas relire à chaque clic » (ça va plus vite).

**b) Les filtres (barre de gauche)** — `st.sidebar.multiselect(...)` crée les menus Ville/Agent. Quand tu choisis, on **filtre** les tableaux (`biens[biens["ville"].isin(...)]`) → tous les chiffres se recalculent tout seuls.

**c) Les KPI (cartes du haut)** — `st.metric("Commissions", ...)`. Chaque chiffre est un calcul simple sur le tableau filtré : la **somme** des commissions des biens vendus, la **moyenne** du délai, le **taux** de conversion (leads gagnés ÷ leads reçus), etc.

**d) Les graphiques** — faits avec **Plotly** (`px.bar`, `go.Scatter`, `px.pie`). On **groupe** les données (`.groupby("source")`) puis on trace. Ex : commissions par mois, leads par source, portefeuille par statut.

---

## 4. LES PRÉDICTIONS — le cœur (explique ça bien 💡)

Le principe du **machine learning** : on montre au modèle **des exemples passés**, il **apprend les règles tout seul**, puis il **prédit** sur du nouveau. Trois briques :

### 🔮 A. Scoring de leads — « qui va convertir ? » (régression logistique)
- **Ce que ça fait** : une **probabilité (%)** qu'un prospect devienne client.
- **Variables** : source, temps de réponse, nb de relances, budget.
- **Le modèle** : **régression logistique** (+ normalisation des variables). Choisi car **monotone et lisible** : plus de relances = **toujours** plus de chances, réponse plus rapide = **toujours** plus. *(Un modèle en escalier comme le Gradient Boosting donnait des trucs illogiques — « 2 relances < 1 relance » — inacceptable pour un client.)*
- **Fiabilité** : AUC ~0.76 sur données de test. Le **classement** est bon (les meilleurs prospects remontent) ; le % reflète la vraie incertitude.

### ⏱️ B. Délai de vente estimé (régression linéaire sur la sur-évaluation)
- **Ce que ça fait** : estime un **nombre de jours** avant la vente.
- **Feature clé** : la **sur-évaluation** = prix au m² du bien ÷ **prix médian au m² de sa ville** (calculé sur l'historique des ventes). >1 = trop cher, <1 = bien placé. C'est LE moteur du délai.
- **Le modèle** : **régression linéaire** (monotone, lisible). R² ~0.94, erreur **± 6 jours** sur données de test.
- **Limite honnête** : le délai réagit surtout au **prix** ; le **type** de bien est peu séparable à surface fixe (un studio EST petit, une maison EST grande → type et surface se confondent dans les données). Et surtout ⬇️

### 📈 C. Prévision du CA — (lissage exponentiel)
On additionne les commissions **par mois**, puis on prévoit 3 mois avec un **lissage exponentiel (méthode de Holt)** — il **pondère plus les mois récents** que les anciens (plus fin qu'une simple droite). *(Avec 2+ ans d'historique, on ajouterait la **saisonnalité** — Holt-Winters. On le dit honnêtement au client.)*

### ✅ Pourquoi les scores sont HONNÊTES (train / test)
Un piège classique : entraîner ET noter le modèle sur les **mêmes** données → les scores sont **gonflés** (le modèle « récite » ce qu'il a vu). Ici on **sépare** : on entraîne sur 75 % des données, et on **mesure la fiabilité sur les 25 % restants (jamais vus)**. Les chiffres affichés (« 74 % de bonnes prédictions », « ± 18 jours d'erreur ») sont donc **défendables devant un data scientist**.
- Lead scoring → on regarde l'**accuracy** (% de bonnes prédictions) + l'**AUC** sur le jeu de test.
- Délai de vente → on regarde l'**erreur moyenne en jours** (MAE) + le **R²**.

### 🔎 Choisir les bonnes variables (feature engineering)
On a **retiré `nb_contacts`** du modèle de délai : les contacts arrivent **après** la mise en vente, donc on ne les connaît pas au moment de **fixer le prix** (utiliser une info du futur = « fuite de données », ça fausse tout). → C'est exactement pourquoi on **regarde les données du client AVANT** de choisir : chaque variable doit être **disponible au moment de la prédiction**.

### ⚙️ Un détail utile : `get_dummies`
Un modèle ne comprend que des **chiffres**, pas le mot « SeLoger ». `pd.get_dummies` transforme une colonne texte (source) en **colonnes 0/1** (une par source). C'est juste une traduction pour le modèle.

### Pourquoi tes prédictions marchent VRAIMENT
Parce que j'ai fabriqué les données fictives avec de **vraies corrélations dedans** : répondre vite augmente la conversion, un bien sur-évalué met plus longtemps à se vendre. Donc le modèle **trouve de vrais patterns** — ce n'est pas du hasard.

---

## 5. Comment le PRÉSENTER (ton pitch + réponses aux questions)

**Le pitch (30 sec)** :
> « J'ai construit une appli data pour une agence immobilière : un tableau de bord interactif de leurs leads et ventes, **plus deux modèles de machine learning** — un qui score les prospects (qui rappeler en priorité) et un qui estime le délai de vente d'un bien. Le tout en Python, déployable en ligne. »

**S'ils demandent…**
- *« C'est de vraies données ? »* → « Non, une démo avec des données fictives réalistes ; pour un vrai client je branche ses données, l'appli marche pareil. »
- *« Quel algo ? »* → « Régression logistique pour le scoring (probabilité de conversion), régression linéaire pour le délai de vente. »
- *« Comment tu récupères les données d'un client ? »* → « Export CSV/Excel de son CRM ou de son logiciel, ou connexion directe — puis nettoyage (Power Query / pandas). »
- *« Ça sert à quoi concrètement ? »* → « L'agence sait où mettre son budget pub, quels leads rappeler en premier, et fixe mieux ses prix. Du temps gagné, des décisions plus rapides. »

Tu sais maintenant **expliquer chaque brique**. C'est ça qui fait pro — pas la complexité, la **clarté**. 🐑
