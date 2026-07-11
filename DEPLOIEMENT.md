# 🚀 Mettre l'appli en ligne (guide pas-à-pas)

But : obtenir un **lien permanent** (URL) qui marche 24h/24, à envoyer à tes clients — puis
éventuellement l'**intégrer dans florynepierson.com**. C'est **gratuit** (Streamlit Community Cloud).

Le dossier est **déjà prêt** (git initialisé, 1er commit fait, `venv` exclu). Il te reste 3 étapes.

---

## ÉTAPE 1 — Envoyer le code sur GitHub

### Le plus simple : GitHub Desktop (appli avec boutons, pas de terminal)
1. Installe **GitHub Desktop** : https://desktop.github.com (si tu ne l'as pas déjà).
2. Ouvre-le → connecte-toi avec ton compte GitHub.
3. Menu **File → Add Local Repository** → choisis le dossier :
   `~/Desktop/freelance-portfolio/dashboard-immo-predictif`
4. En haut, clique **Publish repository**.
   - Nom : `dashboard-immo` (par ex.)
   - **Décoche** « Keep this code private » → laisse-le **public** (c'est le plus simple pour l'hébergement gratuit).
   - Clique **Publish repository**.
✅ Ton code est sur GitHub.

### (Variante terminal, si tu préfères — comme pour ton site)
Crée d'abord un dépôt vide sur github.com (bouton **New repository**, nom `dashboard-immo`,
**sans** cocher « Add a README »). Puis, dans le Terminal :
```bash
cd ~/Desktop/freelance-portfolio/dashboard-immo-predictif
git remote add origin https://github.com/TON-PSEUDO/dashboard-immo.git
git branch -M main
git push -u origin main
```
(remplace `TON-PSEUDO` par ton pseudo GitHub)

---

## ÉTAPE 2 — Déployer sur Streamlit Cloud (gratuit)

1. Va sur **https://share.streamlit.io**
2. **Sign in** → avec **GitHub** → autorise.
3. Bouton **Create app** (ou « New app ») → **« Deploy a public app from GitHub »**.
4. Remplis :
   - **Repository** : `TON-PSEUDO/dashboard-immo`
   - **Branch** : `main`
   - **Main file path** : `app.py`
5. Clique **Deploy**.
6. Patiente **2-3 min** (il installe tout seul les librairies de `requirements.txt`).
7. 🎉 Tu obtiens une **URL du type** : `https://dashboard-immo-xxxx.streamlit.app`
   → **c'est le lien à envoyer à tes clients.** Il marche partout, sur téléphone comme sur ordi.

> Tu peux personnaliser un peu l'adresse dans **Settings → General** de l'appli.

---

## ÉTAPE 3 (optionnelle) — L'intégrer dans florynepierson.com

Une fois l'URL obtenue, ajoute ce bloc dans une page de ton site (là où tu veux montrer la démo) :

```html
<iframe
  src="https://dashboard-immo-xxxx.streamlit.app/?embed=true"
  style="width:100%; height:900px; border:none; border-radius:16px;"
  title="Démo — tableau de bord immobilier IA"
  loading="lazy">
</iframe>
```
Remplace l'URL par la tienne. Le `?embed=true` **masque le menu Streamlit** → l'appli a l'air
d'être **une partie de ton site**. 👌

---

## ⚠️ Bon à savoir (honnête)

- **Mise en veille** : sur l'offre **gratuite**, si personne ne visite l'appli pendant plusieurs
  jours, elle « s'endort ». Au réveil, la 1re visite prend ~30 secondes puis c'est instantané.
  → **Avant un rendez-vous client, ouvre le lien 1 minute avant** pour la « réveiller ». C'est tout.
- **Public** : l'appli est visible par qui a le lien. Parfait pour une démo. **Ne mets pas de vraies
  données client** dans la version publique.
- **Si le déploiement plante** sur une version de librairie → **envoie-moi le message d'erreur**,
  on fige les versions dans `requirements.txt` (2 min à corriger).
- **Mettre à jour l'appli** : à chaque fois que tu modifies le code, un `git push` (ou « Push origin »
  dans GitHub Desktop) → Streamlit **redéploie tout seul**. Magique.

---

Voilà — une fois l'ÉTAPE 2 finie, tu as un lien pro à dégainer à tout moment. 🐑
