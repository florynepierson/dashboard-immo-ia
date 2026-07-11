# -*- coding: utf-8 -*-
"""
Dashboard Agence Immobilière + IA — appli Streamlit (par Floryne Pierson)
Modèles : régression logistique (leads, monotone) + régression linéaire sur la
sur-évaluation (délai). Train/test split, scores honnêtes.  Lancer : streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "donnees")
st.set_page_config(page_title="Agence Immo — Dashboard IA", page_icon="🏡", layout="wide")

# Palette catégorielle chaude (accordée au site : or, olive, terracotta, tan…)
PALETTE = ["#98723f", "#b0895a", "#6f8f5f", "#b5674d", "#c9b48f", "#8a8072", "#5b7f80"]

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400&family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"], .stMarkdown, button, input, select, textarea { font-family:'Inter',-apple-system,sans-serif; }
  .stApp { background:#f4f1ea; }
  .block-container{padding-top:1.8rem;max-width:1280px;}
  h1,h2,h3,h4 { font-family:'Fraunces',Georgia,serif; color:#161a1d; font-weight:600; letter-spacing:-.01em; }
  /* Bandeau — carte ivoire, liseré or (comme le site) */
  .banner{background:linear-gradient(120deg,#fffdf9 0%,#f7f1e6 100%);border:1px solid #e7e0d4;border-left:4px solid #b0895a;border-radius:20px;padding:26px 30px;color:#161a1d;margin-bottom:8px;box-shadow:0 14px 40px rgba(60,45,25,.06);position:relative;overflow:hidden;}
  .banner .ey{font-size:.7rem;letter-spacing:.24em;text-transform:uppercase;color:#98723f;font-weight:600;margin-bottom:7px;}
  .banner h1{font-family:'Fraunces',serif;font-size:1.9rem;font-weight:500;margin:0;color:#161a1d;letter-spacing:-.01em;}
  .banner h1 em{font-style:italic;color:#98723f;}
  .banner p{margin:8px 0 0;color:#6b6357;font-size:.96rem;font-weight:400;}
  /* Cartes KPI — ivoire, liseré or */
  [data-testid="stMetric"]{background:#fffdf9;border:1px solid #e7e0d4;border-top:3px solid #b0895a;border-radius:16px;padding:18px 20px;box-shadow:0 6px 18px rgba(60,45,25,.05);}
  [data-testid="stMetricLabel"] p{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:#8a8072;font-weight:600;}
  [data-testid="stMetricValue"]{font-family:'Fraunces',serif;font-size:1.85rem;font-weight:600;color:#98723f;}
  button[data-baseweb="tab"]{font-weight:600;font-size:1rem;color:#6b6357;}
  button[data-baseweb="tab"][aria-selected="true"]{color:#98723f;}
  hr{border-color:#e7e0d4;}
</style>
""", unsafe_allow_html=True)

# ---------- Données ----------
@st.cache_data
def load(_schema="v4-prix-reels"):  # bump _schema to invalidate the cache when the CSV columns change
    biens = pd.read_csv(os.path.join(DATA, "biens.csv"))
    leads = pd.read_csv(os.path.join(DATA, "leads.csv"))
    for c in ["prix", "commission", "surface_m2", "nb_vues", "nb_contacts", "delai_vente_jours"]:
        biens[c] = pd.to_numeric(biens[c], errors="coerce")
    biens["date_mandat"] = pd.to_datetime(biens["date_mandat"], errors="coerce")
    biens["date_vente"] = pd.to_datetime(biens["date_vente"], errors="coerce")
    leads["date"] = pd.to_datetime(leads["date"], errors="coerce")
    return biens, leads

biens, leads = load()

# ---------- Modèles (train/test → scores honnêtes) ----------
LEAD_FEATURES = ["temps_reponse_h", "nb_relances", "budget", "source"]

@st.cache_resource
def train_lead_model(leads):
    X = pd.get_dummies(leads[LEAD_FEATURES], columns=["source"], drop_first=True)
    y = leads["converti"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)  # scaler → coefficients justes
    acc = accuracy_score(yte, m.predict(Xte))
    try:
        auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    except Exception:
        auc = None
    return m, list(X.columns), acc, auc

@st.cache_resource
def train_delai_model(biens):
    s = biens[(biens["statut"] == "Vendu") & (biens["type"] != "Terrain")].dropna(subset=["delai_vente_jours"]).copy()
    s["prix_m2"] = s["prix"] / s["surface_m2"]
    ref = s.groupby("ville")["prix_m2"].median()                 # prix médian au m² par ville
    s["sur_eval"] = s["prix_m2"] / s["ville"].map(ref)           # >1 = surévalué / <1 = sous-évalué
    # NB : on n'utilise PAS 'type' — il est confondu avec la surface (maison=grand, appart=moyen,
    # studio=petit), donc à surface fixe il ne fait qu'extrapoler des combinaisons inexistantes.
    feats = ["sur_eval", "surface_m2", "ville"]
    X = pd.get_dummies(s[feats], columns=["ville"], drop_first=True)
    y = s["delai_vente_jours"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    pred = m.predict(Xte)
    return m, list(X.columns), mean_absolute_error(yte, pred), r2_score(yte, pred), ref

@st.cache_resource
def train_prix_model(biens):
    # AVM (estimation par comparaison, automatisée) : prix ~ surface + ville, sur le BÂTI
    # (on écarte les terrains : leur m² n'a rien à voir avec celui d'un logement).
    s = biens[biens["type"] != "Terrain"].dropna(subset=["prix", "surface_m2"]).copy()
    feats = ["surface_m2", "nb_pieces", "type", "etat", "dpe", "ville"]
    X = pd.get_dummies(s[feats], columns=["type", "etat", "dpe", "ville"], drop_first=True)
    y = s["prix"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    pred = m.predict(Xte)
    return m, list(X.columns), mean_absolute_error(yte, pred), r2_score(yte, pred)

lead_model, lead_cols, lead_acc, lead_auc = train_lead_model(leads)
delai_model, delai_cols, delai_mae, delai_r2, delai_ref = train_delai_model(biens)
prix_model, prix_cols, prix_mae, prix_r2 = train_prix_model(biens)

def score_lead(source, budget, temps, relances):
    row = pd.DataFrame([{"temps_reponse_h": temps, "nb_relances": relances, "budget": budget, "source": source}])
    row = pd.get_dummies(row, columns=["source"]).reindex(columns=lead_cols, fill_value=0)
    return float(lead_model.predict_proba(row)[0, 1])

def predict_delai(prix, surface, ville):
    se = (prix / max(1, surface)) / delai_ref.get(ville, delai_ref.median())
    row = pd.DataFrame([{"sur_eval": se, "surface_m2": surface, "ville": ville}])
    row = pd.get_dummies(row, columns=["ville"]).reindex(columns=delai_cols, fill_value=0)
    return max(1, int(delai_model.predict(row)[0]))

def predict_prix(surface, pieces, typ, etat, dpe, ville):
    row = pd.DataFrame([{"surface_m2": surface, "nb_pieces": pieces, "type": typ, "etat": etat, "dpe": dpe, "ville": ville}])
    row = pd.get_dummies(row, columns=["type", "etat", "dpe", "ville"]).reindex(columns=prix_cols, fill_value=0)
    return max(0.0, float(prix_model.predict(row)[0]))

def forecast_ca(y):
    y = np.asarray(y, dtype=float)
    if len(y) >= 6:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            fit = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated").fit()
            return np.clip(fit.forecast(3), 0, None), "lissage exponentiel (Holt)"
        except Exception:
            pass
    coef = np.polyfit(np.arange(len(y)), y, 1)
    return np.clip(np.polyval(coef, np.arange(len(y), len(y) + 3)), 0, None), "tendance linéaire"

def recommandations(fb, fl):
    """Génère des recommandations business à partir des données filtrées."""
    recs = []
    if len(fl) > 20:
        fast = fl[fl["temps_reponse_h"] < 2]["converti"].mean()
        slow = fl[fl["temps_reponse_h"] >= 2]["converti"].mean()
        if pd.notna(fast) and pd.notna(slow) and fast > slow + 0.05:
            recs.append(("warn", f"⏱️ Répondre en **moins de 2 h** convertit à **{fast*100:.0f}%** contre **{slow*100:.0f}%** au-delà. Votre temps de réponse moyen est de **{fl['temps_reponse_h'].mean():.1f} h** → accélérez la prise de contact."))
        r_hi = fl[fl["nb_relances"] >= 2]["converti"].mean()
        r_lo = fl[fl["nb_relances"] <= 1]["converti"].mean()
        if pd.notna(r_hi) and pd.notna(r_lo) and r_hi > r_lo + 0.05:
            recs.append(("ok", f"🔁 Les leads relancés **2 fois ou plus** convertissent à **{r_hi*100:.0f}%** contre **{r_lo*100:.0f}%** — relancez davantage."))
    g = fl.groupby("source").agg(n=("converti", "size"), taux=("converti", "mean"))
    g = g[g["n"] >= 8]
    if len(g) >= 2:
        best, worst = g["taux"].idxmax(), g["taux"].idxmin()
        recs.append(("ok", f"📣 Votre **meilleure source** : **{best}** ({g.loc[best, 'taux']*100:.0f}% de conversion) → investissez-y."))
        recs.append(("warn", f"📉 **{worst}** ne convertit qu'à **{g.loc[worst, 'taux']*100:.0f}%** — reconsidérez le budget dépensé sur ce canal."))
    now = pd.concat([biens["date_mandat"], biens["date_vente"]]).max()
    en_cours = fb[fb["statut"] == "En cours"].copy()
    if len(en_cours) and pd.notna(now):
        en_cours["age"] = (now - en_cours["date_mandat"]).dt.days
        stale = en_cours[en_cours["age"] > 150]
        if len(stale):
            recs.append(("warn", f"🏚️ **{len(stale)} bien(s)** sont en vente depuis plus de **150 jours** — à repositionner en prix ou à re-photographier."))
    return recs

# ---------- Filtres ----------
st.sidebar.header("Filtres")
villes = st.sidebar.multiselect("Ville", sorted(biens["ville"].unique()), default=list(biens["ville"].unique()))
agents = st.sidebar.multiselect("Agent", sorted(biens["agent"].unique()), default=list(biens["agent"].unique()))
st.sidebar.caption("Démo — données fictives · Floryne Pierson")
fb = biens[biens["ville"].isin(villes) & biens["agent"].isin(agents)]
fl = leads[leads["ville"].isin(villes) & leads["agent"].isin(agents)]

st.markdown("""<div class="banner">
  <div class="ey">🏡 Tableau de bord agence · pilotage &amp; IA</div>
  <h1>Vos leads, vos ventes et vos prédictions — en un coup d'œil</h1>
  <p>Mis à jour automatiquement · aide à décider où mettre le budget, qui rappeler, à quel prix vendre · démo par Floryne Pierson</p>
</div>""", unsafe_allow_html=True)

# --- Périmètre actif : rend explicite DE QUI / DE QUOI parlent les chiffres affichés ---
_na, _nv = biens["agent"].nunique(), biens["ville"].nunique()
if len(agents) == 1:
    _who = f"👤 Résultats de <b style='color:#98723f'>{agents[0]}</b>"
elif len(agents) >= _na:
    _who = "👥 Tous les agents"
else:
    _who = f"👥 {len(agents)} agents sélectionnés"
if len(villes) == 1:
    _where = f"<b style='color:#98723f'>{villes[0]}</b>"
elif len(villes) >= _nv:
    _where = "toutes les villes"
else:
    _where = f"{len(villes)} villes"
st.markdown(f"<div style='margin:6px 0 14px;font-size:.95rem;color:#6b6357'>{_who} &nbsp;·&nbsp; 📍 {_where}</div>", unsafe_allow_html=True)

vendus = fb[fb["statut"] == "Vendu"]
# Repères "marché local" — indicatifs (démo). Pour un vrai client : brancher les stats réelles du secteur.
MKT_CONV, MKT_DELAI, MKT_REP = 18.0, 85.0, 6.0
conv = fl["converti"].mean() * 100
delai = vendus["delai_vente_jours"].mean()
rep = fl["temps_reponse_h"].mean()
conv = 0 if pd.isna(conv) else conv
delai = 0 if pd.isna(delai) else delai
rep = 0 if pd.isna(rep) else rep
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Commissions encaissées", f"{vendus['commission'].sum():,.0f} €".replace(",", " "))
c2.metric("Biens vendus", f"{len(vendus)}")
c3.metric("Taux de conversion", f"{conv:.1f} %", f"{conv - MKT_CONV:+.1f} pts vs marché")
c4.metric("Délai moyen de vente", f"{delai:.0f} j", f"{delai - MKT_DELAI:+.0f} j vs marché", delta_color="inverse")
c5.metric("Réactivité (1er contact)", f"{rep:.1f} h", f"{rep - MKT_REP:+.1f} h vs marché", delta_color="inverse")
st.caption("▸ Flèche verte = **mieux que le marché local**. *(Repères marché indicatifs pour la démo — branchés sur les vraies stats du secteur chez un client.)*")

st.divider()
tab1, tab2 = st.tabs(["📊 Pilotage", "🔮 Prédictions IA"])

with tab1:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Commissions par mois + prévision 3 mois")
        v = vendus.dropna(subset=["date_vente"]).copy()
        v["mois"] = v["date_vente"].dt.to_period("M").dt.to_timestamp()
        monthly = v.groupby("mois")["commission"].sum().reset_index().sort_values("mois")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["mois"], y=monthly["commission"], mode="lines+markers", name="Réel", line=dict(color="#98723f", width=3)))
        if len(monthly) >= 4:
            fc, method = forecast_ca(monthly["commission"])
            fm = pd.date_range(monthly["mois"].max(), periods=4, freq="MS")[1:]
            fig.add_trace(go.Scatter(x=fm, y=fc, mode="lines+markers", name="Prévu (3 mois)", line=dict(color="#6f8f5f", width=3, dash="dot")))
            st.caption("Méthode : " + method + ". *(Avec 2+ ans d'historique, on ajouterait la saisonnalité.)*")
        fig.update_layout(height=320, margin=dict(t=10, b=0, l=0, r=0), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Leads par canal d'arrivée")
        st.caption("Le **canal** = par où le prospect est venu (portail d'annonces, site de l'agence, vitrine, bouche-à-oreille). La couleur = **taux de conversion** du canal.")
        g = fl.groupby("source").agg(recus=("converti", "size"), gagnes=("converti", "sum")).reset_index()
        g["taux"] = (g["gagnes"] / g["recus"] * 100).round(1)
        gs = g.sort_values("recus"); gs["label"] = gs["taux"].astype(str) + " %"
        fig2 = px.bar(gs, x="recus", y="source", orientation="h", text="label", color="taux",
                      color_continuous_scale="YlOrBr", labels={"recus": "Leads reçus", "source": ""})
        fig2.update_layout(height=286, margin=dict(t=10, b=0, l=0, r=0), coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.subheader("Pipeline commercial")
        st.caption("Un prospect avance par étapes : **nouveau → qualifié → visite → vente**. À chaque marche, certains abandonnent. Les barres montrent **combien il en reste** à chaque étape → on voit **où l'agence perd le plus de monde**.")
        order = ["Nouveau", "Qualifié", "Visite", "Gagné"]
        fn = fl[fl["statut"].isin(order)].groupby("statut").size().reindex(order).fillna(0)
        figf = go.Figure(go.Funnel(y=order, x=fn.values, textinfo="value+percent initial",
                                   marker=dict(color=["#d8ccb8", "#c9b48f", "#b0895a", "#98723f"])))
        figf.update_layout(height=280, margin=dict(t=10, b=0, l=0, r=0))
        st.plotly_chart(figf, use_container_width=True)
    with b:
        st.subheader("Délai moyen de vente par ville")
        st.caption("Comparé au repère marché — au-dessus = biens qui traînent (souvent surévalués).")
        dvv = vendus.groupby("ville")["delai_vente_jours"].mean().reset_index().sort_values("delai_vente_jours")
        figd = px.bar(dvv, x="delai_vente_jours", y="ville", orientation="h",
                      color="ville", color_discrete_sequence=PALETTE, labels={"delai_vente_jours": "jours", "ville": ""})
        figd.add_vline(x=MKT_DELAI, line_dash="dash", line_color="#b6ac9c", annotation_text="marché ~85 j", annotation_position="top")
        figd.update_layout(height=280, margin=dict(t=10, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(figd, use_container_width=True)
    a2, b2 = st.columns(2)
    with a2:
        st.subheader("Commissions par agent")
        ag = vendus.groupby("agent")["commission"].sum().reset_index().sort_values("commission")
        st.plotly_chart(px.bar(ag, x="commission", y="agent", orientation="h", color="agent", color_discrete_sequence=PALETTE).update_layout(height=260, margin=dict(t=10, b=0, l=0, r=0), showlegend=False), use_container_width=True)
    with b2:
        st.subheader("Portefeuille par statut")
        stt = fb.groupby("statut").size().reset_index(name="n")
        st.plotly_chart(px.pie(stt, names="statut", values="n", hole=.55, color_discrete_sequence=["#98723f", "#b0895a", "#c9b48f", "#6f8f5f", "#b6ac9c"]).update_layout(height=260, margin=dict(t=10, b=0, l=0, r=0)), use_container_width=True)

    st.divider()
    st.subheader("💡 Recommandations automatiques")
    st.caption("Générées à partir de vos données — ce que l'agence peut améliorer concrètement.")
    _recs = recommandations(fb, fl)
    if _recs:
        for _k, _m in _recs:
            (st.success if _k == "ok" else st.warning)(_m)
    else:
        st.info("Pas assez de données dans ce filtre pour générer des recommandations.")

with tab2:
    with st.expander("ℹ️ Comment lire ces prédictions (à ouvrir)"):
        st.markdown(
            "- **Scoring de leads** : une **probabilité qu'un prospect devienne client**. But : savoir **qui rappeler en premier**.\n"
            "- **Canal d'arrivée** : par où le prospect est venu (LeBonCoin, SeLoger, site de l'agence, vitrine, recommandation). Certains canaux convertissent mieux → savoir lesquels aide à **placer le budget pub**.\n"
            "- **Délai de vente** : une **estimation en jours** avant de vendre un bien. But : **aider à fixer le prix**.\n"
            "- Modèles **testés sur des données jamais vues** (train/test) → les scores sont **honnêtes**.")

    st.subheader("🎯 Scoring des leads — qui rappeler en priorité ?")
    fiab = f"Fiabilité : **{lead_acc*100:.0f}%** de bonnes prédictions sur données de test"
    if lead_auc:
        fiab += f" · AUC {lead_auc:.2f}"
    st.caption(fiab + ". *(Régression logistique — monotone : bonne source / réponse rapide / relances → toujours plus de chances.)*")

    open_leads = fl[fl["statut"].isin(["Nouveau", "Qualifié", "Visite"])].copy()
    if len(open_leads):
        Xo = pd.get_dummies(open_leads[LEAD_FEATURES], columns=["source"]).reindex(columns=lead_cols, fill_value=0)
        open_leads["score"] = (lead_model.predict_proba(Xo)[:, 1] * 100).round().astype(int)
        open_leads["temp"] = open_leads["score"].apply(
            lambda s: "🔥 Chaud" if s >= 60 else ("🟠 Tiède" if s >= 40 else "❄️ Froid"))
        opts = ["🔥 Chaud", "🟠 Tiède", "❄️ Froid"]
        choix = st.multiselect("Filtrer par température", opts, default=opts,
                               help="🔥 Chaud = à rappeler en priorité · 🟠 Tiède = à travailler · ❄️ Froid = peu probable")
        tbl = open_leads[open_leads["temp"].isin(choix)].sort_values("score", ascending=False)
        show = tbl[["temp", "nom", "source", "ville", "budget", "temps_reponse_h", "nb_relances", "score"]]
        st.dataframe(
            show, use_container_width=True, hide_index=True, height=430,
            column_config={
                "temp": st.column_config.TextColumn("Priorité", help="🔥 Chaud (≥60%) à rappeler d'abord · 🟠 Tiède (40-59%) · ❄️ Froid (<40%)", width="small"),
                "nom": st.column_config.TextColumn("Prospect", help="Nom du prospect"),
                "source": st.column_config.TextColumn("Canal d'arrivée", help="Par où le prospect est venu"),
                "ville": st.column_config.TextColumn("Ville"),
                "budget": st.column_config.NumberColumn("Budget", help="Budget d'achat du prospect", format="%d €"),
                "temps_reponse_h": st.column_config.NumberColumn("Réponse", help="Délai avant le 1er contact de l'agence", format="%.1f h"),
                "nb_relances": st.column_config.NumberColumn("Relances", help="Nombre de fois recontacté"),
                "score": st.column_config.ProgressColumn("Probabilité de conversion", help="Chances estimées de devenir client (modèle IA)", format="%d%%", min_value=0, max_value=100),
            })
        st.caption(f"**{len(tbl)} prospects affichés** ({(open_leads['score'] >= 60).sum()} 🔥 chauds à rappeler en priorité) · triés du plus chaud au plus froid · filtre par température au-dessus.")

    st.markdown("**Teste un prospect en direct :**")
    f1, f2, f3, f4 = st.columns(4)
    src = f1.selectbox("Canal d'arrivée", sorted(leads["source"].unique()))
    bud = f2.select_slider("Budget (€)", options=[120000, 150000, 180000, 220000, 260000, 300000, 350000, 420000], value=220000)
    tps = f3.slider("Répondu en… (h)", 0.5, 72.0, 2.0, 0.5)
    rel = f4.slider("Nb de relances", 0, 4, 1)
    p = score_lead(src, bud, tps, rel)
    st.metric("Probabilité de devenir client", f"{p*100:.0f} %")
    (st.success if p > .5 else st.warning)(f"**{p*100:.0f}%** — " + ("prospect chaud, à rappeler vite." if p > .5 else "à travailler (répondre plus vite / relancer ferait monter ce score)."))
    st.caption("Le % reflète la vraie incertitude (même un bon lead n'est jamais sûr) — l'important est le **classement**.")

    st.divider()
    st.subheader("💶 Estimation de prix — à combien mettre le bien en vente ?")
    st.markdown("<div style='background:#fdf6e9;border:1px solid #e7d9bd;border-left:4px solid #b0895a;border-radius:10px;padding:10px 14px;font-size:.86rem;color:#6b6357;margin-bottom:6px'>⚠️ <b>Démonstration</b> — les prix au m² sont fictifs mais <b>calés sur les niveaux réels du secteur</b> (Metz, Thionville, Nancy…). Sur les vraies transactions d'une agence, le modèle se <b>recalibre automatiquement</b> sur son marché.</div>", unsafe_allow_html=True)
    st.caption(f"Fiabilité : erreur moyenne **± {prix_mae:,.0f} €** sur données de test (R² {prix_r2:.2f}). *(Régression sur surface, type, nb de pièces, état et DPE — par ville. Une estimation par comparaison, automatisée.)*".replace(",", " "))
    TYPES_BATI = [t for t in ["Studio", "Appartement", "Maison"] if t in biens["type"].unique()]
    e1, e2, e3 = st.columns(3)
    etype = e1.selectbox("Type de bien", TYPES_BATI, index=min(1, len(TYPES_BATI) - 1), key="prix_type")
    es = e2.number_input("Surface (m²)", 15, 400, 70, 5, key="prix_surf")
    epieces = e3.number_input("Nb de pièces", 1, 10, 3, 1, key="prix_pieces")
    e4, e5, e6 = st.columns(3)
    eetat = e4.selectbox("État", ["Neuf", "Rénové", "Bon", "À rafraîchir", "Travaux"], index=2, key="prix_etat")
    edpe = e5.selectbox("DPE (étiquette énergie)", ["A", "B", "C", "D", "E", "F", "G"], index=3, key="prix_dpe")
    ev = e6.selectbox("Ville", sorted(biens["ville"].unique()), key="prix_ville")
    pest = predict_prix(es, epieces, etype, eetat, edpe, ev)
    st.metric("Prix estimé", f"{pest:,.0f} €".replace(",", " "), help=f"± {prix_mae:,.0f} €".replace(",", " "))
    st.caption(f"Fourchette réaliste : **{pest - prix_mae:,.0f} € – {pest + prix_mae:,.0f} €** · soit environ **{pest / max(1, es):,.0f} €/m²**.".replace(",", " "))
    st.info("Estimation pour un bien **bâti**, basée sur **surface · type · nb de pièces · état · DPE · ville**. Avec les données réelles d'une agence, on ajouterait aussi l'étage, l'exposition, l'extérieur, la vue. C'est une **aide à l'estimation**, pas une expertise.")

    st.divider()
    st.subheader("⏱️ Délai de vente estimé — pour fixer le prix")
    st.caption(f"Fiabilité : erreur moyenne **± {delai_mae:.0f} jours** sur données de test (R² {delai_r2:.2f}). *(Régression linéaire sur la sur-évaluation vs le prix médian de la ville.)*")
    d1, d2, d3 = st.columns(3)
    dp = d1.number_input("Prix affiché (€)", 50000, 900000, 200000, 10000)
    ds = d2.number_input("Surface (m²)", 15, 400, 70, 5)
    dv = d3.selectbox("Ville", sorted(biens["ville"].unique()))
    est = predict_delai(dp, ds, dv)
    st.metric("Délai estimé", f"≈ {est} jours", help=f"± {delai_mae:.0f} jours")
    st.info("Un délai long = **souvent** un prix trop haut — mais **pas seulement** : photos, description, état du bien, saison, marché jouent aussi. Le modèle ne voit **que** le prix, la surface et la ville → c'est une **aide à la décision**, pas une vérité absolue. *(Le type de bien n'est pas utilisé : dans les données il est confondu avec la surface — une maison est toujours grande, un studio toujours petit — donc il fausserait l'estimation à surface fixe.)*")

st.divider()
st.caption("Conçu par **Floryne Pierson** · Python · scikit-learn · Streamlit — données fictives, modèles évalués sur jeu de test.")
