# -*- coding: utf-8 -*-
"""
Détection d'attaques DDoS — application Streamlit
-------------------------------------------------
Pipeline (identique à l'atelier) :
  chargement -> prétraitement -> découpage 70/15/15 -> normalisation
  -> 5 modèles -> métriques -> prédiction

Lancer :  streamlit run app.py
Connexion : admin / admin
Aucun résultat n'est écrit "en dur" : tout est calculé à partir de dataset.csv.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------
# 1. Réglages généraux
# ----------------------------------------------------------------------------
st.set_page_config(page_title="DDoS Shield", page_icon="🛡️", layout="wide")

VIOLET = "#7C3AED"
CHEMIN_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")
UTILISATEUR, MOT_DE_PASSE = "admin", "admin"

# Colonnes retirées (comme dans l'atelier) pour éviter les "raccourcis"
COLONNES_EXCLUES = ["label", "timestamp", "src_ip", "dst_ip", "src_port"]
# Colonnes numériques saisies dans le formulaire de prédiction
COLONNES_NUM = ["duration_ms", "packet_count", "byte_count", "packets_per_second",
                "bytes_per_second", "avg_packet_size_bytes", "syn_count",
                "ack_count", "connections_10s", "unique_sources_10s"]
LIBELLES = {
    "duration_ms": "Durée du flux (ms)",
    "packet_count": "Nombre de paquets",
    "byte_count": "Nombre d'octets",
    "packets_per_second": "Paquets par seconde",
    "bytes_per_second": "Octets par seconde",
    "avg_packet_size_bytes": "Taille moyenne d'un paquet (octets)",
    "syn_count": "Nombre de paquets SYN",
    "ack_count": "Nombre de paquets ACK",
    "connections_10s": "Connexions sur 10 s",
    "unique_sources_10s": "Sources uniques sur 10 s",
}
# Ces modèles ont besoin de données normalisées (StandardScaler)
MODELES_NORMALISES = ["Régression linéaire", "Régression logistique", "SVM"]

# Logo simple : un bouclier violet avec une coche blanche (SVG)
LOGO_SVG = f"""
<svg width="{{t}}" height="{{t}}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <path d="M32 4 L56 13 V31 C56 46 46 56 32 61 C18 56 8 46 8 31 V13 Z" fill="{VIOLET}"/>
  <path d="M21 32 L29 40 L44 24" fill="none" stroke="white" stroke-width="5"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""


def logo(taille=48):
    return LOGO_SVG.replace("{t}", str(taille))


# Style moderne (cartes, boutons violets)
st.markdown(
    f"""
    <style>
      .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
          background: {VIOLET}; color: white; border: 0; border-radius: 10px;
          padding: .55rem 1.2rem; font-weight: 600;
      }}
      .stButton > button:hover, .stFormSubmitButton > button:hover {{ background: #6D28D9; color: white; }}
      div[data-testid="stMetric"] {{
          background: #F5F3FF; border: 1px solid #DDD6FE; border-radius: 14px; padding: 14px 18px;
      }}
      div[data-testid="stMetric"] * {{ color: #1F1535 !important; }}
      h1, h2, h3 {{ color: {VIOLET}; }}
      .bandeau {{ display:flex; align-items:center; gap:14px; margin-bottom:.5rem; }}
      .bandeau h1 {{ margin:0; padding:0; }}
      .resultat {{ border-radius:14px; padding:18px 22px; font-size:1.2rem; font-weight:700; }}
      .ok  {{ background:#ECFDF5; border:1px solid #6EE7B7; color:#065F46; }}
      .bad {{ background:#FEF2F2; border:1px solid #FCA5A5; color:#991B1B; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# 2. Fonctions du pipeline ML
# ----------------------------------------------------------------------------
def preparer_features(df, colonnes=None):
    """Transforme un tableau brut en variables numériques pour les modèles."""
    X = df.drop(columns=[c for c in COLONNES_EXCLUES if c in df.columns])
    X = pd.get_dummies(X, columns=["protocol", "dst_port"], dtype=int)
    if colonnes is not None:  # aligne sur les colonnes vues à l'entraînement
        X = X.reindex(columns=colonnes, fill_value=0)
    return X


@st.cache_resource(show_spinner=False)
def entrainer(chemin):
    """Charge, nettoie, découpe, normalise et entraîne les 5 modèles."""
    brut = pd.read_csv(chemin)

    # Prétraitement : doublons puis valeurs manquantes (remplacées par 0)
    df = brut.drop_duplicates().copy()
    nb_doublons = len(brut) - len(df)
    nb_nan = int(df.isna().sum().sum())
    df = df.fillna(0)

    y = df["label"]
    X = preparer_features(df)

    # Découpage 70 % entraînement / 15 % validation / 15 % test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    # Normalisation : ajustée sur l'entraînement uniquement (évite la fuite de données)
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_val = scaler.transform(X_val)

    # Les 5 modèles de l'atelier
    modeles = {
        "Régression linéaire": LinearRegression(),
        "Régression logistique": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel="rbf", C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                 eval_metric="logloss", random_state=42, n_jobs=2),
    }

    lignes = []
    for nom, m in modeles.items():
        if nom in MODELES_NORMALISES:
            m.fit(Xs_tr, y_tr)
            sortie = m.predict(Xs_val)
        else:
            m.fit(X_tr, y_tr)
            sortie = m.predict(X_val)
        if nom == "Régression linéaire":      # score continu -> classe (seuil 0,5)
            sortie = (sortie >= 0.5).astype(int)
        lignes.append({
            "Modèle": nom,
            "Accuracy": accuracy_score(y_val, sortie),
            "Précision": precision_score(y_val, sortie, zero_division=0),
            "Recall": recall_score(y_val, sortie, zero_division=0),
            "F1-score": f1_score(y_val, sortie, zero_division=0),
        })
    tableau = pd.DataFrame(lignes).sort_values("F1-score", ascending=False).reset_index(drop=True)

    return {
        "brut": brut, "df": df, "X": X, "colonnes": list(X.columns),
        "nb_doublons": nb_doublons, "nb_nan": nb_nan,
        "tailles": (len(X_tr), len(X_val), len(X_te)),
        "scaler": scaler, "modeles": modeles, "tableau": tableau,
        "X_test": X_te, "y_test": y_te,
    }


def predire(res, nom, X_new):
    """Prédit la classe (0/1) et, si possible, la probabilité d'attaque."""
    modele = res["modeles"][nom]
    X_in = res["scaler"].transform(X_new) if nom in MODELES_NORMALISES else X_new
    if nom == "Régression linéaire":
        return (modele.predict(X_in) >= 0.5).astype(int), None
    classes = modele.predict(X_in)
    proba = modele.predict_proba(X_in)[:, 1] if hasattr(modele, "predict_proba") else None
    return classes, proba


def charger():
    """Entraîne (une seule fois, puis mis en cache)."""
    if not os.path.exists(CHEMIN_CSV):
        st.error("Fichier dataset.csv introuvable à côté de app.py.")
        st.stop()
    with st.spinner("Entraînement des modèles en cours (première ouverture uniquement)…"):
        return entrainer(CHEMIN_CSV)


# ----------------------------------------------------------------------------
# 3. Page de connexion
# ----------------------------------------------------------------------------
def page_connexion():
    _, centre, _ = st.columns([1, 1.2, 1])
    with centre:
        st.markdown(f"<div style='text-align:center'>{logo(90)}</div>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align:center'>DDoS Shield</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center'>Détection d'attaques DDoS par Machine Learning</p>",
                    unsafe_allow_html=True)
        with st.form("connexion"):
            user = st.text_input("Nom d'utilisateur")
            mdp = st.text_input("Mot de passe", type="password")
            envoyer = st.form_submit_button("Se connecter", width="stretch")
        if envoyer:
            if user == UTILISATEUR and mdp == MOT_DE_PASSE:
                st.session_state["connecte"] = True
                st.rerun()
            else:
                st.error("Identifiants incorrects.")


# ----------------------------------------------------------------------------
# 4. Pages de l'application
# ----------------------------------------------------------------------------
def page_donnees():
    st.header("Exploration des données")
    brut = pd.read_csv(CHEMIN_CSV)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flux réseau", f"{len(brut):,}".replace(",", " "))
    c2.metric("Colonnes", brut.shape[1])
    c3.metric("Trafic normal (0)", int((brut["label"] == 0).sum()))
    c4.metric("Attaques DDoS (1)", int((brut["label"] == 1).sum()))

    st.caption("Jeu de données synthétique (pédagogique) : les scores ne prouvent pas "
               "une efficacité sur un réseau réel.")
    st.subheader("Aperçu")
    st.dataframe(brut.head(10), width="stretch")

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Répartition des classes")
        rep = brut["label"].map({0: "Normal", 1: "DDoS"}).value_counts()
        st.bar_chart(rep, color=VIOLET)
    with g2:
        st.subheader("Valeurs manquantes par colonne")
        manq = brut.isna().sum()
        st.bar_chart(manq[manq > 0], color=VIOLET)

    st.subheader("Statistiques descriptives")
    st.dataframe(brut.describe().T, width="stretch")


def page_pretraitement(res):
    st.header("Prétraitement")
    nb_tr, nb_val, nb_te = res["tailles"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Doublons supprimés", res["nb_doublons"])
    c2.metric("Valeurs manquantes (remplacées par 0)", res["nb_nan"])
    c3.metric("Lignes conservées", len(res["df"]))

    st.markdown(
        """
**Étapes appliquées (comme dans l'atelier) :**
1. Suppression des lignes en double (avant le découpage, pour ne pas fausser l'évaluation).
2. Remplacement des valeurs manquantes par 0 (`fillna(0)`).
3. Retrait de `timestamp`, `src_ip`, `dst_ip`, `src_port` : le modèle apprend le comportement du trafic, pas des identifiants.
4. Encodage du protocole (TCP/UDP/ICMP) et du port de destination en colonnes 0/1.
5. Découpage 70 % entraînement / 15 % validation / 15 % test (classes réparties de façon équilibrée).
6. Normalisation (StandardScaler) ajustée sur l'entraînement seulement.
        """
    )
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Variables après encodage", res["X"].shape[1])
    d2.metric("Entraînement", nb_tr)
    d3.metric("Validation", nb_val)
    d4.metric("Test", nb_te)
    with st.expander("Voir les variables utilisées par les modèles"):
        st.write(res["colonnes"])


def page_modeles(res):
    st.header("Entraînement et métriques")
    st.caption("Métriques calculées sur l'ensemble de **validation** (15 %). "
               "Le meilleur modèle est celui qui a le meilleur F1-score.")
    tableau = res["tableau"]
    st.dataframe(tableau.round(3), width="stretch", hide_index=True)
    st.bar_chart(tableau.set_index("Modèle")["F1-score"], color=VIOLET)
    st.info("Accuracy : bonnes prédictions globales · Précision : fiabilité des alertes · "
            "Recall : part des attaques détectées · F1 : équilibre précision/recall.")

    # Évaluation finale sur le test, pour le meilleur modèle
    meilleur = tableau.iloc[0]["Modèle"]
    st.subheader(f"Évaluation finale sur le test — {meilleur}")
    y_test = res["y_test"]
    pred, _ = predire(res, meilleur, res["X_test"])
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
    t2.metric("Précision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
    t3.metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.3f}")
    t4.metric("F1-score", f"{f1_score(y_test, pred, zero_division=0):.3f}")

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=["Normal", "DDoS"], cmap="Purples", ax=ax)
        ax.set_title("Matrice de confusion (test)")
        st.pyplot(fig)
    with g2:
        rapport = classification_report(y_test, pred, target_names=["Normal", "DDoS"],
                                        output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rapport).T.round(3), width="stretch")
        st.caption("Faux négatifs = attaques non détectées (les plus dangereux) ; "
                   "faux positifs = trafic normal signalé à tort.")


def remplir_exemple(res):
    """Remplit le formulaire avec une vraie ligne tirée au hasard du jeu de test."""
    i = np.random.choice(res["X_test"].index)
    ligne = res["df"].loc[i]
    st.session_state["f_protocol"] = ligne["protocol"]
    st.session_state["f_dst_port"] = int(ligne["dst_port"])
    for c in COLONNES_NUM:
        st.session_state[f"f_{c}"] = float(ligne[c])
    vrai = "DDoS" if int(ligne["label"]) == 1 else "Normal"
    st.session_state["info_exemple"] = (
        f"Exemple réel du jeu de test chargé (étiquette réelle : **{vrai}**). "
        "Valable tant que vous ne modifiez pas les valeurs.")


def page_prediction(res):
    st.header("Prédiction")
    tableau = res["tableau"]
    noms = list(tableau["Modèle"])
    choix = st.selectbox("Modèle utilisé", noms,
                         help="Par défaut : le meilleur F1-score sur la validation.")

    onglet1, onglet2 = st.tabs(["Un seul flux", "Fichier CSV (plusieurs flux)"])

    # ---- Un seul flux ----
    with onglet1:
        df = res["df"]
        ports = sorted(int(p) for p in df["dst_port"].unique())
        protocoles = sorted(df["protocol"].unique())
        # Valeurs par défaut = médianes du jeu de données (si pas déjà définies)
        st.session_state.setdefault("f_protocol", protocoles[0])
        st.session_state.setdefault("f_dst_port", ports[0])
        for c in COLONNES_NUM:
            st.session_state.setdefault(f"f_{c}", float(df[c].median()))

        st.button("🎲 Charger un exemple réel du jeu de test",
                  on_click=remplir_exemple, args=(res,))
        if "info_exemple" in st.session_state:
            st.caption(st.session_state["info_exemple"])

        with st.form("form_flux"):
            a, b = st.columns(2)
            a.selectbox("Protocole", protocoles, key="f_protocol")
            b.selectbox("Port de destination", ports, key="f_dst_port")
            cols = st.columns(3)
            for k, c in enumerate(COLONNES_NUM):
                cols[k % 3].number_input(LIBELLES[c], min_value=0.0, key=f"f_{c}")
            lancer = st.form_submit_button("Analyser ce flux", width="stretch")

        if lancer:
            ligne = {"protocol": st.session_state["f_protocol"],
                     "dst_port": st.session_state["f_dst_port"]}
            for c in COLONNES_NUM:
                ligne[c] = st.session_state[f"f_{c}"]
            X_new = preparer_features(pd.DataFrame([ligne]), res["colonnes"])
            classe, proba = predire(res, choix, X_new)
            if int(classe[0]) == 1:
                st.markdown("<div class='resultat bad'>⚠️ Attaque DDoS détectée</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div class='resultat ok'>✅ Trafic normal</div>",
                            unsafe_allow_html=True)
            if proba is not None:
                st.metric("Probabilité d'attaque", f"{proba[0] * 100:.1f} %")
            else:
                st.caption("Ce modèle ne fournit pas de probabilité.")

    # ---- Fichier CSV ----
    with onglet2:
        st.caption("Le fichier doit avoir les mêmes colonnes que dataset.csv "
                   "(la colonne `label` est facultative).")
        fichier = st.file_uploader("Déposer un fichier CSV", type="csv")
        if fichier is not None:
            nouveau = pd.read_csv(fichier)
            manquantes = [c for c in ["protocol", "dst_port"] + COLONNES_NUM
                          if c not in nouveau.columns]
            if manquantes:
                st.error(f"Colonnes manquantes : {', '.join(manquantes)}")
            else:
                propre = nouveau.fillna(0)
                X_new = preparer_features(propre, res["colonnes"])
                classes, proba = predire(res, choix, X_new)
                sortie = nouveau.copy()
                sortie["prediction"] = np.where(classes == 1, "DDoS", "Normal")
                if proba is not None:
                    sortie["proba_attaque"] = proba.round(4)
                m1, m2 = st.columns(2)
                m1.metric("Flux analysés", len(sortie))
                m2.metric("Flux classés DDoS", int((classes == 1).sum()))
                if "label" in nouveau.columns:
                    st.metric("Accuracy sur ce fichier",
                              f"{accuracy_score(nouveau['label'], classes):.3f}")
                st.dataframe(sortie.head(200), width="stretch")
                st.download_button("Télécharger les prédictions (CSV)",
                                   sortie.to_csv(index=False).encode("utf-8"),
                                   "predictions.csv", "text/csv")


# ----------------------------------------------------------------------------
# 5. Programme principal
# ----------------------------------------------------------------------------
def main():
    if not st.session_state.get("connecte"):
        page_connexion()
        return

    with st.sidebar:
        st.markdown(f"<div class='bandeau'>{logo(42)}<h3 style='margin:0'>DDoS Shield</h3></div>",
                    unsafe_allow_html=True)
        page = st.radio("Navigation",
                        ["Données", "Prétraitement", "Modèles et métriques", "Prédiction"])
        st.divider()
        if st.button("Se déconnecter", width="stretch"):
            st.session_state.clear()
            st.rerun()

    st.markdown(f"<div class='bandeau'>{logo(40)}<h1>Détection d'attaques DDoS</h1></div>",
                unsafe_allow_html=True)

    if page == "Données":
        page_donnees()
    else:
        res = charger()
        if page == "Prétraitement":
            page_pretraitement(res)
        elif page == "Modèles et métriques":
            page_modeles(res)
        else:
            page_prediction(res)


main()
