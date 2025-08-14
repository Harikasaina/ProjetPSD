import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import warnings

# --- Configuration de la page ---
st.set_page_config(page_title="Dashboard Marketing PSD 2.0", layout="wide")
st.title("📊 Dashboard d'Analyse Marketing et Segmentation Client")

# --- MODIFICATION : Ignorer les avertissements inutiles de XGBoost ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- Chargement des données et du modèle ---
@st.cache_data
def load_data():
    try:
        clients = pd.read_csv("clustered_clients01.csv")
        campaigns = pd.read_csv("campaign_performance.csv", engine='python')
        return clients, campaigns
    except FileNotFoundError:
        st.error("Erreur : Assurez-vous que les fichiers 'clustered_clients01.csv' et 'campaign_performance.csv' sont bien téléversés.")
        return None, None

@st.cache_resource
def load_model():
    try:
        model = joblib.load('loyalty_model.joblib')
        columns = joblib.load('model_columns.pkl')
        return model, columns
    except FileNotFoundError:
        return None, None

clients_df, campaigns_df = load_data()
model, model_columns = load_model()

if clients_df is None or campaigns_df is None:
    st.error("Erreur : Fichiers de données introuvables. Veuillez les téléverser.")
    st.stop()

# --- Interface utilisateur ---
st.sidebar.header("Navigation")
page_options = ["📈 Vue d'ensemble", "🧑‍🤝‍🧑 Analyse des Segments", "📢 Performance des Campagnes"]
if model is not None:
    page_options.append("🔮 Prédiction de Fidélité")

page = st.sidebar.radio("Choisissez une page :", page_options)

# ... Les 3 premières pages
if page == "📈 Vue d'ensemble":
    st.header("Vue d'ensemble des Indicateurs Clés")
    total_clients = clients_df['Customer_ID'].nunique()
    total_revenue = clients_df['Total_Spent'].sum()
    avg_revenue_per_client = total_revenue / total_clients
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre total de clients", f"{total_clients}")
    col2.metric("Chiffre d'affaires total", f"{total_revenue:,.2f} €")
    col3.metric("Panier moyen par client", f"{avg_revenue_per_client:,.2f} €")
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Répartition des clients par segment")
        cluster_counts = clients_df['Cluster'].value_counts().reset_index()
        fig_pie = px.pie(cluster_counts, names='Cluster', values='count', title="Proportion de chaque segment")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_b:
        st.subheader("Chiffre d'affaires par segment")
        revenue_by_cluster = clients_df.groupby('Cluster')['Total_Spent'].sum().reset_index()
        fig_bar = px.bar(revenue_by_cluster, x='Cluster', y='Total_Spent', title="Contribution de chaque segment au CA", color='Cluster')
        st.plotly_chart(fig_bar, use_container_width=True)

elif page == "🧑‍🤝‍🧑 Analyse des Segments":
    st.header("Exploration détaillée des segments de clients")
    cluster_list = sorted(clients_df['Cluster'].unique())
    selected_cluster = st.selectbox("Sélectionnez un segment à analyser :", cluster_list)
    st.markdown(f"### Analyse du Segment {selected_cluster}")
    segment_df = clients_df[clients_df['Cluster'] == selected_cluster]
    seg1, seg2, seg3 = st.columns(3)
    seg1.metric("Nombre de clients", f"{segment_df['Customer_ID'].nunique()}")
    seg2.metric("Âge moyen", f"{segment_df['Age'].mean():.1f} ans")
    seg3.metric("Dépense moyenne", f"{segment_df['Total_Spent'].mean():,.2f} €")
    st.markdown("---")
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Distribution de l'âge")
        fig_age = px.histogram(segment_df, x='Age', nbins=20, title=f"Distribution de l'âge (Segment {selected_cluster})")
        st.plotly_chart(fig_age, use_container_width=True)
    with col_d:
        st.subheader("Distribution des dépenses")
        fig_spent = px.histogram(segment_df, x='Total_Spent', nbins=20, title=f"Distribution des dépenses (Segment {selected_cluster})")
        st.plotly_chart(fig_spent, use_container_width=True)

elif page == "📢 Performance des Campagnes":
    st.header("Analyse de la Performance des Campagnes Marketing")
    total_budget = campaigns_df['Budget'].sum()
    total_revenue_campaigns = campaigns_df['Revenue'].sum()
    overall_roi = ((total_revenue_campaigns - total_budget) / total_budget) * 100 if total_budget > 0 else 0
    camp1, camp2, camp3 = st.columns(3)
    camp1.metric("Budget total investi", f"{total_budget:,.2f} €")
    camp2.metric("Revenu total généré", f"{total_revenue_campaigns:,.2f} €")
    camp3.metric("ROI Global", f"{overall_roi:.2f} %")
    st.markdown("---")
    kpi_choice = st.selectbox("Choisissez un indicateur à visualiser par canal :", ['ROI (%)', 'CTR (%)', 'CPA (€)', 'CPC (€)', 'Conversions'])
    st.subheader(f"{kpi_choice} par Canal Marketing")
    if kpi_choice in ['Conversions', 'Budget', 'Revenue']:
        perf_by_channel = campaigns_df.groupby('Channel')[kpi_choice].sum().reset_index()
    else:
        perf_by_channel = campaigns_df.groupby('Channel')[kpi_choice].mean().reset_index()
    fig_campaign = px.bar(perf_by_channel, x='Channel', y=kpi_choice, title=f"Performance ({kpi_choice}) par canal", color='Channel')
    st.plotly_chart(fig_campaign, use_container_width=True)

# ==============================================================================
# PAGE 4 : PRÉDICTION DE FIDÉLITÉ
# ==============================================================================
elif page == "🔮 Prédiction de Fidélité":
    st.header("Prédiction de la Fidélité d'un Client")

    if model is None or model_columns is None:
        st.error("Le modèle de prédiction n'est pas chargé. Veuillez vérifier que les fichiers 'loyalty_model.joblib' et 'model_columns.pkl' sont bien téléversés.")
    else:
        st.info("Entrez les informations d'un client pour prédire sa probabilité d'être fidèle (plus de 2 commandes).")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Âge du client", 18, 100, 35)
            total_spent = st.number_input("Total dépensé par le client (€)", 0.0, 10000.0, 500.0, 0.01)
            total_orders = st.number_input("Nombre total de commandes", 1, 100, 1)
            recency = st.number_input("Jours depuis le dernier achat (Récence)", 0, 365, 30)
            total_quantity = st.number_input("Quantité totale d'articles achetés", 1, 500, 10)

        with col2:
            gender = st.selectbox("Genre", ["Female", "Male"])
            locations = clients_df['Location'].unique().tolist()
            location = st.selectbox("Ville (Location)", locations)

        if st.button("Lancer la Prédiction"):
            input_data = pd.DataFrame(columns=model_columns)
            input_data.loc[0] = 0

            input_data['Age'] = age
            input_data['Total_Spent_Calc'] = total_spent
            input_data['Total_Orders'] = total_orders
            input_data['Recency'] = recency
            input_data['Total_Quantity'] = total_quantity

            if f"Gender_{gender}" in model_columns:
                input_data[f"Gender_{gender}"] = 1
            if f"Location_{location}" in model_columns:
                input_data[f"Location_{location}"] = 1

            input_data = input_data[model_columns]

            prediction_proba = model.predict_proba(input_data)[0][1]
            prediction_class = (prediction_proba > 0.5).astype(int)

            st.subheader("Résultat de la Prédiction")
            if prediction_class == 1:
                st.success(f"Ce client est susceptible d'être **fidèle**.")
            else:
                st.warning(f"Ce client est susceptible d'être **occasionnel**.")

            # --- On convertit la probabilité en float standard ---
            st.progress(float(prediction_proba))
            st.metric(label="Probabilité d'être fidèle", value=f"{prediction_proba:.2%}")


pour info, voici le code de mon côté# ==============================================================================

# # Génération du rapport final en pdf

# ==============================================================================

import os 

import pandas as pd

from reportlab.lib.pagesizes import A4

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib.units import cm

from reportlab.lib import colors



# --- Assurez-vous que vos données et modèle sont chargés ---

# (Ce code suppose que les variables 'client_features', 'model', 'model_columns' existent déjà)

client_features = pd.read_csv("clustered_clients01.csv")

model = joblib.load('loyalty_model.joblib')

model_columns = joblib.load('model_columns.pkl')



# --- FONCTION DE GÉNÉRATION DU RAPPORT FINAL ---

def generer_rapport_coeur_de_projet(client_features, model, model_columns):

    doc = SimpleDocTemplate("rapport_strategique_final.pdf", pagesize=A4)

    styles = getSampleStyleSheet()

    story = []



    # --- 1. Titre et Contexte (M1) ---

    story.append(Paragraph("Rapport d'Analyse et Stratégie Marketing", styles['Title']))

    story.append(Spacer(1, 1*cm))

    story.append(Paragraph("<b>Contexte Stratégique (M1)</b>", styles['h2']))

    story.append(Paragraph("L'objectif de ce projet est de développer une stratégie marketing personnalisée. L'analyse SWOT a révélé une forte opportunité dans l'utilisation de nos données clients via l'IA pour un ciblage précis, tout en étant conscient des menaces réglementaires et de la concurrence.", styles['BodyText']))

    story.append(PageBreak())



    # --- 2. Profils des Segments Clients (M4) ---

    story.append(Paragraph("<b>Profils des Segments Clients (M4)</b>", styles['h1']))

    story.append(Spacer(1, 0.5*cm))

    

    # Description textuelle des personas (votre texte)

    story.append(Paragraph("<b>Cluster 0 – Seniors dépensiers :</b> Clients plus âgés avec un panier moyen élevé, sensibles à la qualité.", styles['BodyText']))

    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("<b>Cluster 1 – Jeunes à faible pouvoir d’achat :</b> Jeunes adultes au budget limité, sensibles aux promotions.", styles['BodyText']))

    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("<b>Cluster 2 – Jeunes très rentables :</b> Le segment le plus important en valeur, achètent beaucoup et sont sensibles aux nouveautés.", styles['BodyText']))

    story.append(Spacer(1, 0.2*cm))

    story.append(Paragraph("<b>Cluster 3 – Seniors modérés :</b> Nombreux mais au pouvoir d'achat modéré, sensibles à la fiabilité et au service.", styles['BodyText']))

    story.append(PageBreak())

    

    # --- 3. Potentiel de l'IA (M6) ---

    story.append(Paragraph("<b>Potentiel du Modèle d'IA (M6)</b>", styles['h1']))

    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("Un modèle prédictif a été développé pour évaluer la probabilité de fidélité d'un client. Cet outil permet d'identifier et de cibler les clients à fort potentiel avant même qu'ils ne deviennent fidèles.", styles['BodyText']))

    story.append(Spacer(1, 1*cm))

    

    # Exemple de prédiction (votre exemple)

    input_data = pd.DataFrame([{'Age': 30, 'Total_Spent_Calc': 1200.50, 'Total_Orders': 3, 'Total_Quantity': 10, 'Recency': 20, 'Gender_Male': 1, 'Location_New York': 1}])

    input_aligned = input_data.reindex(columns=model_columns, fill_value=0)

    pred_proba = model.predict_proba(input_aligned)[0][1]

    

    story.append(Paragraph("<b>Exemple d'utilisation :</b>", styles['h2']))

    story.append(Paragraph(f"Un nouveau client homme de 30 ans à New York, après 3 commandes, a une probabilité de <b>{pred_proba:.2%}</b> de devenir un client fidèle. Il s'agit donc d'un profil à fort potentiel à cibler avec des actions de fidélisation.", styles['BodyText']))

    story.append(PageBreak())



    # --- 4. Plan d'Action Stratégique (M7) ---

    story.append(Paragraph("<b>Plan d'Action Stratégique (M7)</b>", styles['h1']))

    story.append(Spacer(1, 0.5*cm))

    

    # Le tableau de stratégie (votre tableau)

    table_data = [

        ["Segment", "Canal Recommandé", "Type de Contenu", "Objectif"],

        ["Fidèles", "Email + App", "Programme VIP, réductions exclusives", "Fidélisation"],

        ["Nouveaux clients", "Social Media", "Tutoriels, témoignages, -10% off", "Conversion"],

        ["Clients à réactiver", "SMS + Email", "“On vous a manqué” + -15%", "Réactivation"],

        ["Inactifs", "TV/Display/Retarget", "Branding + Offre flash", "Notoriété"]

    ]

    

    strategy_table = Table(table_data, colWidths=[3*cm, 4*cm, 6*cm, 3*cm])

    strategy_table.setStyle(TableStyle([

        ('BACKGROUND', (0, 0), (-1, 0), colors.navy),

        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),

        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),

        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),

        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),

        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),

        ('GRID', (0, 0), (-1, -1), 1, colors.black)

    ]))

    story.append(strategy_table)



    # Génération du PDF

    doc.build(story)

    print("✅ Le rapport stratégique final ('rapport_strategique_final.pdf') a été généré.")



# --- Appel de la fonction ---

# Assurez-vous que les variables 'client_features', 'model', et 'model_columns' sont disponibles

generer_rapport_coeur_de_projet(client_features, model, model_columns)







# Bouton Streamlit

if os.path.exists("rapport_strategique_final.pdf"):

    with open("rapport_strategique_final.pdf", "rb") as f:

        st.download_button(

            label="📄 Télécharger le Rapport Final",

            data=f,

            file_name="rapport_strategique_final.pdf",

            mime="application/pdf"

        )

else:

    st.warning("⚠ Rapport non trouvé. Génère-le d'abord.")
