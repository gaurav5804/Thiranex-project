import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Customer Segmentation Dashboard', layout='wide')

st.title('Customer Segmentation Project')
st.write('Upload customer data and segment customers using K-Means clustering.')

uploaded_file = st.file_uploader('Upload CSV File', type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader('Dataset Preview')
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_features = st.multiselect('Select Features for Clustering', numeric_cols, default=numeric_cols[:2])

    if len(selected_features) >= 2:
        X = df[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        k = st.slider('Number of Clusters', 2, 10, 4)
        model = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = model.fit_predict(X_scaled)

        st.subheader('Clustered Data')
        st.dataframe(df.head())

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=selected_features[0], y=selected_features[1], hue='Cluster', palette='viridis', ax=ax)
        st.pyplot(fig)

        st.subheader('Cluster Summary')
        st.dataframe(df.groupby('Cluster')[selected_features].mean())
    else:
        st.warning('Please select at least 2 numeric features for clustering.')
else:
    st.info('Upload a CSV file to begin.')
