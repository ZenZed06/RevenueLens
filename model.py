import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import scipy.stats as stats
import pylab
import time

# ============================
# Page Config & Custom UI
# ============================
st.set_page_config(
    page_title="Business Stats Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide default Streamlit menu, header, footer
hide_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# Custom Page Colors
custom_style = """
<style>
/* Page background */
body {
    background-color: #f0f4f8;
}
/* Title style */
h1 {
    color: #0f4c81;
    font-family: 'Segoe UI', sans-serif;
}
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #e1ecf4;
}
/* Sidebar header text */
[data-testid="stSidebar"] .css-1d391kg {
    color: #0f4c81;
    font-weight: bold;
}
/* Buttons color */
.stButton>button {
    background-color: #0f4c81;
    color: white;
    border-radius: 5px;
}
.stButton>button:hover {
    background-color: #1464a0;
}
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# ============================
# Page Title & Catchy Description (Centered)
# ============================
st.markdown("""
<div style="
    background: radial-gradient(circle at top left, #1f2a44, #0f1624);
    color: #ffffff;
    padding: 40px 30px;
    border-radius: 15px;
    font-family: 'Segoe UI', sans-serif;
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    text-align: center;
">
    <h1 style='margin:0; font-size:85px; color:#ffcc00;'> RevenueLens</h1>
    <p style='margin-top:12px; font-size:20px; line-height:1.6; color:#f0e68c;'>
        Share your business records, instantly visualize key trends, and uncover hidden opportunities to boost revenue. RevenueLens turns your data into actionable insights for smarter, faster growth.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================
# File Uploader
# ============================
st.markdown("<h3 style='color:#ffcc00; font-weight:bold;'>Upload Your Business Records</h3>", unsafe_allow_html=True)
file_uploaded = st.file_uploader("", type="csv")
if file_uploaded:
    try:
        df = pd.read_csv(file_uploaded)
        st.session_state['df'] = df
        st.markdown("<h3 style='color:#ffcc00; font-weight:bold;'>File Preview</h3>", unsafe_allow_html=True)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error Loading File: {e}")

# ============================
# Track Button Clicks
# ============================
if "show_stats" not in st.session_state:
    st.session_state["show_stats"] = False
if "show_plot" not in st.session_state:
    st.session_state["show_plot"] = False

# ============================
# Show File Stats
# ============================
if st.button("Show File Stats"):
    if 'df' in st.session_state:
        st.session_state['show_stats'] = True
    else:
        st.warning("Please upload a CSV file first!")

if st.session_state['show_stats'] and 'df' in st.session_state:
    st.markdown("<h3 style='color:#ffcc00; font-weight:bold;'>📊 Statistical Summary</h3>", unsafe_allow_html=True)
    st.write(st.session_state['df'].describe())

# ============================
# Visualize Features (Pairplot)
# ============================
if st.button("Visualize Features"):
    if 'df' in st.session_state:
        st.session_state['show_plot'] = True
    else:
        st.warning("Please upload a CSV file first!")

if st.session_state['show_plot'] and 'df' in st.session_state:
    st.markdown("<h3 style='color:#ffcc00; font-weight:bold;'>📈 Feature Relationships (Pairplot)</h3>",
                unsafe_allow_html=True)
    numeric_cols = st.session_state['df'].select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns to create a pairplot.")
    else:
        with st.spinner("Generating Pairplot..."):
            fig = sns.pairplot(st.session_state['df'][numeric_cols], kind="scatter", plot_kws={'alpha': 0.4})
            st.success("Visualization generated successfully.")
            st.pyplot(fig.fig)
            plt.close(fig.fig)

# ============================
# Train Model
# ============================
if st.button("Train Model"):
    if 'df' not in st.session_state:
        st.warning("Please upload a CSV file first!")
    else:
        df = st.session_state['df']
        required_cols = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership',
                         'Yearly Amount Spent']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns for model training: {missing_cols}")
        else:
            X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
            y = df['Yearly Amount Spent']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = LinearRegression()
            with st.spinner("Training Model... Please wait"):
                time.sleep(0.5)
                model.fit(X_train, y_train)

            # Store in session state
            st.session_state['model'] = model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test

            # Model Coefficients
            st.success("Model trained successfully!")
            st.markdown("<h3 style='color:#ffcc00; font-weight:bold;'>🤖 Model Training Results </h3>",
                        unsafe_allow_html=True)
            cdf = pd.DataFrame(model.coef_, X.columns, columns=['Coefficients'])
            st.write(cdf)
            strongest_feature = cdf['Coefficients'].idxmax()
            st.markdown(
                f"<h3 style='color:#ffcc00; font-weight:bold;'>Feature with strongest positive impact: {strongest_feature} ({cdf['Coefficients'].max():.2f})</h3>",
                unsafe_allow_html=True)

            # Feature importance bar chart
            st.markdown("<h3 style='color:#ffcc00; font-weight:bold;'>🎯 Feature Importance</h3>",
                        unsafe_allow_html=True)
            coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
            coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis', ax=ax)
            st.pyplot(fig)
            plt.close(fig)

            # Smart suggestions
            st.markdown(
                "<h3 style='color:#ffcc00; font-weight:bold;'>💡 Smart Suggestions to Improve Yearly Amount Spent</h3>",
                unsafe_allow_html=True)
            for feature, coef in zip(X.columns, model.coef_):
                if coef > 0:
                    impact = "high" if coef > 30 else "moderate"
                    st.write(
                        f"Increase **{feature}**. It has a {impact} positive impact ({coef:.2f}) on Yearly Amount Spent.")
                elif coef < 0:
                    st.write(f"Decrease **{feature}**. It negatively affects Yearly Amount Spent ({coef:.2f}).")

            # Model Performance Metrics
            predictions = model.predict(X_test)
            st.markdown("<h3 style='color:#ffcc00; font-weight:bold;'>⚙️ Model Performance Metrics</h3>",
                        unsafe_allow_html=True)

            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)

            st.write(
                f"Mean Absolute Error (MAE): {mae:.2f}  — lower is better; closer to 0 means predictions are closer to actual values")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}  — lower is better; penalizes larger errors more heavily")
            st.write(
                f"Root Mean Squared Error (RMSE): {rmse:.2f}  — lower is better; similar to MAE but more sensitive to outliers")
            st.write(
                f"R² Score: {r2:.4f}  — higher is better; closer to 1 means model explains more variance in the data")

            # Residual Analysis
            residuals = y_test - predictions
            st.markdown("<h3 style='color:#ffcc00; font-weight:bold;'>📉 Residual Analysis</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h3 style='color:#ffcc00; font-weight:bold;'>Histogram of Residuals</h3>",
                            unsafe_allow_html=True)
                fig2, ax = plt.subplots()
                sns.histplot(residuals, bins=20, kde=True, ax=ax)
                st.pyplot(fig2)
                plt.close(fig2)
            with col2:
                st.markdown("<h3 style='color:#ffcc00; font-weight:bold;'>Q-Q Plot</h3>", unsafe_allow_html=True)
                fig3 = plt.figure()
                stats.probplot(residuals, dist="norm", plot=pylab)
                st.pyplot(fig3)
                plt.close(fig3)
