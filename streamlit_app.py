import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Menstrual Health Analysis Dashboard",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e91e63;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e91e63;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e91e63;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_excel("data/DATA SHEET.xlsx", sheet_name=0)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def prepare_logistic_regression_data(df):
    """Prepare data for logistic regression analysis"""
    
    # Define symptom classes and severity mapping
    severity_mapping = {
        'ANXIETY': 'ANXIETY  SEVERITY             (0=NONE, 1=MILD, 2=MODERATE, 3=HIGH) [1. ANXIETY]',
        'ANGER': 'ANGER   SEVERITY             (0=NONE, 1=MILD, 2=MODERATE, 3=HIGH) [2. ANGER OR IRRITABILITY]',
        'MOOD SWINGS': 'MOOD SWINGS   SEVERITY             (0=NONE, 1=MILD, 2=MODERATE, 3=HIGH) [3. MOOD SWINGS]',
        'NERVOUSNESS': 'NERVOUSNESS SEVERITY             (0=NONE, 1=MILD, 2=MODERATE, 3=HIGH) [4. NEROUSNESS]',
        'RESTLESSNESS': 'RESTLESSNESS   SEVERITY             (0=NONE, 1=MILD, 2=MODERATE, 3=HIGH) [5. RESTLESSNESS]',
        'TENSION': 'TENSION   SEVERITY              (0=NONE, 1=MILD, 2=MODERATE, 3=HIGH) [6. TENSION]',
        'CONFUSION': 'CONFUSION    SEVERITY              (0=NONE, 1=MILD, 2=MODERATE, 3=HIGH) [7. CONFUSION]',
        'FORGETFULNESS': 'FORGETFULNESS    SEVERITY              (0=NONE, 1=MILD, 2=MODERATE, 3=HIGH) [8. FORGETFULNESS]',
        'DIFFICULTY IN SLEEPING': 'DIFFICULTY IN SLEEPING    SEVERITY              (0=NONE, 1=MILD, 2=MODERATE, 3=HIGH) [9. DIFFICULTY IN SLEEPING]',
        'DEPRESSION': 'DEPRESSION  SEVERITY              (0=NONE, 1=MILD, 2=MODERATE, 3=HIGH) [10. DEPRESSION(HOPELESS)]',
        'APPETITE INCREASE': 'APPETITE INCREASE  SEVERITY          (0=NONE, 1=MILD, 2=MODERATE, 3=SEVERE) [11. APPETITE INCREASE]',
        'FATIGUE': 'FATIGUE SEVERITY          (0=NONE, 1=MILD, 2=MODERATE, 3=SEVERE) [12. FATIGUE]',
        'HEADACHE': 'HEADACHE  SEVERITY          (0=NONE, 1=MILD, 2=MODERATE, 3=SEVERE) [13. HEADACHE]',
        'FAINTING': 'FAINTING SEVERITY          (0=NONE, 1=MILD, 2=MODERATE, 3=SEVERE) [14. FAINTING]',
        'ABDOMINAL PAIN': 'ABDOMINAL PAIN  SEVERITY          (0=NONE, 1=MILD, 2=MODERATE, 3=SEVERE) [15. ABDOMINAL PAIN/ BACK PAIN]',
        'SWOLLEN EXTREMITIES': ' SWOLLEN EXTREMITIE SEVERITY (0=NONE, 1=MILD, 2=MODERATE, 3=HIGH) [16. SWOLLEN EXTREMITIES]',
        'BREAST TENDERNESS': ' BREAST TENDERNESS   SEVERITY (0=NONE, 1=MILD, 2=MODERATE, 3=HIGH) [17. BREAST TENDERNESS]',
        'ABDOMINAL BLOATING': 'ABDOMINAL BLOATING   SEVERITY (0=NONE, 1=MILD, 2=MODERATE, 3=HIGH) [18. ABDOMINAL BLOATING]',
        'WEIGHT GAIN': 'WEIGHT GAIN SEVERITY (0=NONE, 1=MILD, 2=MODERATE, 3=HIGH) [19. WEIGHT GAIN]',
        'FLUID RETENTION': 'FLUID RETENTION   SEVERITY (0=NONE, 1=MILD, 2=MODERATE, 3=HIGH) [20. FLUID RETENTION]'
    }
    
    # Find period heaviness column
    period_heaviness_col = None
    for col in df.columns:
        if 'heavy' in col.lower() or 'flow' in col.lower():
            period_heaviness_col = col
            break
    
    if period_heaviness_col is None:
        return None, None, None, None
    
    # Create a copy of the dataframe to avoid modifying the original
    df_processed = df.copy()
    
    # Create binary target variable
    df_processed['is_heavy_period'] = (df_processed[period_heaviness_col] == 'Heavy').astype(int)
    
    # Prepare severity features
    severity_features = []
    feature_names = []
    
    for symptom_key, severity_col in severity_mapping.items():
        if severity_col in df_processed.columns:
            severity_features.append(df_processed[severity_col].fillna(0).values)
            feature_names.append(symptom_key)
    
    if not severity_features:
        return None, None, None, None
    
    # Create feature matrix
    X = np.column_stack(severity_features)
    y = df_processed['is_heavy_period'].values
    
    return X, y, feature_names, df_processed

def create_logistic_regression_model(X, y, feature_names):
    """Create and train logistic regression model"""
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit logistic regression model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test_scaled)
    y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = lr_model.score(X_test_scaled, y_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Feature importance
    coefficients = lr_model.coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients),
        'Odds_Ratio': np.exp(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    return {
        'model': lr_model,
        'scaler': scaler,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'feature_importance': feature_importance,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">🩸 Menstrual Health Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Prepare data for analysis
    X, y, feature_names, df_processed = prepare_logistic_regression_data(df)
    if X is None:
        st.error("Could not prepare data for analysis")
        st.stop()
    
    # Use the processed dataframe for all analysis
    df = df_processed
    
    # Sidebar
    st.sidebar.title("📊 Analysis Options")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🤖 Logistic Regression", 
        "📊 Symptom Analysis", 
        "🔍 Interactive Explorer",
        "📋 Reports"
    ])
    
    with tab1:
        st.header("📈 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Participants", len(df))
        
        with col2:
            heavy_periods = df['is_heavy_period'].sum()
            st.metric("Heavy Periods", f"{heavy_periods} ({heavy_periods/len(df)*100:.1f}%)")
        
        with col3:
            st.metric("Symptoms Analyzed", len(feature_names))
        
        with col4:
            st.metric("Data Completeness", f"{(1-df.isnull().sum().sum()/(len(df)*len(df.columns)))*100:.1f}%")
        
        # Period heaviness distribution
        st.subheader("Period Heaviness Distribution")
        period_dist = df['is_heavy_period'].value_counts()
        
        fig = px.pie(
            values=period_dist.values,
            names=['Non-Heavy', 'Heavy'],
            title="Distribution of Period Types",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation for Period Distribution
        st.markdown("""
        **📊 Period Heaviness Distribution Explanation:**
        - **Pie chart** shows the proportion of participants with different period types
        - **Red (Non-Heavy)**: Participants with light or moderate periods
        - **Blue (Heavy)**: Participants with heavy periods
        - **Class Imbalance**: The dataset has significantly more non-heavy periods than heavy periods
        - **Clinical Relevance**: This imbalance affects model performance and requires special consideration in analysis
        - **Sample Size**: Heavy periods represent {:.1f}% of the dataset, which may limit statistical power
        """.format(period_dist[1]/period_dist.sum()*100))
        
        # Age distribution
        if 'Age ' in df.columns:
            st.subheader("Age Distribution")
            fig = px.histogram(
                df, 
                x='Age ', 
                title="Age Distribution of Participants",
                nbins=20,
                color_discrete_sequence=['#e91e63']
            )
            fig.update_layout(xaxis_title="Age (years)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation for Age Distribution
            st.markdown("""
            **📊 Age Distribution Explanation:**
            - **Histogram** shows the frequency distribution of participant ages
            - **X-axis**: Age in years
            - **Y-axis**: Number of participants in each age group
            - **Population Characteristics**: Shows the age demographics of the study population
            - **Clinical Context**: Age can influence menstrual patterns and symptom severity
            - **Research Implications**: Understanding age distribution helps in generalizing findings
            """)
    
    with tab2:
        st.header("🤖 Logistic Regression Analysis")
        st.write("Predicting heavy periods based on symptom severity")
        
        # Create model
        model_results = create_logistic_regression_model(X, y, feature_names)
        
        # Model performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Accuracy", f"{model_results['accuracy']:.3f}")
        
        with col2:
            st.metric("ROC AUC Score", f"{model_results['roc_auc']:.3f}")
        
        with col3:
            st.metric("Training Samples", f"{len(X) - len(model_results['X_test'])}")
        
        # ROC Curve
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(model_results['y_test'], model_results['y_pred_proba'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {model_results["roc_auc"]:.3f})',
            line=dict(color='darkorange', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="ROC Curve: Heavy Period Prediction",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation for ROC Curve
        st.markdown("""
        **📊 ROC Curve Explanation:**
        - **ROC (Receiver Operating Characteristic) Curve** shows the trade-off between sensitivity (true positive rate) and specificity (1 - false positive rate)
        - **AUC (Area Under Curve)** measures overall model performance:
          - **AUC = 1.0**: Perfect classifier
          - **AUC = 0.5**: Random classifier (no predictive power)
          - **AUC < 0.5**: Worse than random
        - **Orange line**: Our model's performance
        - **Blue dashed line**: Random classifier baseline
        - **Current AUC = {:.3f}**: Indicates the model's predictive ability for heavy periods
        """.format(model_results['roc_auc']))
        
        # Feature Importance
        st.subheader("Feature Importance")
        top_features = model_results['feature_importance'].head(15)
        
        fig = px.bar(
            top_features,
            x='Coefficient',
            y='Feature',
            orientation='h',
            title="Top 15 Most Important Features",
            color='Coefficient',
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0
        )
        fig.update_layout(
            xaxis_title="Logistic Regression Coefficient",
            yaxis_title="Symptom",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation for Feature Importance
        st.markdown("""
        **📊 Feature Importance Explanation:**
        - **Logistic Regression Coefficients** show how each symptom affects the probability of having heavy periods
        - **Positive coefficients (Red)**: Symptoms that INCREASE the risk of heavy periods
        - **Negative coefficients (Blue)**: Symptoms that DECREASE the risk of heavy periods
        - **Magnitude**: Larger absolute values indicate stronger predictive power
        - **Odds Ratio**: For every 1-unit increase in symptom severity, the odds of heavy periods change by the coefficient value
        - **Top predictors**: The symptoms with the largest absolute coefficients are most important for prediction
        """)
        
        # Detailed feature importance table
        st.subheader("Detailed Feature Analysis")
        display_features = model_results['feature_importance'].copy()
        display_features['Direction'] = display_features['Coefficient'].apply(
            lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk'
        )
        display_features['Effect Size'] = display_features['Abs_Coefficient'].apply(
            lambda x: 'Strong' if x > 0.5 else 'Moderate' if x > 0.2 else 'Weak'
        )
        
        st.dataframe(
            display_features[['Feature', 'Coefficient', 'Odds_Ratio', 'Direction', 'Effect Size']],
            use_container_width=True
        )
    
    with tab3:
        st.header("📊 Symptom Severity Analysis")
        
        # Symptom selection
        selected_symptoms = st.multiselect(
            "Select symptoms to analyze:",
            feature_names,
            default=feature_names[:5]
        )
        
        if selected_symptoms:
            # Create symptom severity comparison
            st.subheader("Symptom Severity by Period Type")
            
            # Prepare data for visualization
            symptom_data = []
            for symptom in selected_symptoms:
                symptom_idx = feature_names.index(symptom)
                heavy_severity = X[y == 1, symptom_idx]
                non_heavy_severity = X[y == 0, symptom_idx]
                
                for severity in heavy_severity:
                    symptom_data.append({
                        'Symptom': symptom,
                        'Severity': severity,
                        'Period_Type': 'Heavy'
                    })
                
                for severity in non_heavy_severity:
                    symptom_data.append({
                        'Symptom': symptom,
                        'Severity': severity,
                        'Period_Type': 'Non-Heavy'
                    })
            
            symptom_df = pd.DataFrame(symptom_data)
            
            # Box plot
            fig = px.box(
                symptom_df,
                x='Symptom',
                y='Severity',
                color='Period_Type',
                title="Symptom Severity Distribution by Period Type",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            fig.update_layout(
                xaxis_title="Symptom",
                yaxis_title="Severity Score (0-3)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation for Box Plot
            st.markdown("""
            **📊 Symptom Severity Box Plot Explanation:**
            - **Box plots** show the distribution of symptom severity scores for each period type
            - **Red boxes**: Non-heavy periods
            - **Blue boxes**: Heavy periods
            - **Box components**:
              - **Middle line**: Median severity score
              - **Box edges**: 25th and 75th percentiles (interquartile range)
              - **Whiskers**: Extend to 1.5× IQR or data extremes
              - **Dots**: Outliers beyond whiskers
            - **Interpretation**: Higher boxes indicate more severe symptoms
            - **Comparison**: Differences between red and blue boxes show how symptoms vary by period type
            """)
            
            # Statistical tests
            st.subheader("Statistical Analysis")
            test_results = []
            
            for symptom in selected_symptoms:
                symptom_idx = feature_names.index(symptom)
                heavy_severity = X[y == 1, symptom_idx]
                non_heavy_severity = X[y == 0, symptom_idx]
                
                # T-test
                t_stat, p_val = stats.ttest_ind(heavy_severity, non_heavy_severity)
                
                # Correlation
                corr, corr_p = stats.pearsonr(X[:, symptom_idx], y)
                
                test_results.append({
                    'Symptom': symptom,
                    'T_Statistic': t_stat,
                    'T_P_Value': p_val,
                    'Correlation': corr,
                    'Corr_P_Value': corr_p,
                    'Significant_T': p_val < 0.05,
                    'Significant_Corr': corr_p < 0.05
                })
            
            test_df = pd.DataFrame(test_results)
            st.dataframe(test_df, use_container_width=True)
    
    with tab4:
        st.header("🔍 Interactive Data Explorer")
        
        # Demographics analysis
        st.subheader("Demographics Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age vs Heavy Periods
            if 'Age ' in df.columns:
                age_heavy = df[['Age ', 'is_heavy_period']].dropna()
                if len(age_heavy) > 0:
                    fig = px.box(
                        age_heavy,
                        x='is_heavy_period',
                        y='Age ',
                        title="Age Distribution by Period Type",
                        color='is_heavy_period',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
                    )
                    fig.update_layout(
                        xaxis_title="Period Type (0=Non-Heavy, 1=Heavy)",
                        yaxis_title="Age (years)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Weight vs Heavy Periods
            if 'Weight' in df.columns:
                weight_heavy = df[['Weight', 'is_heavy_period']].dropna()
                if len(weight_heavy) > 0:
                    fig = px.box(
                        weight_heavy,
                        x='is_heavy_period',
                        y='Weight',
                        title="Weight Distribution by Period Type",
                        color='is_heavy_period',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
                    )
                    fig.update_layout(
                        xaxis_title="Period Type (0=Non-Heavy, 1=Heavy)",
                        yaxis_title="Weight (kg)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Symptom Correlation Matrix")
        
        # Select symptoms for correlation
        corr_symptoms = st.multiselect(
            "Select symptoms for correlation analysis:",
            feature_names,
            default=feature_names[:10]
        )
        
        if len(corr_symptoms) > 1:
            # Create correlation matrix
            corr_indices = [feature_names.index(symptom) for symptom in corr_symptoms]
            corr_matrix = np.corrcoef(X[:, corr_indices].T)
            
            fig = px.imshow(
                corr_matrix,
                x=corr_symptoms,
                y=corr_symptoms,
                title="Symptom Severity Correlation Matrix",
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0,
                aspect='auto'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("📋 Analysis Reports")
        
        # Generate comprehensive report
        st.subheader("Logistic Regression Summary")
        
        model_results = create_logistic_regression_model(X, y, feature_names)
        
        # Key findings
        st.markdown("### Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Model Performance:**
            - Accuracy: {:.1f}%
            - ROC AUC: {:.3f}
            - Dataset Size: {} participants
            - Features: {} symptoms
            """.format(
                model_results['accuracy'] * 100,
                model_results['roc_auc'],
                len(df),
                len(feature_names)
            ))
        
        with col2:
            st.markdown("""
            **Clinical Insights:**
            - Heavy periods: {:.1f}% of participants
            - Most predictive symptoms identified
            - Statistical significance tested
            - Risk factors quantified
            """.format(
                df['is_heavy_period'].mean() * 100
            ))
        
        # Top risk factors
        st.subheader("Top Risk Factors for Heavy Periods")
        
        top_risk_factors = model_results['feature_importance'].head(10)
        
        for i, (_, row) in enumerate(top_risk_factors.iterrows(), 1):
            direction = "increases" if row['Coefficient'] > 0 else "decreases"
            st.markdown(f"**{i}.** {row['Feature']}: {direction} risk (OR = {row['Odds_Ratio']:.3f})")
        
        # Download report
        st.subheader("Download Analysis Report")
        
        report_text = f"""
# Menstrual Health Analysis Report

## Executive Summary
This analysis examined the relationship between menstrual period heaviness and symptom severity in {len(df)} participants.

## Key Findings
- **Model Accuracy**: {model_results['accuracy']:.1f}%
- **ROC AUC Score**: {model_results['roc_auc']:.3f}
- **Heavy Periods**: {df['is_heavy_period'].mean()*100:.1f}% of participants
- **Symptoms Analyzed**: {len(feature_names)}

## Top Risk Factors
"""
        
        for i, (_, row) in enumerate(model_results['feature_importance'].head(5).iterrows(), 1):
            direction = "increases" if row['Coefficient'] > 0 else "decreases"
            report_text += f"{i}. {row['Feature']}: {direction} risk (OR = {row['Odds_Ratio']:.3f})\n"
        
        report_text += f"""
## Clinical Interpretation
The logistic regression model {'shows good predictive ability' if model_results['roc_auc'] > 0.7 else 'shows moderate predictive ability' if model_results['roc_auc'] > 0.6 else 'shows poor predictive ability'} for heavy periods.

## Recommendations
1. Focus on the most predictive symptoms for early intervention
2. Consider integrated treatment approaches for symptom clusters
3. Further research needed to establish causal relationships
"""
        
        st.download_button(
            label="📥 Download Report",
            data=report_text,
            file_name="menstrual_health_analysis_report.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
