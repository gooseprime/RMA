# 🩸 Menstrual Health Analysis Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Research](https://img.shields.io/badge/Type-Research-purple.svg)](https://github.com)

## 📋 Overview

This project presents a comprehensive analysis of menstrual health data using advanced statistical methods and machine learning techniques. The study examines the relationship between menstrual period heaviness and symptom severity in a dataset of 271 participants.

## 🎯 Key Findings

### Model Performance
- **Accuracy**: 85.4%
- **ROC AUC**: 0.450
- **Dataset**: 271 participants, 19 symptom features
- **Class Distribution**: 11.1% heavy periods, 88.9% non-heavy periods

### Top Risk Factors
1. **Difficulty in Sleeping** (OR = 1.767) - Increases Risk
2. **Weight Gain** (OR = 1.720) - Increases Risk  
3. **Mood Swings** (OR = 1.311) - Increases Risk
4. **Depression** (OR = 1.230) - Increases Risk
5. **Restlessness** (OR = 1.230) - Increases Risk

### Protective Factors
1. **Swollen Extremities** (OR = 0.525) - Decreases Risk
2. **Nervousness** (OR = 0.599) - Decreases Risk
3. **Confusion** (OR = 0.607) - Decreases Risk

## 🖼️ Dashboard Preview

### What You'll See When You Run the App

The Streamlit dashboard provides an interactive interface with 5 main sections:

#### 📈 Overview Tab - Dataset Summary
- **Key Metrics**: Total participants (271), heavy periods (11.1%), symptoms analyzed (19)
- **Period Distribution**: Interactive pie chart showing heavy vs non-heavy periods
- **Age Distribution**: Histogram of participant ages with detailed explanations
- **Data Quality**: Completeness indicators (96.9% complete)

#### 🤖 Logistic Regression Tab - Model Analysis
- **Model Performance**: Accuracy (85.4%), ROC AUC (0.450), training samples
- **ROC Curve**: Interactive plot with detailed explanation of model performance
- **Feature Importance**: Horizontal bar chart showing which symptoms predict heavy periods
- **Detailed Analysis**: Table with coefficients, odds ratios, and statistical significance

#### 📊 Symptom Analysis Tab - Interactive Exploration
- **Symptom Selection**: Multi-select dropdown to choose specific symptoms
- **Box Plots**: Side-by-side comparison of symptom severity by period type
- **Statistical Tests**: T-tests and correlation analysis with p-values
- **Significance Testing**: Color-coded results showing significant associations

#### 🔍 Interactive Explorer Tab - Demographics & Correlations
- **Demographics Analysis**: Age and weight distributions by period type
- **Correlation Matrix**: Heatmap showing relationships between symptoms
- **Customizable Analysis**: Select symptoms for correlation analysis
- **Visual Exploration**: Interactive plots with hover tooltips

#### 📋 Reports Tab - Comprehensive Summary
- **Key Findings**: Model performance and clinical insights
- **Risk Factors**: Top predictors with odds ratios and significance
- **Clinical Recommendations**: Actionable insights for healthcare
- **Download Reports**: Export analysis results as text files

### Sample Visualizations

```
📊 Period Heaviness Distribution
┌─────────────────────────────────────┐
│  🩸 Heavy Periods: 11.1% (30)      │
│  🔵 Non-Heavy: 88.9% (241)         │
└─────────────────────────────────────┘

📈 ROC Curve Performance
┌─────────────────────────────────────┐
│  AUC = 0.450 (Below Random)        │
│  Model struggles with class         │
│  imbalance (11.1% vs 88.9%)        │
└─────────────────────────────────────┘

🎯 Top Risk Factors
┌─────────────────────────────────────┐
│  1. Difficulty in Sleeping (OR=1.77)│
│  2. Weight Gain (OR=1.72)          │
│  3. Mood Swings (OR=1.31)          │
│  4. Depression (OR=1.23)           │
│  5. Restlessness (OR=1.23)         │
└─────────────────────────────────────┘
```

## 🚀 Quick Start

### Windows Users (Easiest)
```bash
# Simply double-click the batch file
run_analysis.bat
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

## 📖 Usage Guide

The Streamlit dashboard provides five main sections:

1. **📈 Overview**: Dataset summary and demographics
2. **🤖 Logistic Regression**: Model performance and feature importance
3. **📊 Symptom Analysis**: Interactive symptom comparison
4. **🔍 Interactive Explorer**: Demographics and correlation analysis
5. **📋 Reports**: Comprehensive summaries and downloads

### 📖 Detailed Visual Guide
For a complete walkthrough with screenshots and detailed explanations of what you'll see, check out the **[Visual Guide](VISUAL_GUIDE.md)** - it shows exactly what each tab looks like and how to use all the features.

## 📁 Project Structure

```
menstrual-health-analysis/
├── 📊 data/DATA SHEET.xlsx          # Primary dataset
├── 📈 figures/                      # Generated visualizations
├── 📋 reports/                      # Analysis reports
├── 🐍 scripts/                      # Analysis scripts
├── 🌐 streamlit_app.py              # Main dashboard
├── 🪟 run_analysis.bat              # Windows batch file
├── 📦 requirements.txt              # Dependencies
├── 📖 README.md                     # This file
├── 🖼️ VISUAL_GUIDE.md               # Detailed visual walkthrough
├── 🔬 TECHNICAL_DOCUMENTATION.md    # Technical specifications
└── 📋 PROJECT_SUMMARY.md            # Complete project overview
```

## 🏗️ Technical Architecture

### Data Processing Pipeline

```mermaid
graph TD
    A[Raw Data] --> B[Data Loading]
    B --> C[Data Cleaning]
    C --> D[Feature Engineering]
    D --> E[Statistical Analysis]
    E --> F[Machine Learning]
    F --> G[Model Evaluation]
    G --> H[Visualization]
    H --> I[Report Generation]
```

### Technology Stack
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: plotly, matplotlib, seaborn
- **Web Interface**: Streamlit
- **Statistical Analysis**: scipy

## 📊 Clinical Implications

### Risk Assessment
- **Screening Tool**: Use identified risk factors for early identification
- **Risk Stratification**: Categorize patients based on symptom profiles
- **Preventive Care**: Target high-risk individuals for intervention

### Treatment Planning
- **Integrated Approach**: Address sleep, weight, and mental health factors
- **Personalized Medicine**: Tailor treatments based on symptom patterns
- **Multidisciplinary Care**: Involve sleep specialists, nutritionists, and mental health professionals

## ⚠️ Limitations

1. **Model Performance**: ROC-AUC of 0.450 indicates poor discriminative ability
2. **Class Imbalance**: 11.1% vs 88.9% distribution affects performance
3. **Cross-sectional Design**: Cannot establish causal relationships
4. **Sample Size**: Limited heavy period cases (n=30)

## 🚀 Future Work

1. **Data Enhancement**: Longitudinal studies, larger sample size
2. **Methodological Improvements**: Advanced ML models, ensemble methods
3. **Clinical Applications**: Intervention studies, real-world validation

## 🤝 Contributing

We welcome contributions from researchers, clinicians, and data scientists.

### Development Setup
```bash
git clone <repository-url>
cd menstrual-health-analysis
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License.

## 📞 Support

- **Documentation**: Check this README and the Streamlit guide
- **Issues**: Report bugs on GitHub Issues
- **Email**: Contact the research team for clinical questions

---

*Last updated: December 2024*