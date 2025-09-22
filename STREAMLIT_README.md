# 🩸 Menstrual Health Analysis Dashboard

## Overview

This interactive dashboard provides comprehensive analysis of menstrual health data, including logistic regression analysis to predict period heaviness based on symptom severity. The application is built with Streamlit and provides multiple visualization and analysis tools.

## Features

### 📈 Overview Tab
- Dataset summary statistics
- Period heaviness distribution
- Age distribution of participants
- Key demographic metrics

### 🤖 Logistic Regression Tab
- Interactive logistic regression model
- ROC curve visualization
- Feature importance analysis
- Model performance metrics
- Detailed coefficient analysis

### 📊 Symptom Analysis Tab
- Symptom severity comparison by period type
- Statistical significance testing
- Interactive symptom selection
- Box plots and distribution analysis

### 🔍 Interactive Explorer Tab
- Demographics analysis
- Correlation matrix visualization
- Age and weight distribution analysis
- Customizable symptom correlation analysis

### 📋 Reports Tab
- Comprehensive analysis summary
- Key findings and clinical insights
- Downloadable reports
- Risk factor analysis

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python run_streamlit.py
   ```
   
   Or manually:
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, navigate to the URL manually

## Data Requirements

The application expects the following data structure:
- Excel file: `data/DATA SHEET.xlsx`
- Sheet name: First sheet (index 0)
- Required columns:
  - Period heaviness information
  - Symptom severity scores (0-3 scale)
  - Demographic information (age, weight, height)

## Key Analysis Results

### Logistic Regression Findings

**Model Performance:**
- Accuracy: 85.4%
- ROC AUC: 0.450
- Dataset: 271 participants
- Features: 19 symptom severity scores

**Top Risk Factors for Heavy Periods:**
1. **Difficulty in Sleeping** (OR = 1.767) - Increases Risk
2. **Weight Gain** (OR = 1.720) - Increases Risk  
3. **Mood Swings** (OR = 1.311) - Increases Risk
4. **Depression** (OR = 1.230) - Increases Risk
5. **Restlessness** (OR = 1.230) - Increases Risk

**Protective Factors:**
1. **Swollen Extremities** (OR = 0.525) - Decreases Risk
2. **Nervousness** (OR = 0.599) - Decreases Risk
3. **Confusion** (OR = 0.607) - Decreases Risk

### Statistical Significance
- 7 out of 19 symptoms show significant associations with heavy periods
- 3 out of 19 symptoms show significant correlations
- Chi-square tests and correlation analysis performed

## Usage Guide

### 1. Overview Tab
- View dataset summary and key metrics
- Examine period heaviness distribution
- Analyze participant demographics

### 2. Logistic Regression Tab
- Review model performance metrics
- Examine ROC curve and feature importance
- Download detailed coefficient analysis

### 3. Symptom Analysis Tab
- Select specific symptoms for analysis
- Compare severity distributions by period type
- View statistical test results

### 4. Interactive Explorer Tab
- Explore demographic relationships
- Analyze symptom correlations
- Customize analysis parameters

### 5. Reports Tab
- Access comprehensive analysis summary
- Download detailed reports
- Review clinical insights and recommendations

## Technical Details

### Dependencies
- `streamlit>=1.28.0` - Web application framework
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning
- `plotly>=5.15.0` - Interactive visualizations
- `matplotlib>=3.6.0` - Static plotting
- `seaborn>=0.12.0` - Statistical visualization
- `scipy>=1.10.0` - Scientific computing

### File Structure
```
├── streamlit_app.py          # Main Streamlit application
├── run_streamlit.py          # Launch script
├── requirements.txt          # Python dependencies
├── data/
│   └── DATA SHEET.xlsx      # Dataset
├── scripts/
│   └── logistic_regression_analysis.py  # Standalone analysis
├── reports/
│   └── LOGISTIC_REGRESSION_REPORT.md    # Detailed report
└── STREAMLIT_README.md       # This file
```

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   streamlit run streamlit_app.py --server.port 8502
   ```

2. **Missing dependencies:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Data file not found:**
   - Ensure `data/DATA SHEET.xlsx` exists
   - Check file permissions

4. **Memory issues:**
   - Close other applications
   - Restart the application

### Performance Tips

- Use the interactive features to filter data
- Select specific symptoms for faster analysis
- Close unused browser tabs

## Clinical Interpretation

### Key Insights
1. **Sleep disturbances** are the strongest predictor of heavy periods
2. **Weight-related factors** show significant associations
3. **Mental health symptoms** are important risk factors
4. Model performance is limited by class imbalance

### Recommendations
1. Screen for sleep disturbances in clinical practice
2. Address weight management in treatment plans
3. Consider integrated mental health support
4. Further research needed for causal relationships

## Support

For technical issues or questions:
1. Check the troubleshooting section
2. Review the detailed report in `reports/LOGISTIC_REGRESSION_REPORT.md`
3. Examine the standalone analysis script in `scripts/`

## License

This project is for research and educational purposes. Please ensure compliance with data privacy regulations when using with real patient data.

---

*Last updated: December 2024*
