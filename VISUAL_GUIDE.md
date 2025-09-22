# 🖼️ Visual Guide: Menstrual Health Analysis Dashboard

## 📋 What You'll See When You Install and Run the App

This guide shows you exactly what to expect when you run the Streamlit dashboard, with detailed descriptions of each section and visualization.

## 🚀 Installation Process

### Step 1: Running the Application

#### Windows Users (Easiest)
```
🪟 Double-click: run_analysis.bat
   ↓
📦 Installing packages...
   ↓
🌐 Opening browser...
   ↓
📊 Dashboard loads at http://localhost:8501
```

#### All Platforms
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Step 2: Browser Interface
```
🌐 Browser opens automatically
📱 URL: http://localhost:8501
🎨 Clean, professional interface
📊 5 main tabs for navigation
```

## 📊 Dashboard Interface Overview

### Main Navigation
```
┌─────────────────────────────────────────────────────────────┐
│ 🩸 Menstrual Health Analysis Dashboard                      │
├─────────────────────────────────────────────────────────────┤
│ 📈 Overview  🤖 Logistic Regression  📊 Symptom Analysis    │
│ 🔍 Interactive Explorer  📋 Reports                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    [Dashboard Content]                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📈 Tab 1: Overview - Dataset Summary

### What You'll See:
```
┌─────────────────────────────────────────────────────────────┐
│ 📈 Dataset Overview                                         │
├─────────────────────────────────────────────────────────────┤
│ 📊 Key Metrics (4 columns)                                 │
│ ┌─────────┬─────────┬─────────┬─────────┐                  │
│ │Total    │Heavy    │Symptoms │Data     │                  │
│ │Participants│Periods│Analyzed│Quality  │                  │
│ │271      │30 (11.1%)│19     │96.9%    │                  │
│ └─────────┴─────────┴─────────┴─────────┘                  │
│                                                             │
│ 🥧 Period Heaviness Distribution                            │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  🔵 Non-Heavy: 88.9% (241 participants)                │ │
│ │  🩸 Heavy: 11.1% (30 participants)                     │ │
│ │                                                         │ │
│ │  📊 Explanation:                                        │ │
│ │  • Pie chart shows proportion of period types          │ │
│ │  • Class imbalance affects model performance          │ │
│ │  • Heavy periods represent 11.1% of dataset           │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 📊 Age Distribution                                         │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  📈 Histogram showing participant ages                  │ │
│ │  • X-axis: Age in years                                │ │
│ │  • Y-axis: Number of participants                      │ │
│ │  • Peak around 19-20 years (college-aged)             │ │
│ │                                                         │ │
│ │  📊 Explanation:                                        │ │
│ │  • Shows age demographics of study population          │ │
│ │  • Age can influence menstrual patterns                │ │
│ │  • Helps in generalizing findings                      │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🤖 Tab 2: Logistic Regression - Model Analysis

### What You'll See:
```
┌─────────────────────────────────────────────────────────────┐
│ 🤖 Logistic Regression Analysis                             │
├─────────────────────────────────────────────────────────────┤
│ 📊 Model Performance Metrics (3 columns)                   │
│ ┌─────────────┬─────────────┬─────────────┐                │
│ │Model        │ROC AUC      │Training     │                │
│ │Accuracy     │Score        │Samples      │                │
│ │85.4%        │0.450        │189          │                │
│ └─────────────┴─────────────┴─────────────┘                │
│                                                             │
│ 📈 ROC Curve                                                │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  📊 Interactive plot showing model performance          │ │
│ │  • Orange line: Our model (AUC = 0.450)               │ │
│ │  • Blue dashed line: Random classifier baseline        │ │
│ │  • X-axis: False Positive Rate                         │ │
│ │  • Y-axis: True Positive Rate                          │ │
│ │                                                         │ │
│ │  📊 Explanation:                                        │ │
│ │  • ROC shows trade-off between sensitivity/specificity │ │
│ │  • AUC = 1.0: Perfect classifier                       │ │
│ │  • AUC = 0.5: Random classifier (no predictive power)  │ │
│ │  • AUC < 0.5: Worse than random                        │ │
│ │  • Current AUC = 0.450: Poor predictive ability        │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 🎯 Feature Importance                                       │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  📊 Horizontal bar chart (top 15 features)             │ │
│ │  • Red bars: Increase heavy period risk                │ │
│ │  • Blue bars: Decrease heavy period risk               │ │
│ │  • Larger bars = stronger predictive power             │ │
│ │                                                         │ │
│ │  📊 Explanation:                                        │ │
│ │  • Coefficients show how symptoms affect probability   │ │
│ │  • Positive = increases risk, Negative = decreases     │ │
│ │  • Magnitude indicates strength of prediction          │ │
│ │  • Odds ratio: change in odds per 1-unit increase      │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 📋 Detailed Feature Analysis Table                         │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │Feature | Coefficient | Odds Ratio | Direction | Effect  │ │
│ │Sleep   | 0.570      | 1.767      | Increases | Strong  │ │
│ │Weight  | 0.542      | 1.720      | Increases | Strong  │ │
│ │Mood    | 0.270      | 1.311      | Increases | Moderate│ │
│ │...     | ...        | ...        | ...       | ...     │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Tab 3: Symptom Analysis - Interactive Exploration

### What You'll See:
```
┌─────────────────────────────────────────────────────────────┐
│ 📊 Symptom Severity Analysis                                │
├─────────────────────────────────────────────────────────────┤
│ 🎛️ Symptom Selection                                        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Multi-select dropdown with 19 symptoms:                │ │
│ │ ☑️ Difficulty in Sleeping                              │ │
│ │ ☑️ Weight Gain                                         │ │
│ │ ☑️ Mood Swings                                         │ │
│ │ ☑️ Depression                                          │ │
│ │ ☑️ Restlessness                                        │ │
│ │ ☐ [Other symptoms...]                                  │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 📊 Symptom Severity by Period Type                         │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  📈 Box plots for each selected symptom                │ │
│ │  • Red boxes: Non-heavy periods                        │ │
│ │  • Blue boxes: Heavy periods                           │ │
│ │  • Middle line: Median severity score                  │ │
│ │  • Box edges: 25th and 75th percentiles               │ │
│ │  • Whiskers: Extend to 1.5× IQR or extremes           │ │
│ │  • Dots: Outliers beyond whiskers                      │ │
│ │                                                         │ │
│ │  📊 Explanation:                                        │ │
│ │  • Higher boxes indicate more severe symptoms          │ │
│ │  • Differences between colors show period type effects │ │
│ │  • Statistical significance shown with p-values       │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 📊 Statistical Analysis Table                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │Symptom | T-Stat | T-P-Value | Correlation | Corr-P     │ │
│ │Sleep   | 2.45   | 0.016     | 0.144       | 0.018      │ │
│ │Weight  | 2.12   | 0.035     | 0.089       | 0.156      │ │
│ │Mood    | 1.98   | 0.049     | 0.120       | 0.048      │ │
│ │...     | ...    | ...       | ...         | ...        │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🔍 Tab 4: Interactive Explorer - Demographics & Correlations

### What You'll See:
```
┌─────────────────────────────────────────────────────────────┐
│ 🔍 Interactive Data Explorer                                │
├─────────────────────────────────────────────────────────────┤
│ 👥 Demographics Analysis (2x2 grid)                        │
│ ┌─────────────────────┬─────────────────────────────────┐   │
│ │📊 Age vs Heavy      │📊 Weight vs Heavy               │   │
│ │Periods              │Periods                          │   │
│ │┌─────────────────┐  │┌─────────────────────────────┐  │   │
│ ││ Box plot showing│  ││ Box plot showing weight     │  │   │
│ ││ age distribution│  ││ distribution by period type │  │   │
│ ││ by period type  │  ││                             │  │   │
│ ││ • Red: Non-heavy│  ││ • Red: Non-heavy periods    │  │   │
│ ││ • Blue: Heavy   │  ││ • Blue: Heavy periods       │  │   │
│ │└─────────────────┘  │└─────────────────────────────┘  │   │
│ └─────────────────────┴─────────────────────────────────┘   │
│                                                             │
│ 🔗 Symptom Correlation Matrix                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  🎛️ Symptom Selection (multi-select):                  │ │
│ │  ☑️ Difficulty in Sleeping  ☑️ Weight Gain            │ │
│ │  ☑️ Mood Swings            ☑️ Depression              │ │
│ │  ☑️ Restlessness           ☐ [Other symptoms...]      │ │
│ │                                                         │ │
│ │  📊 Correlation Heatmap                                 │ │
│ │  ┌─────────────────────────────────────────────────┐   │ │
│ │  │ Color-coded matrix showing correlations         │   │ │
│ │  │ • Red: Positive correlation                     │   │ │
│ │  │ • Blue: Negative correlation                    │   │ │
│ │  │ • White: No correlation                         │   │ │
│ │  │ • Numbers: Correlation coefficients             │   │ │
│ │  │                                                 │   │ │
│ │  │ 📊 Explanation:                                 │   │ │
│ │  │ • Shows relationships between symptoms          │   │ │
│ │  │ • Stronger colors = stronger correlations      │   │ │
│ │  │ • Diagonal = perfect self-correlation (1.0)    │   │ │
│ │  └─────────────────────────────────────────────────┘   │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Tab 5: Reports - Comprehensive Summary

### What You'll See:
```
┌─────────────────────────────────────────────────────────────┐
│ 📋 Analysis Reports                                         │
├─────────────────────────────────────────────────────────────┤
│ 📊 Logistic Regression Summary (2 columns)                 │
│ ┌─────────────────────┬─────────────────────────────────┐   │
│ │📈 Model Performance │🏥 Clinical Insights            │   │
│ │• Accuracy: 85.4%    │• Heavy periods: 11.1% of       │   │
│ │• ROC AUC: 0.450     │  participants                  │   │
│ │• Dataset: 271       │• Most predictive symptoms      │   │
│ │• Features: 19       │  identified                    │   │
│ │                     │• Statistical significance      │   │
│ │                     │  tested                        │   │
│ │                     │• Risk factors quantified       │   │
│ └─────────────────────┴─────────────────────────────────┘   │
│                                                             │
│ 🎯 Top Risk Factors for Heavy Periods                       │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 1. Difficulty in Sleeping: increases risk (OR = 1.767) │ │
│ │ 2. Weight Gain: increases risk (OR = 1.720)            │ │
│ │ 3. Mood Swings: increases risk (OR = 1.311)            │ │
│ │ 4. Depression: increases risk (OR = 1.230)             │ │
│ │ 5. Restlessness: increases risk (OR = 1.230)           │ │
│ │ 6. Abdominal Bloating: increases risk (OR = 1.204)     │ │
│ │ 7. Tension: increases risk (OR = 1.156)                │ │
│ │ 8. Headache: increases risk (OR = 1.096)               │ │
│ │ 9. Fatigue: increases risk (OR = 1.089)                │ │
│ │10. Forgetfulness: increases risk (OR = 1.083)          │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 📥 Download Analysis Report                                 │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │  📄 Comprehensive report with:                          │ │
│ │  • Executive summary                                    │ │
│ │  • Key findings                                         │ │
│ │  • Top risk factors                                     │ │
│ │  • Clinical interpretation                              │ │
│ │  • Recommendations                                      │ │
│ │                                                         │ │
│ │  [📥 Download Report] button                            │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🎨 Visual Design Elements

### Color Scheme
```
🩸 Primary: #e91e63 (Pink - Menstrual health theme)
🔵 Secondary: #4ECDC4 (Teal - Data visualization)
🟢 Success: #96CEB4 (Green - Positive indicators)
🟡 Warning: #FFEAA7 (Yellow - Caution)
🔴 Danger: #FF6B6B (Red - Risk factors)
```

### Interactive Elements
```
🖱️ Hover Effects: Tooltips with additional information
🖱️ Click Interactions: Dynamic updates and filtering
📱 Responsive Design: Adapts to different screen sizes
🎨 Smooth Animations: Professional transitions
```

### Typography
```
📝 Headers: Bold, clear hierarchy
📝 Body Text: Readable font sizes
📝 Code: Monospace for technical content
📝 Emojis: Visual indicators for different sections
```

## 🚀 Getting Started Checklist

### Before You Start:
- [ ] Python 3.8+ installed
- [ ] Internet connection for package installation
- [ ] Data file (`DATA SHEET.xlsx`) in the `data/` folder
- [ ] 4GB+ RAM available
- [ ] Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation Steps:
- [ ] Download/clone the project
- [ ] Navigate to project directory
- [ ] Run `run_analysis.bat` (Windows) or `streamlit run streamlit_app.py`
- [ ] Wait for packages to install
- [ ] Browser opens automatically
- [ ] Dashboard loads at `http://localhost:8501`

### First-Time Usage:
- [ ] Explore the Overview tab to understand the dataset
- [ ] Check the Logistic Regression tab for model performance
- [ ] Try the Symptom Analysis tab with different symptom selections
- [ ] Use the Interactive Explorer for demographic analysis
- [ ] Download reports from the Reports tab

## 🆘 Troubleshooting

### Common Issues:
```
❌ "Python not found"
✅ Solution: Install Python 3.8+ and add to PATH

❌ "Data file not found"
✅ Solution: Ensure DATA SHEET.xlsx is in data/ folder

❌ "Port 8501 already in use"
✅ Solution: Close other Streamlit apps or use different port

❌ "Packages failed to install"
✅ Solution: Check internet connection and try again
```

### Performance Tips:
```
⚡ Close unused browser tabs
⚡ Restart the application if it becomes slow
⚡ Use the interactive features to filter data
⚡ Select specific symptoms for faster analysis
```

---

*This visual guide shows you exactly what to expect when you install and run the Menstrual Health Analysis Dashboard. The interface is designed to be intuitive and educational, with detailed explanations for every visualization.*
