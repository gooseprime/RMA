# RMA: Menstrual Health Analysis and Yoga/Meditation Insights

Comprehensive, end-to-end analysis of menstrual health symptoms integrating statistics, unsupervised ML, and yoga/meditation effectiveness, with publication-ready figures and reports.

## Structure
```
.
├─ data/                 # Raw dataset(s)
│  └─ DATA SHEET.xlsx
├─ scripts/              # All analysis code
│  ├─ statistical_analysis.py
│  ├─ simple_chi_square.py
│  ├─ chi_square_analysis.py
│  ├─ final_chi_square.py
│  ├─ comprehensive_statistical_analysis.py
│  ├─ advanced_pattern_analysis.py
│  ├─ yoga_meditation_recommendations.py
│  ├─ analysis_no_plots.py
│  └─ severity_analysis.py
├─ figures/              # Generated plots and images
│  ├─ demographic_analysis.png
│  ├─ correlation_matrix.png
│  ├─ severity_correlation_matrix.png
│  ├─ symptom_dendrogram.png
│  ├─ pca_analysis.png
│  ├─ symptom_clusters_2d.png
│  ├─ symptom_severity_chart.png
│  ├─ symptom_timing_analysis.png
│  ├─ yoga_effectiveness.png
│  ├─ yoga_effectiveness_detailed.png
│  ├─ class_to_technique_heatmap.png
│  ├─ class_to_technique_bubble.png
│  ├─ technique_impact_top10.png
│  ├─ class_to_family_scaled_by_techniques.png
│  ├─ severity_distribution_by_class.png
│  ├─ mean_severity_by_symptom.png
│  ├─ severity_heatmap_by_class.png
│  ├─ severity_distribution_histograms.png
│  ├─ severity_boxplot_by_class.png
│  └─ severity_correlation_network.png
├─ reports/              # Markdown reports
│  ├─ DATASET_SUMMARY.md
│  ├─ STATISTICAL_ANALYSIS_REPORT.md
│  ├─ ADVANCED_ANALYSIS_REPORT.md
│  ├─ COMPREHENSIVE_REPORT.md
│  ├─ SEVERITY_ANALYSIS_REPORT.md
│  └─ FINAL_REPORT.md
└─ docs/                 # Instruments and docs
   └─ Questionnaire.pdf
```

## Quickstart

1) Install Python deps
```bash
python3 -m pip install -U pandas numpy scipy scikit-learn seaborn matplotlib openpyxl
```

2) Run core analyses
```bash
# Statistical relationships and correlations
python3 scripts/statistical_analysis.py
python3 scripts/comprehensive_statistical_analysis.py

# Advanced ML patterns & yoga effectiveness
python3 scripts/advanced_pattern_analysis.py
python3 scripts/yoga_meditation_recommendations.py

# Symptom severity deep-dive
python3 scripts/severity_analysis.py
```

Outputs are written to `figures/` and summarized in `reports/`.

## Highlights
- Correlation, chi-square (where valid), Cramér's V
- PCA, hierarchical clustering, K-Means
- Technique-level yoga/meditation impact analyses
- Publication-ready figures and PhD-level final report

## Authors
- Anvin P Shibu (CSE)

## License
Add your license of choice (e.g., MIT) here.
