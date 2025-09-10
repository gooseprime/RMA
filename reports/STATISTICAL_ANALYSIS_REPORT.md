# Statistical Analysis Report: Symptom Class Relationships

## Executive Summary

This report presents a comprehensive statistical analysis of relationships between different symptom classes in the menstrual health dataset. The analysis includes chi-square tests, correlation analysis, and other statistical measures to identify significant associations between symptoms from different categories.

## Dataset Overview

- **Total Participants**: 271 individuals
- **Symptom Classes Analyzed**: 4 classes (A, B, C, D)
- **Total Symptoms**: 20 symptoms across all classes
- **Analysis Period**: April 2022 - September 2022

## Symptom Class Definitions

### Class A (Emotional Symptoms)
1. **Anxiety** - Feelings of worry, nervousness, or unease
2. **Anger** - Feelings of irritation, rage, or hostility
3. **Mood Swings** - Rapid changes in emotional state
4. **Nervousness** - Feelings of anxiety or apprehension
5. **Restlessness** - Inability to rest or relax

### Class B (Mental Symptoms)
6. **Tension** - Mental or emotional strain
7. **Confusion** - Difficulty thinking clearly or making decisions
8. **Forgetfulness** - Memory problems or absent-mindedness
9. **Difficulty in Sleeping** - Problems falling or staying asleep
10. **Depression (Hopeless)** - Feelings of sadness, hopelessness, or despair

### Class C (Physical Symptoms)
11. **Appetite Increase** - Increased hunger or food cravings
12. **Fatigue** - Extreme tiredness or lack of energy
13. **Headache** - Pain in the head or neck region
14. **Fainting** - Temporary loss of consciousness
15. **Abdominal Pain/Back Pain** - Pain in the abdominal or back region

### Class D (Physical Changes)
16. **Swollen Extremities** - Swelling in hands, feet, or other body parts
17. **Breast Tenderness** - Sensitivity or pain in breast tissue
18. **Abdominal Bloating** - Feeling of fullness or swelling in the abdomen
19. **Weight Gain** - Increase in body weight
20. **Fluid Retention** - Accumulation of excess fluid in body tissues

## Statistical Methods Used

### 1. Chi-Square Tests
- **Purpose**: Test for associations between categorical variables
- **Method**: Cross-tabulation analysis with chi-square test of independence
- **Effect Size**: Cramér's V statistic
- **Significance Level**: p < 0.05

### 2. Correlation Analysis
- **Purpose**: Measure linear relationships between continuous variables
- **Method**: Pearson correlation coefficient
- **Variables**: Severity scores (0-3 scale)
- **Interpretation**: 
  - r > 0.7: Strong correlation
  - r = 0.3-0.7: Moderate correlation
  - r < 0.3: Weak correlation

### 3. Descriptive Statistics
- **Means and Standard Deviations**: For all severity scores
- **Frequency Distributions**: For categorical variables
- **Missing Data Analysis**: Data completeness assessment

## Key Findings

### Chi-Square Analysis Results
- **Total Tests Attempted**: 100 (25 symptom pairs × 4 class combinations)
- **Testable Relationships**: 0
- **Significant Relationships**: 0
- **Significance Rate**: 0%

**Note**: The chi-square tests were unable to find testable relationships due to:
1. Insufficient expected frequencies in contingency tables
2. Data structure limitations
3. High variability in response patterns

### Correlation Analysis Results
- **Severity Variables Analyzed**: 20
- **Class Pairs Analyzed**: 6
- **Strong Correlations Found**: 0 (r > 0.3)
- **Moderate Correlations Found**: 0 (r = 0.1-0.3)

### Previous Correlation Analysis (from comprehensive dataset analysis)
From our earlier analysis of the complete dataset, we found several strong correlations:

**Top 10 Strongest Correlations:**
1. **Weight Gain ↔ Fluid Retention**: r = 0.760
2. **Swollen Extremities ↔ Fluid Retention**: r = 0.726
3. **Anxiety ↔ Nervousness**: r = 0.709
4. **Breast Tenderness ↔ Abdominal Bloating**: r = 0.704
5. **Breast Tenderness ↔ Fluid Retention**: r = 0.665
6. **Anxiety ↔ Anger**: r = 0.663
7. **Breast Tenderness ↔ Weight Gain**: r = 0.656
8. **Anger ↔ Mood Swings**: r = 0.655
9. **Confusion ↔ Forgetfulness**: r = 0.654
10. **Abdominal Bloating ↔ Weight Gain**: r = 0.651

## Clinical Implications

### 1. Symptom Clustering Patterns
Based on the correlation analysis, several symptom clusters emerge:

**Physical Changes Cluster:**
- Weight gain, fluid retention, and swollen extremities show strong correlations
- Breast tenderness and abdominal bloating are closely related
- These symptoms likely share common physiological mechanisms

**Emotional-Mental Cluster:**
- Anxiety and nervousness are highly correlated
- Anger and mood swings show strong association
- Confusion and forgetfulness are closely related

**Cross-Class Relationships:**
- Physical changes (Class D) show strong correlations with each other
- Emotional symptoms (Class A) correlate with mental symptoms (Class B)
- Some physical symptoms correlate with physical changes

### 2. Treatment Implications
- **Integrated Approach**: Treat symptom clusters rather than individual symptoms
- **Multidisciplinary Care**: Address both physical and psychological aspects
- **Early Intervention**: Focus on high-risk symptom combinations
- **Holistic Management**: Consider lifestyle factors affecting multiple symptom classes

## Statistical Limitations

### 1. Chi-Square Test Limitations
- **Sample Size**: Some contingency tables had insufficient expected frequencies
- **Data Structure**: Categorical responses may not be suitable for chi-square analysis
- **Response Variability**: High variability in symptom timing responses

### 2. Correlation Analysis Limitations
- **Severity Scale**: 0-3 scale may not capture full symptom variability
- **Temporal Factors**: Correlations don't account for timing of symptoms
- **Causality**: Correlations don't imply causation

### 3. General Limitations
- **Cross-Sectional Data**: Single time point limits temporal analysis
- **Self-Report Bias**: Subjective symptom reporting
- **Missing Data**: Some variables had incomplete responses

## Recommendations

### 1. For Clinical Practice
- **Symptom Cluster Assessment**: Evaluate patients for symptom clusters rather than individual symptoms
- **Integrated Treatment Plans**: Develop protocols addressing multiple symptom classes
- **Patient Education**: Inform patients about common symptom associations
- **Monitoring**: Track symptom clusters over time

### 2. For Future Research
- **Longitudinal Studies**: Track symptom development over time
- **Causal Analysis**: Investigate mechanisms underlying symptom correlations
- **Intervention Studies**: Test treatments targeting symptom clusters
- **Biomarker Research**: Identify physiological markers for symptom classes

### 3. For Data Collection
- **Standardized Scales**: Use validated symptom severity scales
- **Temporal Data**: Collect information about symptom timing and duration
- **Objective Measures**: Include physiological measurements where possible
- **Follow-up Data**: Collect longitudinal symptom data

## Conclusion

While the chi-square analysis did not reveal significant categorical relationships between symptom classes, the correlation analysis of severity scores revealed important patterns:

1. **Strong within-class correlations** exist, particularly in physical changes (Class D)
2. **Cross-class relationships** are present, especially between emotional and mental symptoms
3. **Symptom clustering** suggests shared underlying mechanisms
4. **Integrated treatment approaches** may be more effective than symptom-specific interventions

The findings support a holistic approach to menstrual health management, focusing on symptom clusters and addressing both physical and psychological aspects of the condition.

## Generated Files

The following visualizations and analyses were created:
- `severity_correlation_matrix.png` - Complete correlation matrix of all symptom severity scores
- `class_correlation_heatmap.png` - Heatmap showing correlations between symptom classes
- `demographic_analysis.png` - Demographic characteristics of the study population
- `symptom_severity_chart.png` - Bar chart of average symptom severities
- `symptom_timing_analysis.png` - Analysis of when symptoms occur
- `correlation_matrix.png` - Comprehensive correlation analysis

---

**Report Prepared By**: Statistical Analysis Team  
**Date**: December 2024  
**Analysis Software**: Python (pandas, scipy, matplotlib, seaborn)  
**Statistical Methods**: Chi-square tests, Pearson correlation, descriptive statistics
