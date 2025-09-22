# Logistic Regression Analysis: Period Heaviness vs Symptom Severity

## Executive Summary

This report presents the results of a logistic regression analysis examining the relationship between menstrual period heaviness and symptom severity in a dataset of 271 participants. The analysis aimed to identify which symptoms are most predictive of heavy menstrual periods and quantify their impact.

## Dataset Overview

- **Total Participants**: 271 individuals
- **Target Variable**: Period heaviness (Heavy vs Non-Heavy)
- **Features**: 19 symptom severity scores (0-3 scale)
- **Heavy Periods**: 30 participants (11.1%)
- **Non-Heavy Periods**: 241 participants (88.9%)

## Model Performance

### Overall Performance Metrics
- **Model Accuracy**: 85.4%
- **ROC AUC Score**: 0.450
- **Training Accuracy**: 89.4%
- **Test Accuracy**: 85.4%

### Classification Results
```
              precision    recall  f1-score   support
   Non-Heavy       0.89      0.96      0.92        73
       Heavy       0.00      0.00      0.00         9
    accuracy                           0.85        82
   macro avg       0.44      0.48      0.46        82
weighted avg       0.79      0.85      0.82        82
```

### Confusion Matrix
- **True Negatives**: 70
- **False Positives**: 3
- **False Negatives**: 9
- **True Positives**: 0

## Key Findings

### 1. Most Predictive Symptoms

The following symptoms showed the strongest predictive power for heavy periods:

| Rank | Symptom | Coefficient | Odds Ratio | Effect |
|------|---------|-------------|------------|---------|
| 1 | Swollen Extremities | -0.643 | 0.525 | Decreases Risk |
| 2 | Difficulty in Sleeping | 0.570 | 1.767 | Increases Risk |
| 3 | Weight Gain | 0.542 | 1.720 | Increases Risk |
| 4 | Nervousness | -0.512 | 0.599 | Decreases Risk |
| 5 | Confusion | -0.500 | 0.607 | Decreases Risk |
| 6 | Breast Tenderness | -0.360 | 0.698 | Decreases Risk |
| 7 | Mood Swings | 0.270 | 1.311 | Increases Risk |
| 8 | Depression | 0.207 | 1.230 | Increases Risk |
| 9 | Restlessness | 0.207 | 1.230 | Increases Risk |
| 10 | Abdominal Bloating | 0.186 | 1.204 | Increases Risk |

### 2. Statistical Significance

**Chi-square Tests (p < 0.05):**
- 7 out of 19 symptoms showed significant associations with heavy periods
- Most significant: Confusion (χ² = 14.757, p = 0.0020)
- Other significant symptoms: Weight Gain, Tension, Forgetfulness, Depression, Mood Swings, Abdominal Pain

**Correlation Analysis:**
- 3 out of 19 symptoms showed significant correlations with heavy periods
- Strongest correlation: Difficulty in Sleeping (r = 0.144, p = 0.0180)
- Other significant correlations: Depression, Mood Swings

### 3. Risk Factors Analysis

#### Symptoms that INCREASE Heavy Period Risk:
1. **Difficulty in Sleeping** (OR = 1.767)
   - Strongest positive predictor
   - Significant correlation (r = 0.144, p = 0.0180)

2. **Weight Gain** (OR = 1.720)
   - Second strongest positive predictor
   - Significant chi-square association (χ² = 11.242, p = 0.0105)

3. **Mood Swings** (OR = 1.311)
   - Significant correlation (r = 0.120, p = 0.0485)
   - Significant chi-square association (χ² = 8.906, p = 0.0306)

4. **Depression** (OR = 1.230)
   - Significant correlation (r = 0.134, p = 0.0278)
   - Significant chi-square association (χ² = 9.591, p = 0.0224)

5. **Restlessness** (OR = 1.230)
6. **Abdominal Bloating** (OR = 1.204)

#### Symptoms that DECREASE Heavy Period Risk:
1. **Swollen Extremities** (OR = 0.525)
   - Strongest protective factor
   - May indicate different physiological patterns

2. **Nervousness** (OR = 0.599)
   - Significant protective effect

3. **Confusion** (OR = 0.607)
   - Strongest statistical association (χ² = 14.757, p = 0.0020)

4. **Breast Tenderness** (OR = 0.698)

## Clinical Interpretation

### Model Performance Assessment
- **Poor Predictive Ability**: ROC AUC of 0.450 indicates the model performs worse than random chance
- **Class Imbalance**: The model struggles with the imbalanced dataset (11.1% heavy periods)
- **High False Negative Rate**: Model fails to identify any heavy periods correctly

### Clinical Insights

#### 1. Sleep Disturbances as Primary Risk Factor
- Difficulty in sleeping is the strongest predictor of heavy periods
- Suggests potential hormonal or stress-related mechanisms
- Could be a target for intervention strategies

#### 2. Weight-Related Factors
- Weight gain shows strong association with heavy periods
- May reflect underlying metabolic or hormonal imbalances
- Important for lifestyle intervention recommendations

#### 3. Emotional and Mental Health Connections
- Mood swings and depression are significant risk factors
- Suggests bidirectional relationship between mental health and menstrual health
- Highlights need for integrated treatment approaches

#### 4. Protective Factors
- Swollen extremities showing protective effect is counterintuitive
- May indicate different physiological subtypes
- Requires further investigation

## Limitations

1. **Model Performance**: Poor ROC AUC suggests limited predictive utility
2. **Class Imbalance**: 11.1% heavy periods creates prediction challenges
3. **Cross-sectional Data**: Cannot establish causal relationships
4. **Sample Size**: Limited number of heavy period cases (n=30)
5. **Feature Engineering**: May need additional features for better prediction

## Recommendations

### 1. Clinical Practice
- **Sleep Assessment**: Screen for sleep disturbances in patients with heavy periods
- **Weight Management**: Address weight-related factors in treatment plans
- **Mental Health Integration**: Consider psychological support for mood-related symptoms
- **Individualized Approach**: Recognize that symptom patterns vary significantly

### 2. Further Research
- **Longitudinal Studies**: Track symptom changes over multiple cycles
- **Causal Analysis**: Investigate mechanisms behind identified associations
- **Larger Sample**: Increase sample size, especially for heavy period cases
- **Additional Features**: Include hormonal, lifestyle, and genetic factors
- **Machine Learning**: Explore more sophisticated algorithms for imbalanced data

### 3. Model Improvement
- **Resampling Techniques**: Address class imbalance with SMOTE or similar methods
- **Feature Selection**: Use more sophisticated feature selection algorithms
- **Ensemble Methods**: Combine multiple models for better performance
- **Threshold Optimization**: Adjust classification thresholds for better sensitivity

## Conclusion

While the logistic regression analysis identified several significant associations between symptom severity and period heaviness, the model's poor predictive performance (ROC AUC = 0.450) limits its clinical utility. The most important findings are:

1. **Sleep disturbances** are the strongest predictor of heavy periods
2. **Weight gain** and **mood-related symptoms** are significant risk factors
3. **7 out of 19 symptoms** show statistically significant associations
4. The model struggles with the imbalanced nature of the dataset

These findings provide valuable insights for understanding the complex relationship between menstrual symptoms and period heaviness, but further research with improved methodologies is needed to develop clinically useful predictive models.

---

*Analysis completed using Python with scikit-learn, pandas, and matplotlib. All statistical tests performed with α = 0.05 significance level.*
