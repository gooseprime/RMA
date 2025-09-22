import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

def logistic_regression_analysis(file_path):
    """
    Perform logistic regression analysis between period heaviness and symptom severity
    """
    print("=" * 80)
    print("LOGISTIC REGRESSION: PERIOD HEAVINESS vs SYMPTOM SEVERITY")
    print("=" * 80)
    
    # Read the dataset
    df = pd.read_excel(file_path, sheet_name=0)
    
    # Define symptom classes and their severity columns
    class_a_symptoms = [
        'CLASS  A [1. ANXEITY]',
        'CLASS  A [2. ANGER]', 
        'CLASS  A [3. MOOD SWINGS]',
        'CLASS  A [4. NERVOUSNESS]',
        'CLASS  A [5. RESTLESSNESS]'
    ]
    
    class_b_symptoms = [
        'CLASS B [6. TENSION]',
        'CLASS B [7. CONFUSION]',
        'CLASS B [8. FORGETFULNESS]',
        'CLASS B [9. DIFFICULTY IN SLEEPING]',
        'CLASS B [10. DEPRESSION(HOPELESS)]'
    ]
    
    class_c_symptoms = [
        'CLASS  C [11. APPETITE INCREASE]',
        'CLASS  C [12. FATIGUE]',
        'CLASS  C [13. HEADACHE]',
        'CLASS  C [14. FAINTING]',
        'CLASS  C [15. ABDOMINAL PAIN / BACK PAIN]'
    ]
    
    class_d_symptoms = [
        'CLASS D  [16. SWOLLEN EXTREMITIES]',
        'CLASS D  [17. BREAST TENDERNESS]',
        'CLASS D  [18. ABDOMINAL BLOATING]',
        'CLASS D  [19. WEIGHT GAIN]',
        'CLASS D  [20. FLUID RETENTION]'
    ]
    
    all_symptoms = class_a_symptoms + class_b_symptoms + class_c_symptoms + class_d_symptoms
    
    # Create mapping of symptoms to their severity columns
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
    
    # Create symptom to severity mapping
    symptom_to_severity = {}
    for symptom_key, severity_col in severity_mapping.items():
        if severity_col in df.columns:
            for symptom in all_symptoms:
                if symptom_key.lower() in symptom.lower() or symptom_key in symptom:
                    symptom_to_severity[symptom] = severity_col
                    break
    
    # 1. PREPARE THE DATA
    print("\n📊 PREPARING DATA FOR LOGISTIC REGRESSION")
    print("-" * 50)
    
    # Identify period heaviness column
    period_heaviness_col = None
    for col in df.columns:
        if 'heavy' in col.lower() or 'flow' in col.lower():
            period_heaviness_col = col
            break
    
    if period_heaviness_col is None:
        print("❌ Could not find period heaviness column")
        return
    
    print(f"Period heaviness column: {period_heaviness_col}")
    print(f"Period heaviness distribution:")
    print(df[period_heaviness_col].value_counts())
    
    # Create binary target variable (Heavy vs Non-Heavy)
    # Assuming 'Heavy' is the category we want to predict
    df['is_heavy_period'] = (df[period_heaviness_col] == 'Heavy').astype(int)
    
    print(f"\nBinary target distribution:")
    print(f"Heavy periods: {df['is_heavy_period'].sum()} ({df['is_heavy_period'].mean()*100:.1f}%)")
    print(f"Non-heavy periods: {(1-df['is_heavy_period']).sum()} ({(1-df['is_heavy_period']).mean()*100:.1f}%)")
    
    # Prepare severity features
    severity_features = []
    feature_names = []
    
    for symptom in all_symptoms:
        if symptom in symptom_to_severity:
            severity_col = symptom_to_severity[symptom]
            if severity_col in df.columns:
                severity_features.append(df[severity_col].fillna(0).values)
                feature_names.append(symptom.split('[')[1].split(']')[0].strip())
    
    if not severity_features:
        print("❌ No severity features found")
        return
    
    # Create feature matrix
    X = np.column_stack(severity_features)
    y = df['is_heavy_period'].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"Features: {feature_names}")
    
    # 2. EXPLORATORY DATA ANALYSIS
    print("\n\n📈 EXPLORATORY DATA ANALYSIS")
    print("-" * 50)
    
    # Create correlation analysis
    severity_df = pd.DataFrame(X, columns=feature_names)
    severity_df['is_heavy_period'] = y
    
    # Calculate correlations between each symptom severity and heavy periods
    correlations = []
    for feature in feature_names:
        corr, p_val = stats.pearsonr(severity_df[feature], severity_df['is_heavy_period'])
        correlations.append({
            'Symptom': feature,
            'Correlation': corr,
            'P_Value': p_val,
            'Significant': p_val < 0.05
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
    
    print("Top 10 symptoms with strongest correlation to heavy periods:")
    for i, (_, row) in enumerate(corr_df.head(10).iterrows(), 1):
        significance = "✓" if row['Significant'] else "✗"
        print(f"  {i:2d}. {row['Symptom']}: r = {row['Correlation']:.3f}, p = {row['P_Value']:.4f} {significance}")
    
    # 3. LOGISTIC REGRESSION MODEL
    print("\n\n🤖 LOGISTIC REGRESSION MODEL")
    print("-" * 50)
    
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
    
    # Model evaluation
    print("Model Performance:")
    print(f"Training accuracy: {lr_model.score(X_train_scaled, y_train):.3f}")
    print(f"Test accuracy: {lr_model.score(X_test_scaled, y_test):.3f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Heavy', 'Heavy']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {roc_auc:.3f}")
    
    # 4. FEATURE IMPORTANCE ANALYSIS
    print("\n\n🎯 FEATURE IMPORTANCE ANALYSIS")
    print("-" * 50)
    
    # Get coefficients
    coefficients = lr_model.coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients),
        'Odds_Ratio': np.exp(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("Top 10 Most Important Features (by absolute coefficient):")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        direction = "increases" if row['Coefficient'] > 0 else "decreases"
        print(f"  {i:2d}. {row['Feature']}: {direction} heavy period risk")
        print(f"      Coefficient: {row['Coefficient']:.3f}, Odds Ratio: {row['Odds_Ratio']:.3f}")
    
    # 5. STATISTICAL SIGNIFICANCE TESTING
    print("\n\n📊 STATISTICAL SIGNIFICANCE TESTING")
    print("-" * 50)
    
    # Perform chi-square tests for each symptom
    chi_square_results = []
    
    for i, feature in enumerate(feature_names):
        # Create contingency table: symptom severity vs heavy period
        contingency_table = pd.crosstab(
            pd.cut(severity_df[feature], bins=[-0.5, 0.5, 1.5, 2.5, 3.5], labels=['None', 'Mild', 'Moderate', 'Severe']),
            severity_df['is_heavy_period']
        )
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        chi_square_results.append({
            'Symptom': feature,
            'Chi2_Stat': chi2_stat,
            'P_Value': p_value,
            'Significant': p_value < 0.05
        })
    
    chi_square_df = pd.DataFrame(chi_square_results)
    chi_square_df = chi_square_df.sort_values('P_Value')
    
    print("Chi-square test results (symptom severity vs heavy periods):")
    significant_symptoms = chi_square_df[chi_square_df['Significant']]
    print(f"Significant associations: {len(significant_symptoms)}/{len(chi_square_df)}")
    
    if len(significant_symptoms) > 0:
        print("\nSignificant symptoms (p < 0.05):")
        for i, (_, row) in enumerate(significant_symptoms.iterrows(), 1):
            print(f"  {i:2d}. {row['Symptom']}: χ² = {row['Chi2_Stat']:.3f}, p = {row['P_Value']:.4f}")
    
    # 6. VISUALIZATIONS
    print("\n\n📈 CREATING VISUALIZATIONS")
    print("-" * 50)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure 1: Feature Importance Plot
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    
    colors = ['red' if coef > 0 else 'blue' for coef in top_features['Coefficient']]
    bars = plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors, alpha=0.7)
    
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Logistic Regression Coefficient', fontweight='bold')
    plt.title('Feature Importance in Predicting Heavy Periods\n(Red = Increases Risk, Blue = Decreases Risk)', 
              fontweight='bold', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_features['Coefficient'])):
        plt.text(value + (0.01 if value > 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', ha='left' if value > 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('logistic_regression_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created feature importance plot")
    
    # Figure 2: ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('ROC Curve: Heavy Period Prediction', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('logistic_regression_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created ROC curve")
    
    # Figure 3: Correlation Heatmap
    plt.figure(figsize=(14, 10))
    
    # Select top 15 most correlated symptoms
    top_corr_symptoms = corr_df.head(15)['Symptom'].tolist()
    top_corr_indices = [feature_names.index(symptom) for symptom in top_corr_symptoms]
    
    # Create correlation matrix for selected symptoms
    corr_matrix = np.corrcoef(X[:, top_corr_indices].T)
    
    sns.heatmap(corr_matrix, 
                xticklabels=top_corr_symptoms,
                yticklabels=top_corr_symptoms,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Matrix: Top Symptoms vs Heavy Periods', fontweight='bold', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('logistic_regression_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created correlation heatmap")
    
    # Figure 4: Symptom Severity Distribution by Period Type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    # Select top 4 most important symptoms
    top_4_symptoms = feature_importance.head(4)['Feature'].tolist()
    
    for i, symptom in enumerate(top_4_symptoms):
        if i < 4:
            symptom_idx = feature_names.index(symptom)
            
            # Create box plot
            heavy_periods = X[y == 1, symptom_idx]
            non_heavy_periods = X[y == 0, symptom_idx]
            
            data_to_plot = [non_heavy_periods, heavy_periods]
            labels = ['Non-Heavy Periods', 'Heavy Periods']
            
            box_plot = axes[i].boxplot(data_to_plot, labels=labels, patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightblue')
            box_plot['boxes'][1].set_facecolor('lightcoral')
            
            axes[i].set_title(f'{symptom}\nSeverity Distribution by Period Type', fontweight='bold')
            axes[i].set_ylabel('Severity Score (0-3)')
            axes[i].grid(alpha=0.3)
            
            # Add statistical test
            t_stat, p_val = stats.ttest_ind(heavy_periods, non_heavy_periods)
            axes[i].text(0.05, 0.95, f't-test: p = {p_val:.4f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('logistic_regression_symptom_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created symptom severity distribution plots")
    
    # 7. SUMMARY REPORT
    print("\n\n📋 LOGISTIC REGRESSION SUMMARY")
    print("-" * 50)
    
    print(f"Dataset: {len(df)} participants")
    print(f"Features: {len(feature_names)} symptom severity scores")
    print(f"Target: Heavy vs Non-Heavy periods")
    print(f"Model accuracy: {lr_model.score(X_test_scaled, y_test):.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    
    print(f"\nMost predictive symptoms for heavy periods:")
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
        direction = "increases" if row['Coefficient'] > 0 else "decreases"
        print(f"  {i}. {row['Feature']}: {direction} risk (OR = {row['Odds_Ratio']:.3f})")
    
    print(f"\nStatistical significance:")
    print(f"  Significant chi-square tests: {len(significant_symptoms)}/{len(chi_square_df)}")
    print(f"  Significant correlations: {len(corr_df[corr_df['Significant']])}/{len(corr_df)}")
    
    # Clinical interpretation
    print(f"\nClinical Interpretation:")
    if roc_auc > 0.7:
        print("  ✓ Model shows good predictive ability for heavy periods")
    elif roc_auc > 0.6:
        print("  ⚠ Model shows moderate predictive ability for heavy periods")
    else:
        print("  ✗ Model shows poor predictive ability for heavy periods")
    
    if len(significant_symptoms) > 0:
        print(f"  ✓ {len(significant_symptoms)} symptoms show significant association with heavy periods")
        print("  ✓ Symptom severity can be used as a predictor for period heaviness")
    else:
        print("  ✗ No significant associations found between symptoms and period heaviness")
    
    print("\n" + "=" * 80)
    print("Logistic regression analysis complete! 🎉")
    print("Generated files:")
    print("  • logistic_regression_feature_importance.png")
    print("  • logistic_regression_roc_curve.png")
    print("  • logistic_regression_correlation_heatmap.png")
    print("  • logistic_regression_symptom_distributions.png")
    print("=" * 80)
    
    return {
        'model': lr_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'feature_importance': feature_importance,
        'correlations': corr_df,
        'chi_square_results': chi_square_df,
        'roc_auc': roc_auc,
        'accuracy': lr_model.score(X_test_scaled, y_test)
    }

if __name__ == "__main__":
    results = logistic_regression_analysis("../data/DATA SHEET.xlsx")
