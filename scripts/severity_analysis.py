import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, chi2
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

def severity_analysis(file_path):
    """
    Comprehensive analysis of symptom severity patterns with specialized visualizations
    """
    print("=" * 80)
    print("SYMPTOM SEVERITY ANALYSIS")
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
    
    # Get severity columns
    severity_cols = [col for col in df.columns if 'SEVERITY' in col]
    
    # Create mapping of symptoms to their severity columns
    symptom_to_severity = {}
    
    # Map based on the actual column structure
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
    
    all_symptoms = class_a_symptoms + class_b_symptoms + class_c_symptoms + class_d_symptoms
    all_classes = {
        'Class A (Emotional)': class_a_symptoms,
        'Class B (Mental)': class_b_symptoms,
        'Class C (Physical)': class_c_symptoms,
        'Class D (Physical Changes)': class_d_symptoms
    }
    
    # Create symptom to severity mapping
    for symptom_key, severity_col in severity_mapping.items():
        if severity_col in df.columns:
            # Map to the original symptom names
            for symptom in all_symptoms:
                if symptom_key.lower() in symptom.lower() or symptom_key in symptom:
                    symptom_to_severity[symptom] = severity_col
                    break
    
    print(f"Found {len(severity_cols)} severity columns")
    print(f"Mapped {len(symptom_to_severity)} symptoms to severity columns")
    
    # 1. SEVERITY DISTRIBUTION ANALYSIS
    print("\n\n📊 SEVERITY DISTRIBUTION ANALYSIS")
    print("-" * 50)
    
    # Create severity data matrix
    severity_data = []
    for symptom in all_symptoms:
        if symptom in symptom_to_severity:
            severity_col = symptom_to_severity[symptom]
            if severity_col in df.columns:
                severity_values = df[severity_col].dropna()
                if len(severity_values) > 0:
                    severity_data.append({
                        'Symptom': symptom.split('[')[1].split(']')[0].strip(),
                        'Class': next((k for k, v in all_classes.items() if symptom in v), 'Unknown'),
                        'Mean': severity_values.mean(),
                        'Median': severity_values.median(),
                        'Std': severity_values.std(),
                        'Min': severity_values.min(),
                        'Max': severity_values.max(),
                        'Count': len(severity_values),
                        'Severity_0': (severity_values == 0).sum(),
                        'Severity_1': (severity_values == 1).sum(),
                        'Severity_2': (severity_values == 2).sum(),
                        'Severity_3': (severity_values == 3).sum()
                    })
    
    severity_df = pd.DataFrame(severity_data)
    
    # Print summary statistics
    print(f"Total symptoms analyzed: {len(severity_df)}")
    print(f"Overall mean severity: {severity_df['Mean'].mean():.3f}")
    print(f"Overall median severity: {severity_df['Median'].median():.3f}")
    
    # Top 5 most severe symptoms
    print("\nTop 5 Most Severe Symptoms (by mean):")
    top_severe = severity_df.nlargest(5, 'Mean')
    for i, (_, row) in enumerate(top_severe.iterrows(), 1):
        print(f"  {i}. {row['Symptom']} ({row['Class']}): {row['Mean']:.3f}")
    
    # 2. SEVERITY DISTRIBUTION VISUALIZATIONS
    print("\n\n📈 CREATING SEVERITY VISUALIZATIONS")
    print("-" * 50)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure 1: Severity Distribution by Class
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (class_name, symptoms) in enumerate(all_classes.items()):
        class_data = severity_df[severity_df['Class'] == class_name]
        if len(class_data) > 0:
            # Create severity level distribution
            severity_levels = ['None (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)']
            severity_counts = [
                class_data['Severity_0'].sum(),
                class_data['Severity_1'].sum(),
                class_data['Severity_2'].sum(),
                class_data['Severity_3'].sum()
            ]
            
            axes[i].pie(severity_counts, labels=severity_levels, autopct='%1.1f%%', 
                       colors=['#E8F4FD', '#B3D9FF', '#4A90E2', '#1F4E79'])
            axes[i].set_title(f'{class_name}\nSeverity Distribution', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('severity_distribution_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created severity distribution by class pie charts")
    
    # Figure 2: Mean Severity by Symptom (Horizontal Bar Chart)
    plt.figure(figsize=(14, 10))
    severity_df_sorted = severity_df.sort_values('Mean', ascending=True)
    
    # Color bars by class
    colors = []
    for _, row in severity_df_sorted.iterrows():
        if 'Emotional' in row['Class']:
            colors.append('#FF6B6B')
        elif 'Mental' in row['Class']:
            colors.append('#4ECDC4')
        elif 'Physical' in row['Class']:
            colors.append('#45B7D1')
        else:
            colors.append('#96CEB4')
    
    bars = plt.barh(range(len(severity_df_sorted)), severity_df_sorted['Mean'], color=colors, alpha=0.8)
    plt.yticks(range(len(severity_df_sorted)), severity_df_sorted['Symptom'])
    plt.xlabel('Mean Severity Score', fontweight='bold')
    plt.title('Mean Severity by Symptom\n(0=None, 1=Mild, 2=Moderate, 3=Severe)', fontweight='bold', fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, severity_df_sorted['Mean'])):
        plt.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.2f}', 
                va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('mean_severity_by_symptom.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created mean severity by symptom bar chart")
    
    # Figure 3: Severity Heatmap by Class
    plt.figure(figsize=(12, 8))
    
    # Create a matrix for heatmap
    heatmap_data = []
    for class_name, symptoms in all_classes.items():
        class_row = []
        for symptom in symptoms:
            if symptom in symptom_to_severity:
                severity_col = symptom_to_severity[symptom]
                if severity_col in df.columns:
                    mean_severity = df[severity_col].mean()
                    class_row.append(mean_severity if not np.isnan(mean_severity) else 0)
                else:
                    class_row.append(0)
            else:
                class_row.append(0)
        heatmap_data.append(class_row)
    
    # Create heatmap
    symptom_names = [s.split('[')[1].split(']')[0].strip() for s in all_symptoms]
    class_names_short = [name.split('(')[0].strip() for name in all_classes.keys()]
    
    sns.heatmap(heatmap_data, 
                xticklabels=symptom_names,
                yticklabels=class_names_short,
                annot=True, 
                cmap='Reds', 
                fmt='.2f',
                cbar_kws={'label': 'Mean Severity Score'})
    plt.title('Severity Heatmap by Class and Symptom', fontweight='bold', fontsize=14)
    plt.xlabel('Symptoms', fontweight='bold')
    plt.ylabel('Symptom Classes', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('severity_heatmap_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created severity heatmap by class")
    
    # Figure 4: Severity Distribution Histograms
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, (class_name, symptoms) in enumerate(all_classes.items()):
        class_data = severity_df[severity_df['Class'] == class_name]
        if len(class_data) > 0:
            # Create histogram of mean severities
            axes[i].hist(class_data['Mean'], bins=10, alpha=0.7, color=class_colors[i], edgecolor='black')
            axes[i].axvline(class_data['Mean'].mean(), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {class_data["Mean"].mean():.2f}')
            axes[i].axvline(class_data['Mean'].median(), color='blue', linestyle='--', linewidth=2,
                          label=f'Median: {class_data["Mean"].median():.2f}')
            axes[i].set_title(f'{class_name}\nDistribution of Mean Severities', fontweight='bold')
            axes[i].set_xlabel('Mean Severity Score')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('severity_distribution_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created severity distribution histograms")
    
    # Figure 5: Box Plot of Severity by Class
    plt.figure(figsize=(12, 8))
    
    # Prepare data for box plot
    box_data = []
    box_labels = []
    for class_name, symptoms in all_classes.items():
        class_data = severity_df[severity_df['Class'] == class_name]
        if len(class_data) > 0:
            box_data.append(class_data['Mean'].values)
            box_labels.append(class_name)
    
    box_plot = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], class_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Severity Distribution by Class\n(Box Plot)', fontweight='bold', fontsize=14)
    plt.ylabel('Mean Severity Score', fontweight='bold')
    plt.xlabel('Symptom Classes', fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('severity_boxplot_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created severity box plot by class")
    
    # Figure 6: Severity Correlation Network
    plt.figure(figsize=(14, 10))
    
    # Create correlation matrix for severity scores
    severity_matrix = []
    severity_labels = []
    
    for class_name, symptoms in all_classes.items():
        for symptom in symptoms:
            if symptom in symptom_to_severity:
                severity_col = symptom_to_severity[symptom]
                if severity_col in df.columns:
                    severity_matrix.append(df[severity_col].fillna(0).values)
                    severity_labels.append(symptom.split('[')[1].split(']')[0].strip())
    
    if len(severity_matrix) > 1:
        severity_matrix = np.array(severity_matrix).T
        correlation_matrix = np.corrcoef(severity_matrix.T)
        
        # Create correlation heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                    mask=mask,
                    xticklabels=severity_labels,
                    yticklabels=severity_labels,
                    annot=True, 
                    cmap='RdBu_r', 
                    center=0,
                    fmt='.2f',
                    square=True,
                    cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Severity Correlation Matrix\n(All Symptoms)', fontweight='bold', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('severity_correlation_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created severity correlation network")
    
    # Figure 7: Severity vs Demographics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    # Age vs Severity
    if 'Age ' in df.columns:
        age_data = df['Age '].dropna()
        overall_severity = []
        for _, row in df.iterrows():
            if not pd.isna(row['Age ']):
                person_severities = []
                for symptom in all_symptoms:
                    if symptom in symptom_to_severity:
                        severity_col = symptom_to_severity[symptom]
                        if severity_col in df.columns and not pd.isna(row[severity_col]):
                            person_severities.append(row[severity_col])
                if person_severities:
                    overall_severity.append(np.mean(person_severities))
                else:
                    overall_severity.append(np.nan)
        
        # Remove NaN values
        valid_indices = ~np.isnan(overall_severity)
        age_clean = age_data[valid_indices]
        severity_clean = np.array(overall_severity)[valid_indices]
        
        if len(age_clean) > 0:
            axes[0].scatter(age_clean, severity_clean, alpha=0.6, color='skyblue')
            z = np.polyfit(age_clean, severity_clean, 1)
            p = np.poly1d(z)
            axes[0].plot(age_clean, p(age_clean), "r--", alpha=0.8)
            axes[0].set_xlabel('Age')
            axes[0].set_ylabel('Overall Severity Score')
            axes[0].set_title('Age vs Overall Severity', fontweight='bold')
            axes[0].grid(alpha=0.3)
            
            # Calculate correlation
            corr, p_val = stats.pearsonr(age_clean, severity_clean)
            axes[0].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                        transform=axes[0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Weight vs Severity
    if 'Weight' in df.columns:
        weight_data = df['Weight'].dropna()
        if len(weight_data) > 0 and len(severity_clean) > 0:
            weight_clean = weight_data[valid_indices]
            if len(weight_clean) > 0:
                axes[1].scatter(weight_clean, severity_clean, alpha=0.6, color='lightcoral')
                z = np.polyfit(weight_clean, severity_clean, 1)
                p = np.poly1d(z)
                axes[1].plot(weight_clean, p(weight_clean), "r--", alpha=0.8)
                axes[1].set_xlabel('Weight (kg)')
                axes[1].set_ylabel('Overall Severity Score')
                axes[1].set_title('Weight vs Overall Severity', fontweight='bold')
                axes[1].grid(alpha=0.3)
                
                # Calculate correlation
                corr, p_val = stats.pearsonr(weight_clean, severity_clean)
                axes[1].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                            transform=axes[1].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Height vs Severity
    if 'Height' in df.columns:
        height_data = df['Height'].dropna()
        if len(height_data) > 0 and len(severity_clean) > 0:
            height_clean = height_data[valid_indices]
            if len(height_clean) > 0:
                axes[2].scatter(height_clean, severity_clean, alpha=0.6, color='lightgreen')
                z = np.polyfit(height_clean, severity_clean, 1)
                p = np.poly1d(z)
                axes[2].plot(height_clean, p(height_clean), "r--", alpha=0.8)
                axes[2].set_xlabel('Height (cm)')
                axes[2].set_ylabel('Overall Severity Score')
                axes[2].set_title('Height vs Overall Severity', fontweight='bold')
                axes[2].grid(alpha=0.3)
                
                # Calculate correlation
                corr, p_val = stats.pearsonr(height_clean, severity_clean)
                axes[2].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                            transform=axes[2].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # BMI vs Severity
    if 'Height' in df.columns and 'Weight' in df.columns:
        height_data = df['Height'].dropna()
        weight_data = df['Weight'].dropna()
        if len(height_data) > 0 and len(weight_data) > 0 and len(severity_clean) > 0:
            height_clean = height_data[valid_indices]
            weight_clean = weight_data[valid_indices]
            if len(height_clean) > 0 and len(weight_clean) > 0:
                # Calculate BMI
                bmi = weight_clean / ((height_clean / 100) ** 2)
                axes[3].scatter(bmi, severity_clean, alpha=0.6, color='gold')
                z = np.polyfit(bmi, severity_clean, 1)
                p = np.poly1d(z)
                axes[3].plot(bmi, p(bmi), "r--", alpha=0.8)
                axes[3].set_xlabel('BMI (kg/m²)')
                axes[3].set_ylabel('Overall Severity Score')
                axes[3].set_title('BMI vs Overall Severity', fontweight='bold')
                axes[3].grid(alpha=0.3)
                
                # Calculate correlation
                corr, p_val = stats.pearsonr(bmi, severity_clean)
                axes[3].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                            transform=axes[3].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('severity_vs_demographics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created severity vs demographics scatter plots")
    
    # 3. STATISTICAL ANALYSIS OF SEVERITY PATTERNS
    print("\n\n📊 STATISTICAL ANALYSIS OF SEVERITY PATTERNS")
    print("-" * 50)
    
    # Class-wise severity comparison
    print("Class-wise Severity Comparison:")
    for class_name, symptoms in all_classes.items():
        class_data = severity_df[severity_df['Class'] == class_name]
        if len(class_data) > 0:
            print(f"\n{class_name}:")
            print(f"  Mean severity: {class_data['Mean'].mean():.3f} ± {class_data['Mean'].std():.3f}")
            print(f"  Median severity: {class_data['Mean'].median():.3f}")
            print(f"  Range: {class_data['Mean'].min():.3f} - {class_data['Mean'].max():.3f}")
    
    # ANOVA test for class differences
    if len(severity_df) > 0:
        class_groups = []
        for class_name, symptoms in all_classes.items():
            class_data = severity_df[severity_df['Class'] == class_name]['Mean'].values
            if len(class_data) > 0:
                class_groups.append(class_data)
        
        if len(class_groups) > 1:
            f_stat, p_value = stats.f_oneway(*class_groups)
            print(f"\nANOVA Test for Class Differences:")
            print(f"  F-statistic: {f_stat:.3f}")
            print(f"  p-value: {p_value:.6f}")
            if p_value < 0.05:
                print("  Result: Significant differences between classes (p < 0.05)")
            else:
                print("  Result: No significant differences between classes (p ≥ 0.05)")
    
    # 4. SEVERITY THRESHOLD ANALYSIS
    print("\n\n🎯 SEVERITY THRESHOLD ANALYSIS")
    print("-" * 50)
    
    # Define severity thresholds
    thresholds = {
        'Mild': 1.0,
        'Moderate': 2.0,
        'Severe': 2.5
    }
    
    for threshold_name, threshold_value in thresholds.items():
        print(f"\n{threshold_name} Threshold (≥ {threshold_value}):")
        above_threshold = severity_df[severity_df['Mean'] >= threshold_value]
        print(f"  Symptoms above threshold: {len(above_threshold)}/{len(severity_df)} ({len(above_threshold)/len(severity_df)*100:.1f}%)")
        
        if len(above_threshold) > 0:
            print("  Top symptoms above threshold:")
            top_above = above_threshold.nlargest(5, 'Mean')
            for i, (_, row) in enumerate(top_above.iterrows(), 1):
                print(f"    {i}. {row['Symptom']} ({row['Class']}): {row['Mean']:.3f}")
    
    # 5. SUMMARY REPORT
    print("\n\n📋 SEVERITY ANALYSIS SUMMARY")
    print("-" * 50)
    
    print(f"Total symptoms analyzed: {len(severity_df)}")
    print(f"Overall mean severity: {severity_df['Mean'].mean():.3f} ± {severity_df['Mean'].std():.3f}")
    print(f"Severity range: {severity_df['Mean'].min():.3f} - {severity_df['Mean'].max():.3f}")
    
    # Most and least severe symptoms
    most_severe = severity_df.loc[severity_df['Mean'].idxmax()]
    least_severe = severity_df.loc[severity_df['Mean'].idxmin()]
    
    print(f"\nMost severe symptom: {most_severe['Symptom']} ({most_severe['Class']}) - {most_severe['Mean']:.3f}")
    print(f"Least severe symptom: {least_severe['Symptom']} ({least_severe['Class']}) - {least_severe['Mean']:.3f}")
    
    # Class ranking by mean severity
    class_means = severity_df.groupby('Class')['Mean'].mean().sort_values(ascending=False)
    print(f"\nClass ranking by mean severity:")
    for i, (class_name, mean_sev) in enumerate(class_means.items(), 1):
        print(f"  {i}. {class_name}: {mean_sev:.3f}")
    
    print("\n" + "=" * 80)
    print("Severity analysis complete! 🎉")
    print("Generated files:")
    print("  • severity_distribution_by_class.png")
    print("  • mean_severity_by_symptom.png")
    print("  • severity_heatmap_by_class.png")
    print("  • severity_distribution_histograms.png")
    print("  • severity_boxplot_by_class.png")
    print("  • severity_correlation_network.png")
    print("  • severity_vs_demographics.png")
    print("=" * 80)

if __name__ == "__main__":
    severity_analysis("DATA SHEET.xlsx")
