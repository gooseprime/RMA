import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, chi2
from scipy.stats import pearsonr, spearmanr
from scipy.stats.contingency import association
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

def cramers_v(confusion_matrix):
    """Calculate Cramér's V statistic for categorical-categorical association"""
    chi2_stat = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2_stat / (n * min_dim))

def statistical_analysis(file_path):
    """
    Comprehensive statistical analysis focusing on relationships between symptom classes
    """
    print("=" * 80)
    print("STATISTICAL ANALYSIS: SYMPTOM CLASS RELATIONSHIPS")
    print("=" * 80)
    
    # Read the dataset
    df = pd.read_excel(file_path, sheet_name=0)
    
    # Define symptom classes
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
    
    all_classes = {
        'Class A (Emotional)': class_a_symptoms,
        'Class B (Mental)': class_b_symptoms,
        'Class C (Physical)': class_c_symptoms,
        'Class D (Physical Changes)': class_d_symptoms
    }
    
    print("\n📊 SYMPTOM CLASS DEFINITIONS")
    print("-" * 50)
    for class_name, symptoms in all_classes.items():
        print(f"\n{class_name}:")
        for symptom in symptoms:
            print(f"  • {symptom}")
    
    # 1. CHI-SQUARE TESTS BETWEEN CLASSES
    print("\n\n🔬 CHI-SQUARE TESTS BETWEEN SYMPTOM CLASSES")
    print("-" * 50)
    
    chi_square_results = []
    
    # Test relationships between different classes
    class_names = list(all_classes.keys())
    
    for i in range(len(class_names)):
        for j in range(i+1, len(class_names)):
            class1_name = class_names[i]
            class2_name = class_names[j]
            class1_symptoms = all_classes[class1_name]
            class2_symptoms = all_classes[class2_name]
            
            print(f"\n{class1_name} vs {class2_name}:")
            print("-" * 40)
            
            # Test each symptom pair between classes
            for symptom1 in class1_symptoms:
                for symptom2 in class2_symptoms:
                    try:
                        # Create contingency table
                        contingency_table = pd.crosstab(
                            df[symptom1].fillna('Missing'), 
                            df[symptom2].fillna('Missing')
                        )
                        
                        # Perform chi-square test
                        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        # Calculate Cramér's V
                        cramers_v_value = cramers_v(contingency_table)
                        
                        # Store results
                        result = {
                            'Class1': class1_name,
                            'Class2': class2_name,
                            'Symptom1': symptom1,
                            'Symptom2': symptom2,
                            'Chi2_Stat': chi2_stat,
                            'P_Value': p_value,
                            'Cramers_V': cramers_v_value,
                            'Significant': p_value < 0.05
                        }
                        chi_square_results.append(result)
                        
                        # Print significant results
                        if p_value < 0.05:
                            print(f"  ✓ {symptom1.split('[')[-1].split(']')[0]} ↔ {symptom2.split('[')[-1].split(']')[0]}")
                            print(f"    Chi² = {chi2_stat:.3f}, p = {p_value:.4f}, Cramér's V = {cramers_v_value:.3f}")
                        
                    except Exception as e:
                        print(f"  ✗ Error testing {symptom1} vs {symptom2}: {str(e)}")
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(chi_square_results)
    
    # 2. SUMMARY OF SIGNIFICANT RELATIONSHIPS
    print("\n\n📈 SUMMARY OF SIGNIFICANT RELATIONSHIPS (p < 0.05)")
    print("-" * 50)
    
    significant_results = results_df[results_df['Significant'] == True].copy()
    significant_results = significant_results.sort_values('Cramers_V', ascending=False)
    
    if len(significant_results) > 0:
        print(f"Found {len(significant_results)} significant relationships:")
        print("\nTop 10 Strongest Associations (by Cramér's V):")
        for i, (_, row) in enumerate(significant_results.head(10).iterrows(), 1):
            symptom1_clean = row['Symptom1'].split('[')[-1].split(']')[0]
            symptom2_clean = row['Symptom2'].split('[')[-1].split(']')[0]
            print(f"  {i:2d}. {row['Class1']} {symptom1_clean} ↔ {row['Class2']} {symptom2_clean}")
            print(f"      Chi² = {row['Chi2_Stat']:.3f}, p = {row['P_Value']:.4f}, Cramér's V = {row['Cramers_V']:.3f}")
    else:
        print("No significant relationships found at p < 0.05 level.")
    
    # 3. CLASS-LEVEL ASSOCIATION ANALYSIS
    print("\n\n🎯 CLASS-LEVEL ASSOCIATION ANALYSIS")
    print("-" * 50)
    
    # Create class-level contingency tables
    class_associations = []
    
    for i in range(len(class_names)):
        for j in range(i+1, len(class_names)):
            class1_name = class_names[i]
            class2_name = class_names[j]
            
            # Get all significant relationships between these classes
            class_relationships = significant_results[
                ((significant_results['Class1'] == class1_name) & (significant_results['Class2'] == class2_name)) |
                ((significant_results['Class1'] == class2_name) & (significant_results['Class2'] == class1_name))
            ]
            
            if len(class_relationships) > 0:
                avg_cramers_v = class_relationships['Cramers_V'].mean()
                max_cramers_v = class_relationships['Cramers_V'].max()
                num_significant = len(class_relationships)
                
                class_associations.append({
                    'Class1': class1_name,
                    'Class2': class2_name,
                    'Num_Significant': num_significant,
                    'Avg_Cramers_V': avg_cramers_v,
                    'Max_Cramers_V': max_cramers_v
                })
                
                print(f"\n{class1_name} ↔ {class2_name}:")
                print(f"  Significant relationships: {num_significant}")
                print(f"  Average Cramér's V: {avg_cramers_v:.3f}")
                print(f"  Maximum Cramér's V: {max_cramers_v:.3f}")
    
    # 4. SEVERITY CORRELATION ANALYSIS
    print("\n\n📊 SEVERITY CORRELATION ANALYSIS")
    print("-" * 50)
    
    # Get severity columns
    severity_cols = [col for col in df.columns if 'SEVERITY' in col]
    
    # Create correlation matrix for severity scores
    severity_df = df[severity_cols].select_dtypes(include=[np.number])
    severity_corr = severity_df.corr()
    
    # Find strong correlations between different classes
    class_severity_correlations = []
    
    for i in range(len(class_names)):
        for j in range(i+1, len(class_names)):
            class1_name = class_names[i]
            class2_name = class_names[j]
            
            # Get severity columns for each class
            class1_severity = [col for col in severity_cols if any(symptom in col for symptom in all_classes[class1_name])]
            class2_severity = [col for col in severity_cols if any(symptom in col for symptom in all_classes[class2_name])]
            
            # Calculate correlations between classes
            max_corr = 0
            best_pair = None
            
            for col1 in class1_severity:
                for col2 in class2_severity:
                    if col1 in severity_corr.columns and col2 in severity_corr.columns:
                        corr_value = abs(severity_corr.loc[col1, col2])
                        if not np.isnan(corr_value) and corr_value > max_corr:
                            max_corr = corr_value
                            best_pair = (col1, col2)
            
            if best_pair:
                class_severity_correlations.append({
                    'Class1': class1_name,
                    'Class2': class2_name,
                    'Max_Correlation': max_corr,
                    'Best_Pair': best_pair
                })
                
                print(f"\n{class1_name} ↔ {class2_name}:")
                print(f"  Maximum correlation: {max_corr:.3f}")
                print(f"  Best pair: {best_pair[0].split('[')[-1].split(']')[0]} ↔ {best_pair[1].split('[')[-1].split(']')[0]}")
    
    # 5. VISUALIZATION OF RELATIONSHIPS
    print("\n\n📈 CREATING VISUALIZATIONS")
    print("-" * 50)
    
    # Create a heatmap of significant relationships
    if len(significant_results) > 0:
        # Create a matrix of Cramér's V values
        class_pairs = []
        cramers_matrix = []
        
        for i in range(len(class_names)):
            row = []
            for j in range(len(class_names)):
                if i == j:
                    row.append(1.0)  # Perfect correlation with self
                elif i < j:
                    # Find relationship between these classes
                    class1_name = class_names[i]
                    class2_name = class_names[j]
                    
                    class_relationships = significant_results[
                        ((significant_results['Class1'] == class1_name) & (significant_results['Class2'] == class2_name)) |
                        ((significant_results['Class1'] == class2_name) & (significant_results['Class2'] == class1_name))
                    ]
                    
                    if len(class_relationships) > 0:
                        avg_cramers_v = class_relationships['Cramers_V'].mean()
                        row.append(avg_cramers_v)
                    else:
                        row.append(0.0)
                else:
                    # Use symmetric value
                    row.append(cramers_matrix[j][i] if j < len(cramers_matrix) else 0.0)
            cramers_matrix.append(row)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        cramers_df = pd.DataFrame(cramers_matrix, 
                                index=[name.replace(' (', '\n(') for name in class_names],
                                columns=[name.replace(' (', '\n(') for name in class_names])
        
        sns.heatmap(cramers_df, annot=True, cmap='YlOrRd', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.3f')
        plt.title('Average Cramér\'s V Between Symptom Classes\n(Significant Relationships Only)', 
                 fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig('symptom_class_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created symptom class relationships heatmap")
    
    # Create bar chart of significant relationships by class
    if len(class_associations) > 0:
        class_assoc_df = pd.DataFrame(class_associations)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Number of significant relationships
        class_pairs = [f"{row['Class1']}\n↔\n{row['Class2']}" for _, row in class_assoc_df.iterrows()]
        ax1.bar(class_pairs, class_assoc_df['Num_Significant'], color='skyblue', alpha=0.7)
        ax1.set_title('Number of Significant Relationships\nBetween Symptom Classes')
        ax1.set_ylabel('Number of Significant Relationships')
        ax1.tick_params(axis='x', rotation=45)
        
        # Average Cramér's V
        ax2.bar(class_pairs, class_assoc_df['Avg_Cramers_V'], color='lightcoral', alpha=0.7)
        ax2.set_title('Average Cramér\'s V\nBetween Symptom Classes')
        ax2.set_ylabel('Average Cramér\'s V')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('class_association_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Created class association summary charts")
    
    # 6. STATISTICAL SUMMARY
    print("\n\n📋 STATISTICAL SUMMARY")
    print("-" * 50)
    
    total_tests = len(results_df)
    significant_tests = len(significant_results)
    significance_rate = (significant_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Total chi-square tests performed: {total_tests}")
    print(f"Significant relationships found: {significant_tests}")
    print(f"Significance rate: {significance_rate:.1f}%")
    
    if len(significant_results) > 0:
        print(f"\nCramér's V statistics:")
        print(f"  Mean: {significant_results['Cramers_V'].mean():.3f}")
        print(f"  Median: {significant_results['Cramers_V'].median():.3f}")
        print(f"  Range: {significant_results['Cramers_V'].min():.3f} - {significant_results['Cramers_V'].max():.3f}")
        
        # Effect size interpretation
        weak_effect = len(significant_results[significant_results['Cramers_V'] < 0.1])
        moderate_effect = len(significant_results[(significant_results['Cramers_V'] >= 0.1) & (significant_results['Cramers_V'] < 0.3)])
        strong_effect = len(significant_results[significant_results['Cramers_V'] >= 0.3])
        
        print(f"\nEffect size interpretation (Cramér's V):")
        print(f"  Weak effect (< 0.1): {weak_effect} relationships")
        print(f"  Moderate effect (0.1-0.3): {moderate_effect} relationships")
        print(f"  Strong effect (≥ 0.3): {strong_effect} relationships")
    
    # 7. RECOMMENDATIONS
    print("\n\n💡 RECOMMENDATIONS BASED ON STATISTICAL ANALYSIS")
    print("-" * 50)
    
    if len(significant_results) > 0:
        print("1. STRONGEST ASSOCIATIONS TO INVESTIGATE:")
        top_3 = significant_results.head(3)
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            symptom1_clean = row['Symptom1'].split('[')[-1].split(']')[0]
            symptom2_clean = row['Symptom2'].split('[')[-1].split(']')[0]
            print(f"   {i}. {row['Class1']} {symptom1_clean} ↔ {row['Class2']} {symptom2_clean} (V = {row['Cramers_V']:.3f})")
        
        print("\n2. CLINICAL IMPLICATIONS:")
        print("   • Consider integrated treatment approaches for strongly associated symptoms")
        print("   • Focus on symptom clusters rather than individual symptoms")
        print("   • Develop targeted interventions for the most significant relationships")
        
        print("\n3. FURTHER RESEARCH:")
        print("   • Investigate causal relationships in the strongest associations")
        print("   • Study temporal patterns of symptom co-occurrence")
        print("   • Evaluate treatment effectiveness on symptom clusters")
    else:
        print("No significant relationships found. Consider:")
        print("   • Increasing sample size")
        print("   • Refining symptom categories")
        print("   • Using different statistical approaches")
    
    print("\n" + "=" * 80)
    print("Statistical analysis complete! 🎉")
    print("Generated files:")
    print("  • symptom_class_relationships.png")
    print("  • class_association_summary.png")
    print("=" * 80)

if __name__ == "__main__":
    statistical_analysis("DATA SHEET.xlsx")
