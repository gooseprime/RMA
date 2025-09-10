import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr, spearmanr, kruskal, mannwhitneyu
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

def comprehensive_statistical_analysis(file_path):
    """
    Comprehensive statistical analysis including chi-square, correlation, and other tests
    """
    print("=" * 80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS: SYMPTOM CLASS RELATIONSHIPS")
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
            clean_name = symptom.split('[')[-1].split(']')[0]
            print(f"  • {clean_name}")
    
    # 1. SEVERITY CORRELATION ANALYSIS
    print("\n\n🔗 SEVERITY CORRELATION ANALYSIS")
    print("-" * 50)
    
    # Get severity columns
    severity_cols = [col for col in df.columns if 'SEVERITY' in col]
    severity_df = df[severity_cols].select_dtypes(include=[np.number])
    
    print(f"Found {len(severity_df.columns)} severity columns for correlation analysis")
    
    # Calculate correlation matrix
    correlation_matrix = severity_df.corr()
    
    # Find strong correlations between different classes
    class_correlations = []
    
    class_names = list(all_classes.keys())
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
            correlations = []
            
            for col1 in class1_severity:
                for col2 in class2_severity:
                    if col1 in correlation_matrix.columns and col2 in correlation_matrix.columns:
                        corr_value = correlation_matrix.loc[col1, col2]
                        if not np.isnan(corr_value):
                            correlations.append(abs(corr_value))
                            if abs(corr_value) > max_corr:
                                max_corr = abs(corr_value)
                                best_pair = (col1, col2)
            
            if best_pair and len(correlations) > 0:
                avg_corr = np.mean(correlations)
                class_correlations.append({
                    'Class1': class1_name,
                    'Class2': class2_name,
                    'Max_Correlation': max_corr,
                    'Avg_Correlation': avg_corr,
                    'Best_Pair': best_pair,
                    'Num_Comparisons': len(correlations)
                })
                
                print(f"\n{class1_name} ↔ {class2_name}:")
                print(f"  Maximum correlation: {max_corr:.3f}")
                print(f"  Average correlation: {avg_corr:.3f}")
                print(f"  Number of comparisons: {len(correlations)}")
                if best_pair:
                    symptom1 = best_pair[0].split('[')[-1].split(']')[0] if '[' in best_pair[0] else best_pair[0]
                    symptom2 = best_pair[1].split('[')[-1].split(']')[0] if '[' in best_pair[1] else best_pair[1]
                    print(f"  Best pair: {symptom1} ↔ {symptom2}")
    
    # 2. CHI-SQUARE TESTS FOR CATEGORICAL RELATIONSHIPS
    print("\n\n🔬 CHI-SQUARE TESTS FOR CATEGORICAL RELATIONSHIPS")
    print("-" * 50)
    
    chi_square_results = []
    
    # Test relationships between different classes using categorical data
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
                        
                        # Check if table has sufficient data
                        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                            continue
                        
                        # Check if expected frequencies are adequate
                        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        # Check if more than 20% of expected frequencies are less than 5
                        expected_freq_lt_5 = (expected < 5).sum()
                        total_expected = expected.size
                        if expected_freq_lt_5 / total_expected > 0.2:
                            continue  # Skip if chi-square assumptions not met
                        
                        # Calculate Cramér's V
                        n = contingency_table.sum().sum()
                        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                        
                        # Store results
                        result = {
                            'Class1': class1_name,
                            'Class2': class2_name,
                            'Symptom1': symptom1.split('[')[-1].split(']')[0],
                            'Symptom2': symptom2.split('[')[-1].split(']')[0],
                            'Chi2_Stat': chi2_stat,
                            'P_Value': p_value,
                            'Cramers_V': cramers_v,
                            'Significant': p_value < 0.05
                        }
                        chi_square_results.append(result)
                        
                        # Print significant results
                        if p_value < 0.05:
                            print(f"  ✓ {result['Symptom1']} ↔ {result['Symptom2']}")
                            print(f"    Chi² = {chi2_stat:.3f}, p = {p_value:.4f}, Cramér's V = {cramers_v:.3f}")
                        
                    except Exception as e:
                        continue  # Skip problematic tests
    
    # 3. SUMMARY OF SIGNIFICANT RELATIONSHIPS
    print("\n\n📈 SUMMARY OF SIGNIFICANT RELATIONSHIPS")
    print("-" * 50)
    
    # Chi-square results
    if len(chi_square_results) > 0:
        significant_chi = [r for r in chi_square_results if r['Significant']]
        
        if len(significant_chi) > 0:
            significant_chi.sort(key=lambda x: x['Cramers_V'], reverse=True)
            
            print(f"Chi-square tests: {len(significant_chi)} significant relationships found")
            print("\nTop 10 Strongest Chi-square Associations:")
            for i, result in enumerate(significant_chi[:10], 1):
                print(f"  {i:2d}. {result['Class1']} {result['Symptom1']} ↔ {result['Class2']} {result['Symptom2']}")
                print(f"      Chi² = {result['Chi2_Stat']:.3f}, p = {result['P_Value']:.4f}, Cramér's V = {result['Cramers_V']:.3f}")
        else:
            print("Chi-square tests: No significant relationships found at p < 0.05 level")
    else:
        print("Chi-square tests: No testable relationships found")
    
    # Correlation results
    if len(class_correlations) > 0:
        strong_correlations = [r for r in class_correlations if r['Max_Correlation'] > 0.3]
        
        if len(strong_correlations) > 0:
            strong_correlations.sort(key=lambda x: x['Max_Correlation'], reverse=True)
            
            print(f"\nCorrelation analysis: {len(strong_correlations)} strong correlations found (r > 0.3)")
            print("\nStrongest Correlations Between Classes:")
            for i, result in enumerate(strong_correlations[:5], 1):
                print(f"  {i}. {result['Class1']} ↔ {result['Class2']}: r = {result['Max_Correlation']:.3f}")
        else:
            print("\nCorrelation analysis: No strong correlations found (r > 0.3)")
    
    # 4. VISUALIZATION OF RELATIONSHIPS
    print("\n\n📈 CREATING VISUALIZATIONS")
    print("-" * 50)
    
    # Create correlation heatmap for severity scores
    if len(severity_df.columns) > 0:
        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        plt.title('Correlation Matrix of Symptom Severity Scores', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig('severity_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created severity correlation matrix")
    
    # Create class-level correlation heatmap
    if len(class_correlations) > 0:
        # Create a matrix of correlations between classes
        corr_matrix = np.zeros((len(class_names), len(class_names)))
        
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i == j:
                    corr_matrix[i, j] = 1.0  # Perfect correlation with self
                elif i < j:
                    # Find correlation between these classes
                    class1_name = class_names[i]
                    class2_name = class_names[j]
                    
                    class_corr = next((r for r in class_correlations 
                                     if (r['Class1'] == class1_name and r['Class2'] == class2_name) or
                                        (r['Class1'] == class2_name and r['Class2'] == class1_name)), None)
                    
                    if class_corr:
                        corr_matrix[i, j] = class_corr['Max_Correlation']
                        corr_matrix[j, i] = class_corr['Max_Correlation']
                    else:
                        corr_matrix[i, j] = 0.0
                        corr_matrix[j, i] = 0.0
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        corr_df = pd.DataFrame(corr_matrix, 
                              index=[name.replace(' (', '\n(') for name in class_names],
                              columns=[name.replace(' (', '\n(') for name in class_names])
        
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.3f')
        plt.title('Maximum Correlations Between Symptom Classes\n(Severity Scores)', 
                 fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig('class_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created class correlation heatmap")
    
    # 5. STATISTICAL SUMMARY
    print("\n\n📋 COMPREHENSIVE STATISTICAL SUMMARY")
    print("-" * 50)
    
    total_chi_tests = len(chi_square_results)
    significant_chi_tests = len([r for r in chi_square_results if r['Significant']])
    chi_significance_rate = (significant_chi_tests / total_chi_tests) * 100 if total_chi_tests > 0 else 0
    
    print(f"Chi-square Analysis:")
    print(f"  Total tests performed: {total_chi_tests}")
    print(f"  Significant relationships: {significant_chi_tests}")
    print(f"  Significance rate: {chi_significance_rate:.1f}%")
    
    print(f"\nCorrelation Analysis:")
    print(f"  Class pairs analyzed: {len(class_correlations)}")
    strong_correlations = [r for r in class_correlations if r['Max_Correlation'] > 0.3]
    print(f"  Strong correlations (r > 0.3): {len(strong_correlations)}")
    
    if len(strong_correlations) > 0:
        avg_strong_corr = np.mean([r['Max_Correlation'] for r in strong_correlations])
        print(f"  Average strong correlation: {avg_strong_corr:.3f}")
    
    # 6. RECOMMENDATIONS
    print("\n\n💡 RECOMMENDATIONS BASED ON COMPREHENSIVE ANALYSIS")
    print("-" * 50)
    
    if len(chi_square_results) > 0:
        significant_chi = [r for r in chi_square_results if r['Significant']]
        
        if len(significant_chi) > 0:
            print("1. CHI-SQUARE FINDINGS:")
            print("   • Significant categorical relationships found between symptom classes")
            print("   • Consider these relationships in treatment planning")
            print("   • Focus on symptom clusters rather than individual symptoms")
        
    if len(strong_correlations) > 0:
        print("\n2. CORRELATION FINDINGS:")
        print("   • Strong correlations found between severity scores of different classes")
        print("   • These suggest shared underlying mechanisms")
        print("   • Consider integrated treatment approaches")
        
        # Find strongest correlation
        strongest = max(strong_correlations, key=lambda x: x['Max_Correlation'])
        print(f"   • Strongest relationship: {strongest['Class1']} ↔ {strongest['Class2']} (r = {strongest['Max_Correlation']:.3f})")
    
    print("\n3. CLINICAL IMPLICATIONS:")
    print("   • Develop symptom cluster-based treatment protocols")
    print("   • Consider multidisciplinary approaches for complex cases")
    print("   • Focus on early intervention for high-risk symptom combinations")
    
    print("\n4. FURTHER RESEARCH:")
    print("   • Investigate causal mechanisms in strong relationships")
    print("   • Study temporal patterns of symptom development")
    print("   • Evaluate treatment effectiveness on symptom clusters")
    print("   • Consider longitudinal studies to track symptom evolution")
    
    print("\n" + "=" * 80)
    print("Comprehensive statistical analysis complete! 🎉")
    print("Generated files:")
    print("  • severity_correlation_matrix.png")
    print("  • class_correlation_heatmap.png")
    print("=" * 80)

if __name__ == "__main__":
    comprehensive_statistical_analysis("DATA SHEET.xlsx")
