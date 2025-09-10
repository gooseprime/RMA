import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
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
    if min_dim == 0:
        return 0
    return np.sqrt(chi2_stat / (n * min_dim))

def simple_chi_square_analysis(file_path):
    """
    Simple chi-square analysis of relationships between symptom classes
    """
    print("=" * 80)
    print("CHI-SQUARE ANALYSIS: SYMPTOM CLASS RELATIONSHIPS")
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
                        
                        # Check if table has sufficient data
                        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                            continue
                        
                        # Perform chi-square test
                        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        # Calculate Cramér's V
                        cramers_v_value = cramers_v(contingency_table)
                        
                        # Store results
                        result = {
                            'Class1': class1_name,
                            'Class2': class2_name,
                            'Symptom1': symptom1.split('[')[-1].split(']')[0],
                            'Symptom2': symptom2.split('[')[-1].split(']')[0],
                            'Chi2_Stat': chi2_stat,
                            'P_Value': p_value,
                            'Cramers_V': cramers_v_value,
                            'Significant': p_value < 0.05
                        }
                        chi_square_results.append(result)
                        
                        # Print significant results
                        if p_value < 0.05:
                            print(f"  ✓ {result['Symptom1']} ↔ {result['Symptom2']}")
                            print(f"    Chi² = {chi2_stat:.3f}, p = {p_value:.4f}, Cramér's V = {cramers_v_value:.3f}")
                        
                    except Exception as e:
                        continue  # Skip problematic tests
    
    # 2. SUMMARY OF SIGNIFICANT RELATIONSHIPS
    print("\n\n📈 SUMMARY OF SIGNIFICANT RELATIONSHIPS (p < 0.05)")
    print("-" * 50)
    
    if len(chi_square_results) > 0:
        # Filter significant results
        significant_results = [r for r in chi_square_results if r['Significant']]
        
        if len(significant_results) > 0:
            # Sort by Cramér's V
            significant_results.sort(key=lambda x: x['Cramers_V'], reverse=True)
            
            print(f"Found {len(significant_results)} significant relationships:")
            print("\nTop 15 Strongest Associations (by Cramér's V):")
            for i, result in enumerate(significant_results[:15], 1):
                print(f"  {i:2d}. {result['Class1']} {result['Symptom1']} ↔ {result['Class2']} {result['Symptom2']}")
                print(f"      Chi² = {result['Chi2_Stat']:.3f}, p = {result['P_Value']:.4f}, Cramér's V = {result['Cramers_V']:.3f}")
        else:
            print("No significant relationships found at p < 0.05 level.")
    else:
        print("No testable relationships found.")
    
    # 3. CLASS-LEVEL ASSOCIATION ANALYSIS
    print("\n\n🎯 CLASS-LEVEL ASSOCIATION ANALYSIS")
    print("-" * 50)
    
    if len(chi_square_results) > 0:
        significant_results = [r for r in chi_square_results if r['Significant']]
        
        if len(significant_results) > 0:
            # Create class-level analysis
            class_associations = {}
            
            for result in significant_results:
                class_pair = tuple(sorted([result['Class1'], result['Class2']]))
                if class_pair not in class_associations:
                    class_associations[class_pair] = []
                class_associations[class_pair].append(result)
            
            for class_pair, results in class_associations.items():
                class1_name, class2_name = class_pair
                avg_cramers_v = np.mean([r['Cramers_V'] for r in results])
                max_cramers_v = max([r['Cramers_V'] for r in results])
                num_significant = len(results)
                
                print(f"\n{class1_name} ↔ {class2_name}:")
                print(f"  Significant relationships: {num_significant}")
                print(f"  Average Cramér's V: {avg_cramers_v:.3f}")
                print(f"  Maximum Cramér's V: {max_cramers_v:.3f}")
    
    # 4. VISUALIZATION OF RELATIONSHIPS
    print("\n\n📈 CREATING VISUALIZATIONS")
    print("-" * 50)
    
    if len(chi_square_results) > 0:
        significant_results = [r for r in chi_square_results if r['Significant']]
        
        if len(significant_results) > 0:
            # Create a heatmap of significant relationships
            cramers_matrix = np.zeros((len(class_names), len(class_names)))
            
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    if i == j:
                        cramers_matrix[i, j] = 1.0  # Perfect correlation with self
                    elif i < j:
                        # Find relationship between these classes
                        class1_name = class_names[i]
                        class2_name = class_names[j]
                        
                        class_relationships = [r for r in significant_results 
                                            if (r['Class1'] == class1_name and r['Class2'] == class2_name) or
                                               (r['Class1'] == class2_name and r['Class2'] == class1_name)]
                        
                        if len(class_relationships) > 0:
                            avg_cramers_v = np.mean([r['Cramers_V'] for r in class_relationships])
                            cramers_matrix[i, j] = avg_cramers_v
                            cramers_matrix[j, i] = avg_cramers_v  # Make symmetric
                        else:
                            cramers_matrix[i, j] = 0.0
                            cramers_matrix[j, i] = 0.0
            
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
            class_associations = {}
            for result in significant_results:
                class_pair = tuple(sorted([result['Class1'], result['Class2']]))
                if class_pair not in class_associations:
                    class_associations[class_pair] = []
                class_associations[class_pair].append(result)
            
            if len(class_associations) > 0:
                class_pairs = []
                num_significant = []
                avg_cramers_v = []
                
                for class_pair, results in class_associations.items():
                    class_pairs.append(f"{class_pair[0]}\n↔\n{class_pair[1]}")
                    num_significant.append(len(results))
                    avg_cramers_v.append(np.mean([r['Cramers_V'] for r in results]))
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Number of significant relationships
                ax1.bar(class_pairs, num_significant, color='skyblue', alpha=0.7)
                ax1.set_title('Number of Significant Relationships\nBetween Symptom Classes')
                ax1.set_ylabel('Number of Significant Relationships')
                ax1.tick_params(axis='x', rotation=45)
                
                # Average Cramér's V
                ax2.bar(class_pairs, avg_cramers_v, color='lightcoral', alpha=0.7)
                ax2.set_title('Average Cramér\'s V\nBetween Symptom Classes')
                ax2.set_ylabel('Average Cramér\'s V')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig('class_association_summary.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print("✓ Created class association summary charts")
    
    # 5. STATISTICAL SUMMARY
    print("\n\n📋 STATISTICAL SUMMARY")
    print("-" * 50)
    
    if len(chi_square_results) > 0:
        total_tests = len(chi_square_results)
        significant_tests = len([r for r in chi_square_results if r['Significant']])
        significance_rate = (significant_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"Total chi-square tests performed: {total_tests}")
        print(f"Significant relationships found: {significant_tests}")
        print(f"Significance rate: {significance_rate:.1f}%")
        
        if significant_tests > 0:
            significant_results = [r for r in chi_square_results if r['Significant']]
            cramers_v_values = [r['Cramers_V'] for r in significant_results]
            
            print(f"\nCramér's V statistics:")
            print(f"  Mean: {np.mean(cramers_v_values):.3f}")
            print(f"  Median: {np.median(cramers_v_values):.3f}")
            print(f"  Range: {np.min(cramers_v_values):.3f} - {np.max(cramers_v_values):.3f}")
            
            # Effect size interpretation
            weak_effect = len([v for v in cramers_v_values if v < 0.1])
            moderate_effect = len([v for v in cramers_v_values if 0.1 <= v < 0.3])
            strong_effect = len([v for v in cramers_v_values if v >= 0.3])
            
            print(f"\nEffect size interpretation (Cramér's V):")
            print(f"  Weak effect (< 0.1): {weak_effect} relationships")
            print(f"  Moderate effect (0.1-0.3): {moderate_effect} relationships")
            print(f"  Strong effect (≥ 0.3): {strong_effect} relationships")
    else:
        print("No testable relationships found in the dataset.")
    
    # 6. RECOMMENDATIONS
    print("\n\n💡 RECOMMENDATIONS BASED ON CHI-SQUARE ANALYSIS")
    print("-" * 50)
    
    if len(chi_square_results) > 0:
        significant_results = [r for r in chi_square_results if r['Significant']]
        
        if len(significant_results) > 0:
            # Sort by Cramér's V
            significant_results.sort(key=lambda x: x['Cramers_V'], reverse=True)
            
            print("1. STRONGEST ASSOCIATIONS TO INVESTIGATE:")
            top_3 = significant_results[:3]
            for i, result in enumerate(top_3, 1):
                print(f"   {i}. {result['Class1']} {result['Symptom1']} ↔ {result['Class2']} {result['Symptom2']} (V = {result['Cramers_V']:.3f})")
            
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
            print("   • Checking data quality and completeness")
    else:
        print("No testable relationships found. Check data format and completeness.")
    
    print("\n" + "=" * 80)
    print("Chi-square analysis complete! 🎉")
    print("Generated files:")
    print("  • symptom_class_relationships.png")
    print("  • class_association_summary.png")
    print("=" * 80)

if __name__ == "__main__":
    simple_chi_square_analysis("DATA SHEET.xlsx")
