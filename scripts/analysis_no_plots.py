import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

def comprehensive_analysis(file_path):
    """
    Comprehensive analysis of the menstrual health dataset without interactive plots
    """
    print("=" * 80)
    print("COMPREHENSIVE MENSTRUAL HEALTH DATASET ANALYSIS")
    print("=" * 80)
    
    # Read the dataset
    df = pd.read_excel(file_path, sheet_name=0)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. DEMOGRAPHIC ANALYSIS
    print("\n📊 DEMOGRAPHIC ANALYSIS")
    print("-" * 50)
    
    # Age distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Demographic Analysis', fontsize=16, fontweight='bold')
    
    # Age distribution
    axes[0,0].hist(df['Age '].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Age Distribution')
    axes[0,0].set_xlabel('Age (years)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(df['Age '].mean(), color='red', linestyle='--', label=f'Mean: {df["Age "].mean():.1f}')
    axes[0,0].legend()
    
    # Height vs Weight scatter
    axes[0,1].scatter(df['Height'], df['Weight'], alpha=0.6, color='green')
    axes[0,1].set_title('Height vs Weight')
    axes[0,1].set_xlabel('Height (cm)')
    axes[0,1].set_ylabel('Weight (kg)')
    
    # Age of first period
    axes[1,0].hist(df['1. Age of First period'].dropna(), bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].set_title('Age of First Period')
    axes[1,0].set_xlabel('Age (years)')
    axes[1,0].set_ylabel('Frequency')
    
    # Period duration
    axes[1,1].hist(df['5. How long is your period duration (in days) ? Please click below the option'].dropna(), 
                   bins=10, alpha=0.7, color='purple', edgecolor='black')
    axes[1,1].set_title('Period Duration')
    axes[1,1].set_xlabel('Days')
    axes[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('demographic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print demographic statistics
    print(f"Age Statistics:")
    print(f"  Mean: {df['Age '].mean():.1f} years")
    print(f"  Median: {df['Age '].median():.1f} years")
    print(f"  Range: {df['Age '].min():.0f} - {df['Age '].max():.0f} years")
    print(f"  Standard Deviation: {df['Age '].std():.1f} years")
    
    print(f"\nHeight Statistics:")
    print(f"  Mean: {df['Height'].mean():.1f} cm")
    print(f"  Range: {df['Height'].min():.0f} - {df['Height'].max():.0f} cm")
    
    print(f"\nWeight Statistics:")
    print(f"  Mean: {df['Weight'].mean():.1f} kg")
    print(f"  Range: {df['Weight'].min():.0f} - {df['Weight'].max():.0f} kg")
    
    print(f"\nAge of First Period:")
    print(f"  Mean: {df['1. Age of First period'].mean():.1f} years")
    print(f"  Range: {df['1. Age of First period'].min():.0f} - {df['1. Age of First period'].max():.0f} years")
    
    # 2. MENSTRUAL CYCLE ANALYSIS
    print("\n🩸 MENSTRUAL CYCLE ANALYSIS")
    print("-" * 50)
    
    # Regular periods
    regular_periods = df['2. Do you have regular periods (Y/N)'].value_counts()
    print("Regular Periods Distribution:")
    for period_type, count in regular_periods.items():
        percentage = (count / len(df)) * 100
        print(f"  {period_type}: {count} ({percentage:.1f}%)")
    
    # Period intervals
    print("\nPeriod Intervals:")
    intervals = df['3. Regular intervals between periods'].value_counts()
    for interval, count in intervals.head(5).items():
        percentage = (count / len(df)) * 100
        print(f"  {interval}: {count} ({percentage:.1f}%)")
    
    # Period heaviness
    print("\nPeriod Heaviness:")
    heaviness = df['4. How heavy is your menstrual period usually?'].value_counts()
    for level, count in heaviness.items():
        percentage = (count / len(df)) * 100
        print(f"  {level}: {count} ({percentage:.1f}%)")
    
    # 3. SYMPTOM SEVERITY ANALYSIS
    print("\n😰 SYMPTOM SEVERITY ANALYSIS")
    print("-" * 50)
    
    # Get all severity columns
    severity_cols = [col for col in df.columns if 'SEVERITY' in col]
    
    # Calculate mean severity for each symptom
    severity_data = []
    symptom_names = []
    
    for col in severity_cols:
        if df[col].dtype in ['int64', 'float64']:
            mean_severity = df[col].mean()
            severity_data.append(mean_severity)
            # Clean up symptom name
            symptom_name = col.split('[')[-1].split(']')[0] if '[' in col else col
            symptom_names.append(symptom_name)
    
    # Create severity bar chart
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(symptom_names)), severity_data, color='lightcoral', alpha=0.7)
    plt.xlabel('Symptoms')
    plt.ylabel('Mean Severity (0=None, 1=Mild, 2=Moderate, 3=High/Severe)')
    plt.title('Average Symptom Severity Across All Participants', fontweight='bold')
    plt.xticks(range(len(symptom_names)), symptom_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, severity) in enumerate(zip(bars, severity_data)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{severity:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('symptom_severity_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print top 5 most severe symptoms
    severity_df = pd.DataFrame({
        'Symptom': symptom_names,
        'Mean_Severity': severity_data
    }).sort_values('Mean_Severity', ascending=False)
    
    print("Top 5 Most Severe Symptoms:")
    for i, (_, row) in enumerate(severity_df.head().iterrows(), 1):
        print(f"  {i}. {row['Symptom']}: {row['Mean_Severity']:.2f}")
    
    # 4. SYMPTOM TIMING ANALYSIS
    print("\n⏰ SYMPTOM TIMING ANALYSIS")
    print("-" * 50)
    
    # Get all CLASS columns (symptom timing)
    class_cols = [col for col in df.columns if 'CLASS' in col]
    
    # Analyze timing patterns
    timing_analysis = {}
    for col in class_cols:
        timing_counts = df[col].value_counts()
        timing_analysis[col] = timing_counts
    
    # Create timing visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Symptom Timing Analysis', fontsize=16, fontweight='bold')
    
    # Plot timing for different symptom classes
    class_groups = {
        'Class A (Emotional)': [col for col in class_cols if 'CLASS  A' in col],
        'Class B (Mental)': [col for col in class_cols if 'CLASS B' in col],
        'Class C (Physical)': [col for col in class_cols if 'CLASS  C' in col],
        'Class D (Physical Changes)': [col for col in class_cols if 'CLASS D' in col]
    }
    
    for i, (group_name, cols) in enumerate(class_groups.items()):
        if i < 4:  # Only plot first 4 groups
            ax = axes[i//2, i%2]
            
            # Combine all timing data for this class
            all_timing = []
            for col in cols:
                timing_data = df[col].value_counts()
                for timing, count in timing_data.items():
                    all_timing.extend([timing] * count)
            
            if all_timing:
                timing_series = pd.Series(all_timing)
                timing_counts = timing_series.value_counts()
                
                # Create pie chart
                colors = ['#ff9999', '#66b3ff', '#99ff99']
                wedges, texts, autotexts = ax.pie(timing_counts.values, 
                                                labels=timing_counts.index,
                                                autopct='%1.1f%%',
                                                colors=colors[:len(timing_counts)],
                                                startangle=90)
                ax.set_title(f'{group_name}\nSymptom Timing')
    
    plt.tight_layout()
    plt.savefig('symptom_timing_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. LIFESTYLE AND TREATMENT ANALYSIS
    print("\n🧘 LIFESTYLE AND TREATMENT ANALYSIS")
    print("-" * 50)
    
    # Work interference
    work_interference = df['6.  Have menstrual problems ever interfered with your work responsibilities?(Y/N)'].value_counts()
    print("Work Interference:")
    for response, count in work_interference.items():
        percentage = (count / len(df)) * 100
        print(f"  {response}: {count} ({percentage:.1f}%)")
    
    # Pain killer usage
    pain_killers = df['9. Have you taken any pain killers prescribed by doctor in the last 3 months? (Y/N)'].value_counts()
    print("\nPain Killer Usage (Last 3 months):")
    for response, count in pain_killers.items():
        percentage = (count / len(df)) * 100
        print(f"  {response}: {count} ({percentage:.1f}%)")
    
    # Yoga/Meditation practice
    yoga_meditation = df['10. Do you practice any Yoga or Meditation (Y/N)'].value_counts()
    print("\nYoga/Meditation Practice:")
    for response, count in yoga_meditation.items():
        percentage = (count / len(df)) * 100
        print(f"  {response}: {count} ({percentage:.1f}%)")
    
    # 6. CORRELATION ANALYSIS
    print("\n🔗 CORRELATION ANALYSIS")
    print("-" * 50)
    
    # Select numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
    plt.title('Correlation Matrix of Numerical Variables', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find highest correlations
    correlation_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if not np.isnan(corr_value):
                correlation_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    corr_value
                ))
    
    # Sort by absolute correlation value
    correlation_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("Top 10 Strongest Correlations:")
    for i, (var1, var2, corr) in enumerate(correlation_pairs[:10], 1):
        print(f"  {i:2d}. {var1} ↔ {var2}: {corr:.3f}")
    
    # 7. SUMMARY INSIGHTS
    print("\n💡 KEY INSIGHTS AND RECOMMENDATIONS")
    print("-" * 50)
    
    print("1. DEMOGRAPHIC INSIGHTS:")
    print(f"   • Study population: {len(df)} participants, mostly young adults (mean age: {df['Age '].mean():.1f} years)")
    print(f"   • {regular_periods.get('Y', 0)} participants ({regular_periods.get('Y', 0)/len(df)*100:.1f}%) have regular periods")
    print(f"   • Most common period interval: {intervals.index[0]} ({intervals.iloc[0]} participants)")
    
    print("\n2. SYMPTOM PATTERNS:")
    print(f"   • Most severe symptom: {severity_df.iloc[0]['Symptom']} (severity: {severity_df.iloc[0]['Mean_Severity']:.2f})")
    print(f"   • {work_interference.get('Y', 0)} participants ({work_interference.get('Y', 0)/len(df)*100:.1f}%) report work interference")
    print(f"   • {pain_killers.get('Y', 0)} participants ({pain_killers.get('Y', 0)/len(df)*100:.1f}%) use prescribed pain killers")
    
    print("\n3. LIFESTYLE FACTORS:")
    yoga_yes = sum([count for response, count in yoga_meditation.items() if 'Y' in str(response).upper()])
    print(f"   • {yoga_yes} participants practice yoga/meditation ({yoga_yes/len(df)*100:.1f}%)")
    
    print("\n4. DATA QUALITY:")
    total_missing = df.isnull().sum().sum()
    print(f"   • Total missing values: {total_missing} out of {df.shape[0] * df.shape[1]} possible values")
    print(f"   • Data completeness: {((df.shape[0] * df.shape[1] - total_missing) / (df.shape[0] * df.shape[1])) * 100:.1f}%")
    
    print("\n5. RECOMMENDATIONS FOR FURTHER ANALYSIS:")
    print("   • Investigate correlations between symptom severity and demographic factors")
    print("   • Analyze the effectiveness of yoga/meditation on symptom management")
    print("   • Study the relationship between period regularity and symptom patterns")
    print("   • Examine the impact of lifestyle factors on menstrual health")
    
    print("\n" + "=" * 80)
    print("Comprehensive analysis complete! 🎉")
    print("Generated visualizations:")
    print("  • demographic_analysis.png")
    print("  • symptom_severity_chart.png") 
    print("  • symptom_timing_analysis.png")
    print("  • correlation_matrix.png")
    print("=" * 80)

if __name__ == "__main__":
    comprehensive_analysis("DATA SHEET.xlsx")
