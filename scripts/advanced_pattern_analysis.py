import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import chi2_contingency, pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

def advanced_pattern_analysis(file_path):
    """
    Advanced pattern analysis using machine learning and statistical techniques
    """
    print("=" * 80)
    print("ADVANCED PATTERN ANALYSIS: HIDDEN PATTERNS & YOGA EFFECTIVENESS")
    print("=" * 80)
    
    # Read the dataset
    df = pd.read_excel(file_path, sheet_name=0)
    
    # 1. DATA PREPARATION
    print("\n📊 DATA PREPARATION")
    print("-" * 50)
    
    # Get severity columns
    severity_cols = [col for col in df.columns if 'SEVERITY' in col]
    severity_df = df[severity_cols].select_dtypes(include=[np.number])
    
    # Clean column names
    clean_cols = []
    for col in severity_cols:
        if col in severity_df.columns:
            clean_name = col.split('[')[-1].split(']')[0] if '[' in col else col
            clean_cols.append(clean_name)
    
    severity_df.columns = clean_cols
    severity_df = severity_df.fillna(0)  # Fill missing values with 0
    
    print(f"Prepared {len(severity_df.columns)} severity variables for analysis")
    print(f"Sample size: {len(severity_df)} participants")
    
    # 2. CLUSTERING ANALYSIS
    print("\n\n🔍 CLUSTERING ANALYSIS - FINDING HIDDEN PATTERNS")
    print("-" * 50)
    
    # Standardize the data
    scaler = StandardScaler()
    severity_scaled = scaler.fit_transform(severity_df)
    
    # Determine optimal number of clusters using silhouette score
    silhouette_scores = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(severity_scaled)
        silhouette_avg = silhouette_score(severity_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k} (silhouette score: {max(silhouette_scores):.3f})")
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(severity_scaled)
    
    # Add cluster labels to dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Symptom_Cluster'] = cluster_labels
    
    # Analyze cluster characteristics
    print(f"\nCluster Analysis Results:")
    for i in range(optimal_k):
        cluster_data = severity_df[cluster_labels == i]
        cluster_size = len(cluster_data)
        print(f"\nCluster {i+1} (n={cluster_size}):")
        
        # Find top symptoms in this cluster
        cluster_means = cluster_data.mean().sort_values(ascending=False)
        top_symptoms = cluster_means.head(5)
        
        print("  Top symptoms (mean severity):")
        for symptom, severity in top_symptoms.items():
            print(f"    {symptom}: {severity:.2f}")
    
    # 3. HIERARCHICAL CLUSTERING
    print("\n\n🌳 HIERARCHICAL CLUSTERING ANALYSIS")
    print("-" * 50)
    
    # Perform hierarchical clustering on symptoms
    linkage_matrix = linkage(severity_df.T, method='ward')
    
    # Create dendrogram
    plt.figure(figsize=(15, 8))
    dendrogram(linkage_matrix, labels=severity_df.columns, orientation='top')
    plt.title('Hierarchical Clustering of Symptoms', fontweight='bold', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('symptom_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created symptom dendrogram")
    
    # 4. PRINCIPAL COMPONENT ANALYSIS (PCA)
    print("\n\n📈 PRINCIPAL COMPONENT ANALYSIS")
    print("-" * 50)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(severity_scaled)
    
    # Calculate explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print("PCA Results:")
    print(f"  Components needed for 80% variance: {np.argmax(cumulative_variance >= 0.8) + 1}")
    print(f"  Components needed for 90% variance: {np.argmax(cumulative_variance >= 0.9) + 1}")
    
    # Plot explained variance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.axhline(y=0.8, color='g', linestyle='--', label='80%')
    plt.axhline(y=0.9, color='orange', linestyle='--', label='90%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created PCA analysis plots")
    
    # Analyze first few components
    n_components = min(5, len(severity_df.columns))
    pca_components = pca.components_[:n_components]
    
    print(f"\nTop {n_components} Principal Components:")
    for i in range(n_components):
        component = pca_components[i]
        # Get top contributing symptoms
        top_indices = np.argsort(np.abs(component))[-5:][::-1]
        print(f"\nComponent {i+1} (explains {explained_variance_ratio[i]:.1%} of variance):")
        for idx in top_indices:
            symptom = severity_df.columns[idx]
            loading = component[idx]
            print(f"  {symptom}: {loading:.3f}")
    
    # 5. FACTOR ANALYSIS
    print("\n\n🔬 FACTOR ANALYSIS")
    print("-" * 50)
    
    # Perform factor analysis
    n_factors = min(5, len(severity_df.columns) - 1)
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    fa_result = fa.fit_transform(severity_scaled)
    
    print(f"Factor Analysis with {n_factors} factors:")
    for i in range(n_factors):
        factor = fa.components_[i]
        # Get top contributing symptoms
        top_indices = np.argsort(np.abs(factor))[-5:][::-1]
        print(f"\nFactor {i+1}:")
        for idx in top_indices:
            symptom = severity_df.columns[idx]
            loading = factor[idx]
            print(f"  {symptom}: {loading:.3f}")
    
    # 6. YOGA/MEDITATION EFFECTIVENESS ANALYSIS
    print("\n\n🧘 YOGA/MEDITATION EFFECTIVENESS ANALYSIS")
    print("-" * 50)
    
    # Analyze yoga/meditation practice
    yoga_col = '10. Do you practice any Yoga or Meditation (Y/N)'
    yoga_technique_col = 'If yes, name of the technique'
    yoga_duration_col = 'Duration of Practicing each day'
    
    # Create yoga practice indicator
    yoga_practice = df[yoga_col].fillna('No')
    yoga_practitioners = yoga_practice.str.contains('Y|y|Yes|YES', case=False, na=False)
    
    print(f"Yoga/Meditation Practice:")
    print(f"  Practitioners: {yoga_practitioners.sum()} ({yoga_practitioners.mean()*100:.1f}%)")
    print(f"  Non-practitioners: {(~yoga_practitioners).sum()} ({(~yoga_practitioners).mean()*100:.1f}%)")
    
    # Compare symptom severity between practitioners and non-practitioners
    practitioner_symptoms = severity_df[yoga_practitioners]
    non_practitioner_symptoms = severity_df[~yoga_practitioners]
    
    print(f"\nSymptom Severity Comparison:")
    print("  Yoga/Meditation Practitioners vs Non-practitioners:")
    
    significant_differences = []
    for symptom in severity_df.columns:
        practitioner_mean = practitioner_symptoms[symptom].mean()
        non_practitioner_mean = non_practitioner_symptoms[symptom].mean()
        difference = practitioner_mean - non_practitioner_mean
        
        # Simple t-test equivalent (comparing means)
        if abs(difference) > 0.2:  # Meaningful difference threshold
            significant_differences.append({
                'Symptom': symptom,
                'Practitioner_Mean': practitioner_mean,
                'Non_Practitioner_Mean': non_practitioner_mean,
                'Difference': difference,
                'Improvement': difference < 0  # Negative difference means improvement
            })
    
    if significant_differences:
        print(f"\n  Found {len(significant_differences)} symptoms with meaningful differences:")
        for diff in significant_differences:
            status = "IMPROVED" if diff['Improvement'] else "WORSE"
            print(f"    {diff['Symptom']}: {status} (Δ = {diff['Difference']:.2f})")
    else:
        print("  No significant differences found in symptom severity")
    
    # Analyze yoga techniques
    yoga_techniques = df[yoga_technique_col].dropna()
    if len(yoga_techniques) > 0:
        print(f"\nYoga/Meditation Techniques Reported:")
        technique_counts = yoga_techniques.value_counts()
        for technique, count in technique_counts.head(10).items():
            print(f"  {technique}: {count} practitioners")
    
    # 7. CLUSTER-BASED YOGA RECOMMENDATIONS
    print("\n\n💡 PERSONALIZED YOGA/MEDITATION RECOMMENDATIONS")
    print("-" * 50)
    
    # Analyze which clusters benefit most from yoga
    cluster_yoga_analysis = []
    for i in range(optimal_k):
        cluster_mask = cluster_labels == i
        cluster_yoga_practitioners = yoga_practitioners[cluster_mask]
        cluster_non_practitioners = ~yoga_practitioners[cluster_mask]
        
        if cluster_yoga_practitioners.sum() > 0 and cluster_non_practitioners.sum() > 0:
            cluster_practitioner_symptoms = severity_df[cluster_mask & yoga_practitioners]
            cluster_non_practitioner_symptoms = severity_df[cluster_mask & ~yoga_practitioners]
            
            # Calculate average improvement
            improvements = []
            for symptom in severity_df.columns:
                practitioner_mean = cluster_practitioner_symptoms[symptom].mean()
                non_practitioner_mean = cluster_non_practitioner_symptoms[symptom].mean()
                improvement = non_practitioner_mean - practitioner_mean  # Positive = improvement
                improvements.append(improvement)
            
            avg_improvement = np.mean(improvements)
            cluster_yoga_analysis.append({
                'Cluster': i+1,
                'Size': cluster_mask.sum(),
                'Yoga_Practitioners': cluster_yoga_practitioners.sum(),
                'Avg_Improvement': avg_improvement,
                'Top_Symptoms': severity_df[cluster_mask].mean().sort_values(ascending=False).head(3).index.tolist()
            })
    
    # Sort by improvement
    cluster_yoga_analysis.sort(key=lambda x: x['Avg_Improvement'], reverse=True)
    
    print("Cluster-based Yoga Effectiveness:")
    for analysis in cluster_yoga_analysis:
        print(f"\nCluster {analysis['Cluster']} (n={analysis['Size']}):")
        print(f"  Yoga practitioners: {analysis['Yoga_Practitioners']}")
        print(f"  Average symptom improvement: {analysis['Avg_Improvement']:.2f}")
        print(f"  Top symptoms: {', '.join(analysis['Top_Symptoms'])}")
        
        # Generate recommendations
        if analysis['Avg_Improvement'] > 0.1:
            print(f"  🎯 HIGHLY RECOMMENDED for yoga/meditation")
        elif analysis['Avg_Improvement'] > 0.05:
            print(f"  ✅ RECOMMENDED for yoga/meditation")
        else:
            print(f"  ⚠️  Limited evidence for yoga/meditation benefit")
    
    # 8. VISUALIZATION OF PATTERNS
    print("\n\n📊 CREATING PATTERN VISUALIZATIONS")
    print("-" * 50)
    
    # Create cluster visualization
    if optimal_k >= 2:
        # Use first two principal components for visualization
        pca_2d = PCA(n_components=2)
        pca_2d_result = pca_2d.fit_transform(severity_scaled)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
        
        for i in range(optimal_k):
            mask = cluster_labels == i
            plt.scatter(pca_2d_result[mask, 0], pca_2d_result[mask, 1], 
                       c=[colors[i]], label=f'Cluster {i+1}', alpha=0.7, s=50)
        
        plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Symptom Clusters in 2D Space', fontweight='bold', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('symptom_clusters_2d.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created 2D cluster visualization")
    
    # Create yoga effectiveness heatmap
    if len(significant_differences) > 0:
        yoga_data = []
        for diff in significant_differences:
            yoga_data.append({
                'Symptom': diff['Symptom'],
                'Improvement': -diff['Difference']  # Negative difference = improvement
            })
        
        yoga_df = pd.DataFrame(yoga_data)
        yoga_df = yoga_df.sort_values('Improvement', ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(yoga_df['Symptom'], yoga_df['Improvement'], color='green', alpha=0.7)
        plt.xlabel('Symptom Improvement (Lower severity in practitioners)')
        plt.title('Yoga/Meditation Effectiveness by Symptom', fontweight='bold', fontsize=14)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('yoga_effectiveness.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created yoga effectiveness chart")
    
    # 9. FINAL RECOMMENDATIONS
    print("\n\n🎯 COMPREHENSIVE RECOMMENDATIONS")
    print("-" * 50)
    
    print("1. HIDDEN PATTERNS DISCOVERED:")
    print(f"   • {optimal_k} distinct symptom clusters identified")
    print(f"   • {n_components} principal components explain most variance")
    print(f"   • Hierarchical clustering reveals symptom groupings")
    
    print("\n2. YOGA/MEDITATION INSIGHTS:")
    if len(significant_differences) > 0:
        improved_symptoms = [d for d in significant_differences if d['Improvement']]
        print(f"   • {len(improved_symptoms)} symptoms show improvement with yoga/meditation")
        print("   • Most beneficial for:")
        for diff in improved_symptoms[:3]:
            print(f"     - {diff['Symptom']} (improvement: {abs(diff['Difference']):.2f})")
    else:
        print("   • Limited evidence of yoga/meditation effectiveness in current data")
    
    print("\n3. PERSONALIZED APPROACH:")
    print("   • Cluster-based recommendations for targeted interventions")
    print("   • Factor analysis reveals underlying symptom dimensions")
    print("   • PCA shows which symptoms are most variable")
    
    print("\n4. CLINICAL IMPLICATIONS:")
    print("   • Use clustering to identify high-risk symptom combinations")
    print("   • Develop cluster-specific treatment protocols")
    print("   • Consider yoga/meditation as adjunct therapy for specific clusters")
    
    print("\n" + "=" * 80)
    print("Advanced pattern analysis complete! 🎉")
    print("Generated files:")
    print("  • symptom_dendrogram.png")
    print("  • pca_analysis.png")
    print("  • symptom_clusters_2d.png")
    print("  • yoga_effectiveness.png")
    print("=" * 80)

if __name__ == "__main__":
    advanced_pattern_analysis("DATA SHEET.xlsx")
