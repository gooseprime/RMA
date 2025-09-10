# Menstrual Health: Comprehensive Analysis & Recommendations

## 1) Executive Summary
- Study of 271 participants, 60 variables (demographics, cycle, symptoms, lifestyle)
- Data completeness: 96.9% (503 missing out of 16,260)
- Two distinct symptom clusters discovered (high vs moderate severity)
- Strong correlations among physical-change symptoms; emotional–mental symptoms also correlate strongly
- Yoga/meditation associated with measurable symptom improvements (0.2–0.4 severity points on average)

## 2) Dataset Overview
- Timeframe: Apr–Sep 2022
- Numerical columns: 25; categorical: 34; datetime: 1
- No duplicate rows

### Key Descriptive Stats
- Age: mean 20.4 (range 12–67), SD 4.3
- Height: mean 156.4 cm
- Weight: mean 52.9 kg
- Age of first period: mean 13.0 years
- Period regularity: 83.4% Yes
- Most common cycle length: 27–29 days (32.1%)
- Period flow: Moderate (77.5%), Heavy (11.1%), Light (8.5%)

## 3) Visual Overview
- demographic_analysis.png
- symptom_severity_chart.png
- symptom_timing_analysis.png

## 4) Statistical Tests and Metrics
### 4.1 Correlation Highlights (severity scores)
Top associations (r):
- Weight gain ↔ Fluid retention: 0.760
- Swollen extremities ↔ Fluid retention: 0.726
- Anxiety ↔ Nervousness: 0.709
- Breast tenderness ↔ Abdominal bloating: 0.704
- Breast tenderness ↔ Fluid retention: 0.665
- Anger ↔ Mood swings: 0.655
- Confusion ↔ Forgetfulness: 0.654
(See correlation_matrix.png and severity_correlation_matrix.png)

### 4.2 Chi‑Square Tests (categorical timing variables)
- Attempted broad set of inter-class tests; many contingency tables violated assumptions (expected < 5).
- Where testable, no robust class-level categorical associations persisted after checks.
- We therefore emphasize correlation-based severity analysis and ML-driven structure.

## 5) Advanced Pattern Discovery
### 5.1 Clustering (K-Means)
- Optimal k (silhouette): 2
- Cluster 1 (n=139): higher severity; top: abdominal/back pain, anger, mood swings, restlessness, sleep difficulty
- Cluster 2 (n=132): moderate severity; top: abdominal/back pain, mood swings, restlessness, anger, tension
- Visuals: symptom_clusters_2d.png

### 5.2 Hierarchical Clustering
- Symptom grouping dendrogram: symptom_dendrogram.png

### 5.3 PCA (Principal Component Analysis)
- 80% variance with ~8 components; 90% with ~13
- Component 1 (~51.1% variance): anxiety, appetite, nervousness, bloating, fatigue
- Component 2 (~9.5%): mood/anger vs fluid retention/weight gain
- Visual: pca_analysis.png

### 5.4 Factor Analysis (5 factors)
- F1: Anxiety, Nervousness, Fluid retention, Appetite, Fatigue
- F2: Anger, Mood swings, (vs) Weight gain, Fluid retention
- F3: Restlessness/Nervousness (vs) Tension/Depression
- F4: Forgetfulness, Confusion (cognitive) with mood/restlessness cross‑loadings
- F5: Mood swings with fatigue/tension component

## 6) Yoga/Meditation Effectiveness
- Practitioners: 30 (11.1%); Non-practitioners: 241 (88.9%)
- 14 symptoms show meaningful improvement among practitioners (Δ ≈ 0.2–0.4)
- Largest improvements: Confusion (0.41), Fluid retention (0.39), Sleep difficulty (0.37), Bloating (0.36), Swollen extremities (0.36), Anxiety (0.34), Depression (0.33)
- Visuals: yoga_effectiveness.png, yoga_effectiveness_detailed.png

### Cluster-based effectiveness
- Cluster 1: Avg improvement ≈ 0.07 (Recommended)
- Cluster 2: Limited evidence overall benefit
- Visual: cluster_recommendations.png

## 7) Recommendations
### 7.1 Clinical/Programmatic
- Treat by clusters, not isolated symptoms; integrate mental and physical care
- Prioritize anxiety/sleep management due to centrality in PCA and improvements with yoga
- Use correlations to bundle symptoms (e.g., fluid retention–weight–swollen extremities)

### 7.2 Yoga/Meditation Protocols
- Anxiety & Nervousness: Pranayama, Mindfulness meditation, Yoga Nidra, Gentle Hatha (15–30 min, AM/PM)
- Anger & Mood Swings: Kundalini, heart‑opening poses, loving‑kindness meditation, chanting, Yin (20–45 min)
- Sleep: Yoga Nidra, legs‑up‑the‑wall, Savasana, 4‑7‑8 breathing (10–20 min pre‑bed)
- Physical Pain/Fatigue: Restorative yoga, gentle stretching, body scan (15–30 min)
- Physical Changes (bloating/weight): Twists, core, digestive flows, dynamic sequences (20–40 min, AM)

### 7.3 Weekly Plan (example)
- Mon: Sun Salutations, Warrior, breath (30m, AM)
- Tue: Gentle Hatha, meditation, pranayama (25m, PM)
- Wed: Strength/balance/core (35m, AM)
- Thu: Yin/restorative/stretch (20m, PM)
- Fri: Heart‑opening + loving‑kindness/chanting (30m, AM)
- Sat: Yoga Nidra + body scan (45m, afternoon)
- Sun: Gentle flow + meditation + journaling (25m, AM)

## 8) Limitations
- Chi‑square assumption violations for several categorical comparisons
- Self-reported symptoms; cross‑sectional snapshot; limited practitioner sample (n=30)
- Severity scale (0–3) may compress variance

## 9) Files Produced
- DATASET_SUMMARY.md, STATISTICAL_ANALYSIS_REPORT.md, ADVANCED_ANALYSIS_REPORT.md
- Visuals: demographic_analysis.png, symptom_severity_chart.png, symptom_timing_analysis.png, correlation_matrix.png, severity_correlation_matrix.png, symptom_dendrogram.png, pca_analysis.png, symptom_clusters_2d.png, yoga_effectiveness.png, yoga_effectiveness_detailed.png, cluster_recommendations.png

## 10) Takeaways
- Anxiety‑centric structure explains largest share of symptom variance
- Clear high‑severity cluster suggests targeted, multi‑modal interventions
- Yoga/meditation shows measurable, practical benefits across mental and physical domains

## 11) Graphs & Explanations (Gallery)

### 11.1 Demographic Overview
![Demographic Analysis](demographic_analysis.png)
Explanation:
- Shows distributions for Age, Height vs Weight, Age of First Period, and Period Duration.
- Age is concentrated in late teens to early 20s; first period around 13 years; typical period length clustered at 3–5 days.

### 11.2 Symptom Severity (Averages)
![Symptom Severity Chart](symptom_severity_chart.png)
Explanation:
- Bars show mean severity (0–3) per symptom.
- Abdominal/Back Pain is highest on average; emotional symptoms (anger, mood swings, restlessness) are also elevated.

### 11.3 Symptom Timing (Before vs During)
![Symptom Timing Analysis](symptom_timing_analysis.png)
Explanation:
- Pie charts summarize when symptoms occur across classes.
- Many emotional/physical symptoms peak during the period; several also present before.

### 11.4 Correlation Matrix (All Variables)
![Correlation Matrix](correlation_matrix.png)
Explanation:
- Heatmap of pairwise correlations across numerical severity variables.
- Strongest blocks appear among physical-change symptoms (fluid retention, weight gain, swelling) and among emotional/mental symptoms.

### 11.5 Correlation Matrix (Severity Scores)
![Severity Correlation Matrix](severity_correlation_matrix.png)
Explanation:
- Focused heatmap for severity scores only; clearer view of symptom-to-symptom associations.
- Top pairs include Weight Gain–Fluid Retention, Swollen Extremities–Fluid Retention, Anxiety–Nervousness.

### 11.6 Symptom Dendrogram (Hierarchical Clustering)
![Symptom Dendrogram](symptom_dendrogram.png)
Explanation:
- Tree shows hierarchical grouping of symptoms based on similarity.
- Physical changes cluster together; cognitive/emotional items (confusion, forgetfulness, mood) group as a subcluster.

### 11.7 PCA Analysis
![PCA Analysis](pca_analysis.png)
Explanation:
- Left: variance explained by each component; Right: cumulative variance (80% ~ 8 PCs).
- PC1 (≈51%) is dominated by anxiety-like and fatigue/digestive symptoms, indicating a broad “distress/physiologic” dimension.

### 11.8 Symptom Clusters (2D Projection)
![Symptom Clusters 2D](symptom_clusters_2d.png)
Explanation:
- K-Means (k=2) shown in PCA 2D space; two clusters separate by overall severity/mix.
- Cluster 1 exhibits higher composite severity; target for priority interventions.

### 11.9 Yoga Effectiveness (Summary)
![Yoga Effectiveness](yoga_effectiveness.png)
Explanation:
- Bars show improvement (non‑practitioner mean – practitioner mean). Positive = better with yoga.
- Largest gains: confusion, fluid retention, sleep difficulty, bloating, anxiety, depression.

### 11.10 Yoga Effectiveness (Detailed Top 10)
![Yoga Effectiveness Detailed](yoga_effectiveness_detailed.png)
Explanation:
- Top 10 symptoms ranked by improvement magnitude; supports targeted yoga prescriptions.

### 11.11 Cluster Recommendations
![Cluster Recommendations](cluster_recommendations.png)
Explanation:
- Left: cluster sizes; Right: top 5 symptoms by cluster with mean severity.
- Guides cluster‑specific yoga/meditation and clinical planning.

### 11.12 Class → Technique Family Heatmap
![Class to Technique Heatmap](class_to_technique_heatmap.png)
Explanation:
- Rows are symptom classes (A–D); columns are yoga technique families.
- Cell value estimates impact (higher = stronger observed improvement among practitioners mapped to that class).
- Emotional/Mental classes align more with Breath/Meditation/Gentle Yoga; Physical/Changes align with Dynamic/Gentle.

### 11.13 Class → Technique Bubble Chart
![Class to Technique Bubble](class_to_technique_bubble.png)
Explanation:
- Same mapping as the heatmap; bubble size/color encodes estimated impact.
- Quick way to spot strongest class-to-technique patterns at a glance.

### 11.14 Named Techniques (Top 10 by Improvement)
![Technique Impact Top 10](technique_impact_top10.png)
Explanation:
- Ranks named techniques reported by participants by their average observed improvement.
- Useful for communicating concrete practices (e.g., Pranayama, Yoga Nidra, Kundalini Yoga).

### 11.15 Classes → Families Scaled by Techniques
![Classes to Families Scaled](class_to_family_scaled_by_techniques.png)
Explanation:
- Class-to-family impact heatmap further weighted by observed effects of named techniques.
- Emphasizes families with both class alignment and technique-level evidence.

## 12) Questionnaire → Dataset Mapping
Source: Questionnaire.pdf

- Timestamp → `Timestamp`
- Participant identity → `Name`, `Candidate name`
- Demographics (Date/Age/Height/Weight) → `Date`, `Age`, `Height`, `Weight`
- Menstrual history → `1. Age of First period`, `2. Do you have regular periods (Y/N)`, `3. Regular intervals between periods`, `4. How heavy...`, `5. How long...`
- Class A items (timing + severity) → `CLASS  A [...]` and corresponding `... SEVERITY [...]`
- Class B items (timing + severity) → `CLASS B [...]` and corresponding `... SEVERITY [...]`
- Class C items (timing + severity) → `CLASS  C [...]` and corresponding `... SEVERITY [...]`
- Class D items (timing + severity) → `CLASS D  [...]` and corresponding `... SEVERITY [...]`
- Work interference / cramps / pelvic pain / pain killers → columns 53–59 per sheet
- Yoga/Meditation practice & technique & duration → `10. Do you practice any Yoga or Meditation (Y/N)`, `If yes, name of the technique`, `Duration of Practicing each day`

Notes:
- Minor text normalization was applied (e.g., trailing spaces, capitalization, synonyms) for consistent analysis.
- Timing questions (BEFORE/DURING) appear in `CLASS ...` columns; severities (0–3) in `... SEVERITY [...]` columns.
