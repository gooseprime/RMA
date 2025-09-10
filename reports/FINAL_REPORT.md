# Menstrual Health Symptomatology: An Integrated Statistical and Machine Learning Examination with Yoga/Meditation Interventions

Author: Anvin P Shibu  
Affiliation: Department of Computer Science and Engineering (CSE)  
Date: 2025-09-03


## Abstract
We present a comprehensive analysis of menstrual health symptomatology in a cohort of 271 participants, integrating classical statistics, multivariate modeling, and unsupervised machine learning. We quantify correlational structure among symptom severities, evaluate categorical associations, and discover latent dimensions and patient clusters. Leveraging self-reported yoga/meditation practice, we estimate intervention-linked symptom improvements and translate findings into pragmatic, cluster-aware recommendations. The study reveals two robust symptom clusters, a dominant anxiety-centric dimension explaining the majority of variance, and measurable improvements associated with specific yoga/meditation modalities. We contribute an end-to-end analytical pipeline and a set of decision-support visualizations intended for both clinical and research use.


## 1. Introduction
Menstrual health presents a complex interaction of physical, emotional, and cognitive symptoms whose co-expression varies across individuals and time. This study aims to (1) characterize the multivariate structure of symptoms, (2) discover latent factors and patient phenotypes, and (3) evaluate real-world effectiveness signals from yoga/meditation practice reported by participants. Building on these findings, we propose cluster- and factor-informed intervention strategies.


## 2. Data & Instrumentation
- Source: Survey-derived dataset of N = 271 participants (April–September 2022).  
- Instrument: Items were drawn from a structured questionnaire (Questionnaire.pdf), covering demographics, menstrual history, symptom timing (before/during), symptom severities (0–3), and lifestyle/treatment (e.g., yoga/meditation, pain-killers).
- Variables: 60 columns (25 numerical, 34 categorical, 1 datetime).  
- Completeness: 96.9% (503 missing of 16,260 fields).  
- Duplicates: None detected.

Questionnaire → dataset mapping is provided in Section 10.


## 3. Cohort Overview
- Age: mean 20.4 (SD 4.3), range 12–67.  
- Height: mean 156.4 cm; Weight: mean 52.9 kg.  
- Age at menarche: mean 13.0 years.  
- Cycle regularity: 83.4% report regular cycles; modal cycle length 27–29 days (32.1%).  
- Flow: Predominantly moderate (77.5%).

Figure 1 summarizes the cohort’s demographics and distributions.

![Figure 1. Demographic Distributions](demographic_analysis.png)
Caption: Age distribution, height–weight scatter, age at menarche, and period duration histograms. The cohort is dominated by late-adolescent/young-adult participants. Menarche concentrates near 13 years; period duration clusters between 3–5 days.


## 4. Symptom Landscape
We analyze average severity (0–3 scale) across 20 symptoms spanning four classes (A: Emotional, B: Mental, C: Physical, D: Physical Changes). Abdominal/back pain presents the highest average severity, followed by prominent emotional symptoms (anger, mood swings, restlessness).

![Figure 2. Mean Symptom Severities](symptom_severity_chart.png)
Caption: Mean severity per symptom. Pain and emotional dysregulation (anger, mood swings) rank highly; sleep difficulty is also notable.

Symptom timing indicates that many symptoms peak during the period, while several also occur premenstrually.

![Figure 3. Symptom Timing (Before vs During)](symptom_timing_analysis.png)
Caption: Distribution of symptom occurrence timing across classes A–D.


## 5. Correlational Structure
Pairwise correlations among severity scores reveal strong clustering within physical-change items and substantial co-movement among emotional/mental symptoms.

![Figure 4. Global Correlation Matrix](correlation_matrix.png)
![Figure 5. Severity-Only Correlation Matrix](severity_correlation_matrix.png)
Caption: Heatmaps of pairwise Pearson correlations. Exemplars include: Weight gain ↔ Fluid retention (r = 0.760), Swollen extremities ↔ Fluid retention (r = 0.726), Anxiety ↔ Nervousness (r = 0.709), Breast tenderness ↔ Abdominal bloating (r = 0.704).

Interpretation: Physical retention/weight-change phenomena form a coherent subnetwork; anxiety-related constructs form another, suggesting shared mechanisms or co-precipitating processes.


## 6. Categorical Associations (Chi-Square)
We attempted extensive chi-square tests across symptom-timing categories. Many contingency tables failed standard assumptions (e.g., >20% expected counts <5), limiting inferential power. We therefore treat categorical associations cautiously and prioritize continuous severities for structural inference.


## 7. Unsupervised Structure Discovery
### 7.1 Clustering (K-Means)
Silhouette analysis identified k = 2 as optimal. Two patient phenotypes emerged:
- Cluster 1 (n = 139): Higher composite severity; salient features include abdominal/back pain, anger, mood swings, restlessness, and sleep difficulty.  
- Cluster 2 (n = 132): Moderate severity profile; similar symptom set at reduced intensity.

![Figure 6. Symptom Clusters in 2D (PCA Projection)](symptom_clusters_2d.png)
Caption: K-Means clusters in the first two principal components. Separation aligns with global severity and symptom mix.

### 7.2 Hierarchical Clustering of Symptoms

![Figure 7. Symptom Dendrogram](symptom_dendrogram.png)
Caption: Ward-linkage dendrogram indicates natural symptom groupings. Physical-change symptoms aggregate; cognitive–emotional items cluster together.

### 7.3 Principal Component Analysis (PCA)
PC1 explains ≈51.1% of variance and loads on anxiety, nervousness, fatigue, appetite, and abdominal bloating—an overarching “distress/physiologic reactivity” dimension. 80% cumulative variance is obtained with ~8 PCs; 90% with ~13 PCs.

![Figure 8. PCA Variance Profiles](pca_analysis.png)
Caption: Explained and cumulative variance curves. High PC1 dominance indicates a strong general factor with secondary dimensions capturing specific symptom families.


## 8. Yoga/Meditation Effectiveness Signals
Participants reporting yoga/meditation practice (11.1%) exhibit measurable improvements vs non-practitioners, particularly in confusion (Δ ≈ 0.41), fluid retention (0.39), sleep difficulty (0.37), abdominal bloating (0.36), swollen extremities (0.36), anxiety (0.34), and depression (0.33).

![Figure 9. Yoga/Meditation Effectiveness (Summary)](yoga_effectiveness.png)
![Figure 10. Yoga/Meditation Effectiveness (Top 10)](yoga_effectiveness_detailed.png)
Caption: Average improvement computed as non-practitioner mean minus practitioner mean (positive = benefit). Results support clinically meaningful improvements across cognitive/emotional and physical-change domains.


## 9. From Patterns to Practice: Technique-Level and Class–Technique Mappings
We synthesize symptom classes with technique families and named techniques to inform actionable prescriptions.

### 9.1 Class → Technique Family Patterns

![Figure 11. Class-to-Technique Family Heatmap](class_to_technique_heatmap.png)
![Figure 12. Class-to-Technique Family Bubble Map](class_to_technique_bubble.png)
Caption: Estimated impact by symptom class and technique family. Emotional/Mental favor Breath, Meditation, Gentle Yoga; Physical/Changes align with Dynamic and Gentle regimens.

### 9.2 Named Techniques: Observed Improvements

![Figure 13. Named Techniques: Top 10 by Improvement](technique_impact_top10.png)
Caption: Cleaned named techniques ranked by observed participant-level improvement (e.g., Pranayama, Yoga Nidra, Kundalini Yoga). This provides concrete practice-level guidance beyond families.

### 9.3 Family Patterns Scaled by Technique Effects

![Figure 14. Classes → Families Scaled by Named Techniques](class_to_family_scaled_by_techniques.png)
Caption: Class–family matrix upweighted by observed technique-level effects; reinforces emphasis on Breath/Meditation for emotional/mental domains and Dynamic/Gentle for physical-change symptomatology.


## 10. Questionnaire → Dataset Mapping (Operationalization)
- Timestamp → `Timestamp`  
- Participant identity → `Name`, `Candidate name`  
- Demographics → `Date`, `Age`, `Height`, `Weight`  
- Menstrual history → `1. Age of First period`, `2. Do you have regular periods (Y/N)`, `3. Regular intervals between periods`, `4. How heavy...`, `5. How long...`  
- Class A timing/severity → `CLASS  A [...]` and corresponding `... SEVERITY [...]`  
- Class B timing/severity → `CLASS B [...]` and corresponding `... SEVERITY [...]`  
- Class C timing/severity → `CLASS  C [...]` and corresponding `... SEVERITY [...]`  
- Class D timing/severity → `CLASS D  [...]` and corresponding `... SEVERITY [...]`  
- Functioning and treatment → `6. Work responsibilities (Y/N)`, `7. Cramps`, `8. Pelvic pain frequency`, `9. Pain killers (Y/N)`  
- Yoga/meditation → `10. Do you practice... (Y/N)`, `If yes, name of the technique`, `Duration of Practicing each day`  

Note: Minimal normalization resolved trailing spaces, capitalization inconsistencies, and common synonyms to align analysis-ready fields with questionnaire semantics.


## 11. Discussion
This study delineates a clear multivariate organization of menstrual health symptoms, dominated by an anxiety-centric general factor with specific substructures reflecting physical-change processes and cognitive/emotional dynamics. The discovery of two patient clusters suggests practical pathways for tailoring care intensity. Yoga/meditation appears associated with meaningful symptom reductions, particularly for cognitive/emotional strain and fluid-retention–related phenomena.

Methodologically, the convergence of correlational analysis, PCA, factor analysis, and clustering strengthens confidence in the structural findings. While chi-square analyses were limited by contingency assumptions, the continuous severity space provided robust signal. The class–technique mapping emerges as a pragmatic bridge from structure to intervention selection.


## 12. Limitations
- Observational design; causality cannot be inferred.  
- Self-report bias may affect symptom and practice reporting.  
- Practitioner group is modest (≈11%), introducing sampling uncertainty.  
- Severity scale (0–3) compresses variance; finer-grained measures could improve sensitivity.  
- Chi-square tests constrained by sparse cells; categorical inference remains tentative.


## 13. Conclusions and Recommendations
- Clinical: Adopt cluster-aware, multimodal management; prioritize anxiety and sleep hygiene given their centrality and responsiveness.  
- Interventions: Emphasize Breath (Pranayama) and Meditation (Mindfulness, Yoga Nidra) for emotional/mental burdens; deploy Gentle/Restorative for pain/fatigue; use Dynamic sequences (twists, core, digestive flows) for bloating/weight/retention domains.  
- Research: Extend to longitudinal tracking; test protocolized, cluster-specific yoga regimens; integrate biomarkers to probe mechanisms.


## 14. Advanced Methodological Extensions (For Deeper Understanding)
To augment robustness, interpretability, and causal insight, the following extensions are recommended:

- Causal Inference & Design
  - Propensity score modeling (matching/weighting), inverse probability of treatment weighting (IPTW), and doubly robust estimators to isolate yoga effects under selection bias.
  - Sensitivity analyses for unmeasured confounding; E-values; negative control outcomes/exposures.
  - Prospective designs: stepped-wedge or randomized encouragement to strengthen causal identification.

- Longitudinal & Temporal Modeling
  - Repeated-measures designs across cycles; mixed-effects (multilevel) models for within-person change.
  - State-space/HMMs or sequence mining to capture temporal symptom trajectories and transitions.

- Latent Variable & Structural Modeling
  - Confirmatory factor analysis (CFA) and structural equation modeling (SEM) to validate latent constructs and directional pathways (e.g., anxiety → sleep → pain).
  - Latent class/profile analysis to refine phenotype discovery beyond K-Means.

- Modern ML & Explainability
  - Nonlinear embeddings (UMAP/t-SNE) for manifold structure; density clustering (HDBSCAN) for irregular clusters.
  - Supervised models (gradient boosting, random forests) with SHAP for symptom importance, subgroup heterogeneity, and treatment-response prediction.
  - Stability selection and bootstrapping for model robustness; nested cross-validation to reduce optimism.

- Network & Graphical Models
  - Symptom networks (Gaussian Graphical Models, partial correlations) to uncover conditional dependences.
  - Community detection to identify tightly coupled symptom groups; intervention targeting on network hubs.

- Robustness, Missing Data, and Measurement
  - Multiple imputation with chained equations; robust estimators for heavy tails/outliers.
  - Item Response Theory (IRT) to evaluate psychometrics of symptom scales; reliability/validity indices.

- Multimodal Data Integration
  - Incorporate wearables (sleep/HRV/activity), nutrition, and hormonal/biomarker panels to link physiology with symptom clusters.
  - Text mining on free-text responses (if any) via topic modeling or transformers for qualitative signals.

- Reproducibility & Governance
  - Pre-registration of hypotheses; versioned pipelines; data audits; FAIR data principles.

These extensions will increase causal credibility, refine phenotyping, and improve personalization of recommendations.


## References
- Survey instrument: Questionnaire.pdf (in repository).  
- Analysis stack: Python (pandas, scipy, scikit-learn, seaborn, matplotlib).


## Figures (Quick Index)
1. Demographic Distributions – demographic_analysis.png  
2. Mean Symptom Severities – symptom_severity_chart.png  
3. Symptom Timing – symptom_timing_analysis.png  
4. Global Correlation Matrix – correlation_matrix.png  
5. Severity Correlation Matrix – severity_correlation_matrix.png  
6. Symptom Clusters (PCA Projection) – symptom_clusters_2d.png  
7. Symptom Dendrogram – symptom_dendrogram.png  
8. PCA Variance Profiles – pca_analysis.png  
9. Yoga Effectiveness (Summary) – yoga_effectiveness.png  
10. Yoga Effectiveness (Top 10) – yoga_effectiveness_detailed.png  
11. Class→Technique Family Heatmap – class_to_technique_heatmap.png  
12. Class→Technique Family Bubble – class_to_technique_bubble.png  
13. Named Techniques Top 10 – technique_impact_top10.png  
14. Classes→Families Scaled by Techniques – class_to_family_scaled_by_techniques.png  
