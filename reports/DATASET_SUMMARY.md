# Menstrual Health Dataset Analysis Summary

## Dataset Overview
- **Total Participants**: 271 individuals
- **Variables**: 60 columns covering demographics, menstrual cycle patterns, symptoms, and lifestyle factors
- **Data Collection Period**: April 2022 - September 2022
- **Data Completeness**: 96.9% (503 missing values out of 16,260 possible values)

## Key Demographic Findings

### Age Distribution
- **Mean Age**: 20.4 years
- **Age Range**: 12-67 years
- **Median Age**: 19 years
- **Standard Deviation**: 4.3 years

### Physical Characteristics
- **Mean Height**: 156.4 cm (range: 4-189 cm)
- **Mean Weight**: 52.9 kg (range: 30-95 kg)
- **Mean Age of First Period**: 13.0 years (range: 6-17 years)

## Menstrual Cycle Patterns

### Period Regularity
- **Regular Periods**: 226 participants (83.4%)
- **Irregular Periods**: 43 participants (15.9%)

### Cycle Length Distribution
1. **27-29 days**: 87 participants (32.1%) - Most common
2. **24-26 days**: 57 participants (21.0%)
3. **30-32 days**: 52 participants (19.2%)
4. **Less than 24 days**: 21 participants (7.7%)
5. **More than 35 days**: 17 participants (6.3%)

### Period Flow
- **Moderate**: 210 participants (77.5%)
- **Heavy**: 30 participants (11.1%)
- **Light**: 23 participants (8.5%)

## Symptom Analysis

### Most Severe Symptoms (Top 5)
1. **Abdominal/Back Pain**: 1.96 severity (0-3 scale)
2. **Anger/Irritability**: 1.74 severity
3. **Mood Swings**: 1.74 severity
4. **Restlessness**: 1.70 severity
5. **Tension**: 1.52 severity

### Symptom Categories
- **Class A (Emotional)**: Anxiety, Anger, Mood Swings, Nervousness, Restlessness
- **Class B (Mental)**: Tension, Confusion, Forgetfulness, Sleep Issues, Depression
- **Class C (Physical)**: Appetite, Fatigue, Headache, Fainting, Abdominal Pain
- **Class D (Physical Changes)**: Swollen Extremities, Breast Tenderness, Bloating, Weight Gain, Fluid Retention

## Lifestyle and Treatment Patterns

### Work Impact
- **Work Interference**: 152 participants (56.1%) report menstrual problems affecting work
- **No Work Interference**: 116 participants (42.8%)

### Pain Management
- **Pain Killer Usage**: 40 participants (14.8%) used prescribed pain killers in last 3 months
- **No Pain Killers**: 230 participants (84.9%)

### Wellness Practices
- **Yoga/Meditation Practice**: 30 participants (11.1%)
- **No Practice**: 200 participants (73.8%)

## Key Correlations Found

### Strongest Correlations (r > 0.65)
1. **Weight Gain ↔ Fluid Retention**: 0.760
2. **Swollen Extremities ↔ Fluid Retention**: 0.726
3. **Anxiety ↔ Nervousness**: 0.709
4. **Breast Tenderness ↔ Abdominal Bloating**: 0.704
5. **Breast Tenderness ↔ Fluid Retention**: 0.665

## Clinical Insights

### High-Impact Symptoms
- **Abdominal/Back Pain** is the most severe symptom, affecting daily activities
- **Emotional symptoms** (anger, mood swings) are highly prevalent and severe
- **Physical changes** (weight gain, fluid retention) are strongly correlated

### Population Characteristics
- Study population is predominantly young adults (college-aged)
- High percentage (83.4%) have regular menstrual cycles
- Significant work interference reported by majority (56.1%)

### Treatment Gaps
- Low usage of prescribed pain management (14.8%)
- Limited adoption of wellness practices like yoga/meditation (11.1%)
- High work impact suggests need for better symptom management strategies

## Recommendations for Further Research

1. **Intervention Studies**: Investigate effectiveness of yoga/meditation on symptom management
2. **Correlation Analysis**: Explore relationships between demographic factors and symptom severity
3. **Longitudinal Studies**: Track symptom patterns across menstrual cycles
4. **Lifestyle Impact**: Study relationship between period regularity and symptom patterns
5. **Treatment Efficacy**: Evaluate different pain management strategies

## Data Quality Notes

- **High Completeness**: 96.9% data completeness indicates good data quality
- **No Duplicates**: Clean dataset with no duplicate entries
- **Missing Data**: Primarily in optional fields (yoga techniques, additional symptoms)
- **Consistent Formatting**: Well-structured survey responses

---

*Analysis completed using Python with pandas, matplotlib, and seaborn libraries. Visualizations generated include demographic analysis, symptom severity charts, timing analysis, and correlation matrices.*
