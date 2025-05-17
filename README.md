#  AI-Powered Healthcare Facility Insights  
### Rating Prediction, Inconsistency Auditing & Governance Readiness  
**Using CMS Hospital Compare Data**

---

##  Goal  
To build a healthcare-focused AI/ML and business intelligence (BI) pipeline that:
- Analyzes hospital quality metrics such as mortality, readmission, and patient experience
- Flags high-performing hospitals
- Detects inconsistencies and missing data for audit readiness
- Supports data-driven planning for VHA Home HealthCare or similar health networks

---

## Intended Audience  
- Business Intelligence Analysts  
- Healthcare Data Scientists  
- Hospital Operations & Strategy Teams  
- Public Sector and Non-Profit Health Decision Makers  
- AI Researchers in Health Informatics  

---

##  Problem Statement  
VHA Home HealthCare needs a scalable, interpretable analytics system to:
- Benchmark hospital quality across the country  
- Identify partners with proven performance  
- Detect anomalies in reporting  
- Drive operational improvements and compliance efforts  

---

##  Dataset  
- **Source**: [CMS Hospital General Information](https://data.cms.gov/provider-data)
- **Format**: CSV  
- **Fields Used**:
  - Hospital overall rating  
  - Mortality national comparison  
  - Readmission national comparison  
  - Patient experience national comparison  
  - Hospital Type, Ownership, City

---

##  Strategy & Pipeline (All in One Code Block)

```python
# Step 1: Import Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the Dataset
df = pd.read_csv("/content/HospInfo.csv")

# Step 3: Preprocess and Clean Data
df.columns = df.columns.str.strip()
df["Hospital overall rating"] = pd.to_numeric(df["Hospital overall rating"], errors="coerce")

# Step 4: KPI Summary – Hospital Ratings
rating_summary = df["Hospital overall rating"].value_counts().sort_index()
print("Hospital Ratings Summary:\n", rating_summary)

# Step 5: Visualize Mortality Comparison
plt.figure(figsize=(10, 4))
sns.countplot(y="Mortality national comparison", data=df,
              order=df["Mortality national comparison"].value_counts().index)
plt.title("Mortality Comparison")
plt.xlabel("Number of Hospitals")
plt.tight_layout()
plt.show()

# Step 6: Flag High-Performing Hospitals
df["High Performer"] = (
    df["Readmission national comparison"] == "Below the national average"
) & (
    df["Mortality national comparison"] == "Below the national average"
)
print(df[df["High Performer"]][["Hospital Name", "High Performer"]])

# Step 7: Missing Rating Check
missing_ratings = df[df["Hospital overall rating"].isnull()]
print("Hospitals with Missing Ratings:\n", missing_ratings[["Hospital Name", "Hospital overall rating"]])

# Step 8: Inconsistency Audit (Governance)
inconsistencies = df[
    (df["Readmission national comparison"] == "Above the national average") &
    (df["Mortality national comparison"] == "Below the national average")
]
print("Inconsistent Records:\n", inconsistencies[[
    "Hospital Name", "Readmission national comparison", "Mortality national comparison"
]])

# Step 9: Patient Experience Summary
experience_summary = df["Patient experience national comparison"].value_counts()
print("Patient Experience Summary:\n", experience_summary)

# Step 10: Facility Scope – Cities and Ownership Types
print("Number of Unique Cities:", df["City"].nunique())
print("Ownership Types:", df["Hospital Ownership"].unique())

##  Challenges

    Many hospitals (~1245) have missing ratings

    Inconsistent entries where readmission is above national average but mortality is below (131 cases)

    Patient experience data often marked as “Not Available”

    Ownership and location data is non-standard and inconsistent

## Machine Learning Extensions (Optional)

    Build a classifier to predict high-performing hospitals

    Use unsupervised clustering for healthcare facility segmentation

    Build dashboards with Streamlit or Power BI

    Add model explainability with SHAP or LIME

## Conceptual Enhancement – AGI Vision

This pipeline can evolve into an AGI-aligned system that:

    Reads new benchmarks from peer-reviewed journals

    Integrates clinical policy updates autonomously

    Uses LLMs to summarize hospital performance for non-technical leaders

    Acts as a reasoning engine for health data policy and risk forecasting
