# PMC Health Resource Optimization

This project focuses on analyzing and optimizing hospital infrastructure and service distribution under the Pune Municipal Corporation (PMC). The analysis leverages data from various health facilities to assess the availability of critical resources like beds, medical staff, and emergency services. By exploring patterns in patient footfall and facility types, the project helps identify potential service gaps and opportunities for better resource allocation across the city's healthcare system.

---

## Project Overview

- **Objective:** Assess the availability and distribution of critical health resources across PMC facilities and identify infrastructure gaps.  
- **Business Question:** How can PMC improve the allocation and utilization of hospital resources to better serve patient demand?  
- **Key Metric:** Bed Occupancy Rate (BOR), calculated from average patient footfall and available beds.  
- **Approach:** Data cleaning, feature engineering, exploratory analysis, and visual insight generation using Python.  

---

## Dataset

- **Source:** PMC Hospital Infrastructure Dataset  
- **Rows:** ~700+  
- **Features:** Facility Type, Class (Public/Private), Beds, Staff (Doctors, Nurses, Midwives), Ambulance Access, Pharmacy Availability, Patient Footfall  
- **Cleaning Actions:** Renamed columns, standardized yes/no values, handled missing values, grouped facility types  

---

## Tools and Technologies

- **Language:** Python  
- **Libraries:** pandas, numpy, seaborn, matplotlib  
- **Environment:** Google Colab / Jupyter Notebook  
- **Methodology:** EDA + Feature Engineering (future-ready for CRISP-DM modeling structure)  

---

## Methodology (Adaptable to CRISP-DM)

1. **Business Understanding:** Understand PMC’s health delivery needs and bottlenecks  
2. **Data Understanding:** Explore structure, types, and completeness of the hospital infrastructure dataset  
3. **Data Preparation:** Rename columns, fix inconsistent entries, clean categorical labels, calculate Bed Occupancy Rate  
4. **EDA:** Use count plots and bar charts to visualize the distribution of services and resources  

---

## Key Insights

- Most facilities are **public hospitals** and **dispensaries**  
- **Ambulance and pharmacy services** are not universally available  
- Some facilities serve **very high patient volumes** despite limited bed capacity  
- Calculated **Bed Occupancy Rate (BOR)** helps highlight potentially overburdened facilities  

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "/content/drive/MyDrive/Customer Loyalty/PMC Hospital Infrastructure.csv"
df = pd.read_csv(file_path)
df.head()

# Rename columns for clarity
df.rename(columns={
    'Type  (Hospital / Nursing Home / Lab)': 'Type',
    'Pharmacy Available : Yes/No': 'Pharmacy',
    'Class : (Public / Private)': 'Class'
}, inplace=True)

df.columns
df.head(2)
df.isnull().sum()

# Drop unnecessary column
df = df.drop('Ward No.', axis=1)
display(df.head())

# Clean string data
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Convert to lowercase
for col in ['City Name', 'Zone Name', 'Ward Name', 'Facility Name', 'Type', ' Class : (Public / Private)', 'Pharmacy', 'Ambulance Service Available']:
    df[col] = df[col].str.lower()

# Standardize 'Type' values
type_replacements = {
    'hospital (maternity home)': 'hospital',
    'nursing home': 'nursing home',
    'lab': 'lab',
    'maternity': 'hospital',
    'general': 'hospital',
    'maternity + general': 'hospital',
    'opthalmology': 'hospital',
    'dental': 'hospital',
    'speciality': 'hospital',
    'ortho': 'hospital'
}
df['Type'] = df['Type'].replace(type_replacements)

# Standardize 'Pharmacy' values
pharmacy_replacements = {
    'yes': 'yes',
    'no': 'no',
    'n.a.': 'no'
}
df['Pharmacy'] = df['Pharmacy'].replace(pharmacy_replacements)

# Standardize 'Ambulance Service Available' values
ambulance_replacements = {
    'yes': 'yes',
    'no': 'no',
    'n.a.': 'no'
}
df['Ambulance Service Available'] = df['Ambulance Service Available'].replace(ambulance_replacements)

display(df.head())
df.head(3)
df.describe()
df.select_dtypes(include='object').describe()

# Visualization: Facility Types
plt.figure(figsize=(10, 6))
sns.countplot(
    data=df,
    y='Type',
    order=df['Type'].value_counts().index,
    palette='magma'
)
plt.title('Distribution of Facility Types')
plt.xlabel('Count')
plt.ylabel('Facility Type')
plt.show()

# Visualization: Public vs Private
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x=' Class : (Public / Private)')
plt.title('Distribution of Facility Class (Public/Private)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Visualization: Pharmacy Availability
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Pharmacy')
plt.title('Distribution of Facilities with Pharmacy Available')
plt.xlabel('Pharmacy Available')
plt.ylabel('Count')
plt.show()

# Visualization: Ambulance Availability
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Ambulance Service Available')
plt.title('Distribution of Facilities with Ambulance Service Available')
plt.xlabel('Ambulance Service Available')
plt.ylabel('Count')
plt.show()

# Average Beds by Facility Type
average_beds_by_type = df.groupby('Type')['Number of Beds in facility type'].mean().round(1).sort_values(ascending=False)
display(average_beds_by_type)

plt.figure(figsize=(10, 6))
average_beds_by_type.head().plot(kind='barh')
plt.title('Top 5 Facility Types by Average Number of Beds')
plt.xlabel('Average Number of Beds')
plt.ylabel('Facility Type')
plt.gca().invert_yaxis()
plt.show()

# Average Footfall by Facility Type
average_footfall_by_type = df.groupby('Type')['Average Monthly Patient Footfall'].mean().round(1).sort_values(ascending=False)
display(average_footfall_by_type)

plt.figure(figsize=(10, 6))
average_footfall_by_type.head().plot(kind='barh')
plt.title('Top 5 Facility Types by Average Monthly Patient Footfall')
plt.xlabel('Average Monthly Patient Footfall')
plt.ylabel('Facility Type')
plt.gca().invert_yaxis()
plt.show()

