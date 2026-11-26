# CodeAlpha Internship — Daily Work Log

## Day 1 — 20 November 2026
- Started Machine Learning internship.
- Set up the Python development environment.
- Installed essential ML libraries.
- Explored the assigned project tasks.

## Day 2 — 21 November 2026
- Created a sample credit scoring dataset.
- Practised Python file handling.
- Organized project files.

## Day 3 — 22 November 2026
- Practised NumPy and Pandas.
- Explored dataset operations.
- Tested Python scripts.
- Organized the credit scoring project folder.

## Day 4 — 23 November 2026
- Explored the Credit Scoring dataset.
- Identified input features and target variable.
- Practised Exploratory Data Analysis (EDA) using Pandas.
- Checked dataset shape, information and data types.
- Checked missing values and duplicate records.
- Generated descriptive statistics using describe().
- Created and executed day4_eda.py successfully.

### Key Learning
Understood the basic steps of inspecting and
understanding a dataset before model development.

Dataset Selection – Research and Decision History

1. Initial Dataset Considered: German Credit Dataset

Initially, the German Credit Dataset was considered for the Credit Scoring project.

- It contains approximately 1,000 records and 20 features.
- It is a commonly used educational dataset for credit risk classification.
- It provides an opportunity to practise classification algorithms and model evaluation.

Reason for not selecting it:

Since this internship project is being developed in an Indian context, we wanted a dataset that would be more relevant to the Indian financial and lending environment.

2. Alternative Dataset Considered: LoanTap Dataset

The LoanTap dataset was subsequently explored as another candidate.

During data inspection, we examined its columns, missing values, numerical statistics and unusual values.

However, its geographical context and data provenance need further verification before using it as the final Indian dataset.

3. Final Dataset Selection Criteria

The final dataset should:

- Have a clearly documented source.
- Be relevant to the Indian lending environment.
- Contain sufficient records and useful financial features.
- Have a clearly defined prediction target.
- Be suitable for educational machine learning experiments.

4. Learning Takeaway

Dataset selection is an important part of a machine learning project. A dataset should not be selected only because it is popular or easy to use. Its source, features, target variable and relevance to the problem must also be evaluated.

Final Dataset Candidate – L&T Vehicle Loan Default Prediction

After researching multiple datasets, L&T Vehicle Loan Default Prediction was identified as the strongest candidate for the Credit Scoring Model.

Dataset details:

- Source: Kaggle
- Original context: L&T Financial Services vehicle-loan default prediction challenge
- Records: Approximately 233,154
- Variables: 41
- Target variable: "loan_default"
- Problem type: Binary classification

Reason for selection:
The dataset focuses on predicting whether a borrower will default on the first EMI. It contains borrower information, loan details and credit-bureau history, making it relevant to the Credit Scoring internship task.

License consideration:
Kaggle displays the license as "Other (specified in description)". The available description does not clarify redistribution permissions. Therefore, dataset redistribution rights remain unverified. Raw data will not be published in the GitHub repository.

Learning outcome:
Dataset selection requires checking the prediction target, source, geographical context, feature relevance, data quality and usage permissions rather than selecting a dataset based only on its size or popularity.

Day 5 – Credit Dataset Understanding and Analysis

Project: Credit Scoring Model

Work Completed

1. Inspected the L&T Vehicle Loan Default Prediction dataset.
2. Analysed dataset dimensions, data types, missing values and duplicate records.
3. Studied numerical and categorical statistics.
4. Examined the data dictionary and classified features.
5. Investigated credit score distributions and descriptions.
6. Calculated default percentages across credit score categories.
7. Created a horizontal bar chart showing default rates by credit score description.

Key Findings

- Dataset contains 233,154 records and 41 columns.
- Target variable: "loan_default".
- Default cases: approximately 21.71%.
- Missing values: 7,661, all in "Employment.Type".
- Duplicate records: 0.
- Customers without bureau history: 116,950.
- Highest observed default rate: M-Very High Risk (30.46%).
- B-Very Low Risk had an observed default rate of approximately 13.13%.

Files Created

- "day5_inspect_lnt.py"
- "day5_quality_lnt.py"
- "day5_statistics_lnt.py"
- "feature_analysis.md"
- "credit_score_default_analysis.png"

Learning Outcomes

Learned exploratory data analysis, feature classification, target distribution analysis, group-wise default rate calculation and data visualization using Pandas and Matplotlib.

Next Step

Begin data preprocessing after completing further feature and data-quality checks.

## Day 6 - 25 Nov 2026

### Duplicate Record Analysis
- Initially, the original dataset contained 0 duplicate rows.
- After removing identifier columns, 8 duplicate feature combinations were identified (16 rows involved).
- These records were retained because different customers may have identical feature values.
- Removing them could lead to unnecessary data loss.combinations were identified (16 rows involved).

These records were retained because different customers
may have identical feature values. Removing them could
result in unnecessary data loss.

### Target Variable Analysis
- Total records: 233,154
- No Default (0): 182,543 (78.29%)
- Default (1): 50,611 (21.71%)
- Observation: The dataset has class imbalance.
- Decision: Accuracy alone will not be sufficient
  for evaluating the classification model.

  ### Day 6 – Data Preprocessing
- Handled missing Employment.Type values using Unknown.
- Converted date columns into datetime format.
- Corrected future DOB interpretations.
- Created Age feature at disbursement.
- Removed UniqueID, Employee_code_ID and Date.of.Birth.
- Checked missing values: 0 remaining.
- Identified 8 duplicate feature combinations (16 rows involved); retained them.
- Analysed target distribution and identified class imbalance.
- Saved and verified train_preprocessed.csv.
- Final dataset shape: 233154 rows × 39 columns.

Day 7 – Feature Engineering

- Identified numerical, categorical, date and duration features.
- Converted average account age into months.
- Converted credit history length into months.
- Extracted month and year from disbursal dates.
- Investigated categorical feature cardinality.
- Compared credit score values with their descriptions.
- Analysed default rates across credit-score groups.
- Created "Has_Normal_Bureau_Score" as a derived feature.
- Verified all four newly engineered features.
- Confirmed zero missing values in the new features.
- Saved the engineered dataset locally.

Dataset: 233,154 rows × 44 columns.

## Daily Learning
More updates will be added as the internship progresses.