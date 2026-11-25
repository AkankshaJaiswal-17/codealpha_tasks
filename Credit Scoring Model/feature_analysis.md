Feature Analysis – L&T Vehicle Loan Default Prediction

1. Target Variable

- "loan_default": Indicates whether the customer defaulted on the first EMI.

2. Potentially Useful Features

- Loan details: disbursed_amount, asset_cost, ltv
- Credit history: PERFORM_CNS.SCORE, PRI and SEC account information
- Employment: Employment.Type
- Credit behaviour: NO.OF_INQUIRIES, DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS
- Credit history duration: AVERAGE.ACCT.AGE, CREDIT.HISTORY.LENGTH

3. Identifier Columns

- UniqueID
- Employee_code_ID

These will be excluded from the initial model because they identify records or employees rather than directly representing customer credit behaviour.

4. Sensitive or Location-Related Features

- Date.of.Birth
- Current_pincode
- MobileNo_Avl_Flag
- Aadhar_flag
- PAN_flag
- VoterID_flag
- Driving_flag
- Passport_flag

These require privacy, fairness and relevance checks before considering their use.

5. Date and Categorical Features

- DisbursalDate
- Employment.Type
- PERFORM_CNS.SCORE.DESCRIPTION
- AVERAGE.ACCT.AGE
- CREDIT.HISTORY.LENGTH

These may require preprocessing or transformation.

6. Data Leakage Consideration

Only information available at the time of loan disbursement should be considered for prediction.

7. Initial Approach

The initial model will use carefully selected loan and credit-history features. Final feature selection will be based on data quality, relevance, fairness and model evaluation.

8. Credit Score and Default Rate Analysis

The relationship between credit score descriptions and loan default was investigated using group-wise default rates.

Key Observations

- M-Very High Risk had an observed default rate of approximately 30.46%.
- B-Very Low Risk had an observed default rate of approximately 13.13%.
- Customers with no bureau history represented 116,950 records, with an observed default rate of approximately 23.13%.
- The results suggest that credit bureau information may be useful for predicting loan default.

Important Considerations

- Customers with no bureau history should not automatically be removed.
- Categories with very few records require cautious interpretation.
- These are observational patterns, not causal conclusions.
- Further analysis and model evaluation are required before making feature-selection decisions.

9. Credit Score Default Rate Visualization

A horizontal bar chart was created to visualize the observed loan default percentage across credit score description categories.

Findings

- M-Very High Risk recorded the highest observed default rate at approximately 30.46%.
- B-Very Low Risk recorded approximately 13.13%.
- No Bureau History Available recorded approximately 23.13%, representing a large group of 116,950 customers.
- The visualization suggests an association between credit score categories and observed loan default rates.

Conclusion

Credit score information appears potentially useful for the prediction task. However, the findings are descriptive and do not establish causation. Categories with very small sample sizes must be interpreted cautiously.

Visualization: "credit_score_default_analysis.png"