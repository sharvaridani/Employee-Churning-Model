# Employee-Churning-Model
📌Introduction

Employee churn, or voluntary attrition, presents major challenges to HR teams—leading to higher costs, talent gaps, and reduced team performance. This project uses machine learning models to analyze employee data and predict the likelihood of churn, enabling companies to take proactive retention steps.

🎯 Problem Statement

Develop a machine learning model that accurately predicts whether an employee is likely to leave an organization. This will assist in reducing turnover costs, improving HR decisions, and enhancing workforce stability.

🎯 Objectives
  
  Predict Churn: Classify whether an employee will leave or stay
  Analyze Key Drivers: Identify key factors influencing churn
  Enable HR Insights: Support strategic HR decisions using predictive data
  Optimize Retention: Help reduce attrition and hiring costs

📁 Dataset Description
 
 Source: Kaggle – HR Analytics Employee Attrition Dataset
 Size: 14,999 records × 10 features
 Key Features:
 satisfaction_level: Job satisfaction score
 number_project: Projects handled
 average_monthly_hours: Workload measure
 time_spend_company: Tenure
 salary, promotion_last_5years, work_accident: Other key indicators

🛠️ Data Preprocessing
  No missing values
  Label Encoding: Converted categorical salary data
  Feature Scaling: Standardized numerical features
  Train-test split: 80% training, 20% testing

📊 Exploratory Data Analysis (EDA)
 
 Correlation heatmaps to detect feature relationships
 Distribution plots to visualize satisfaction, tenure, work hours
 Class imbalance identified (more employees stayed than left)
 Key indicators of churn: satisfaction_level, number_project, time_spend_company

🤖 Machine Learning Models

    Logistic Regression: Baseline model for binary classification
    Decision Tree: Captures non-linear patterns
    Random Forest: Ensemble model for improved accuracy
    SVM (Support Vector Machine): Best performer with high precision and recall

🧪 Evaluation Metrics

    Confusion Matrix:
    TP: 673 | TN: 2259 | FP: 31 | FN: 37

    Other Metrics:
    Accuracy, Precision, Recall, F1-Score

📌 Results
 Best Performing Model: ✅ SVM
 Top Features Identified:
 satisfaction_level
 number_project
 time_spend_company

🌟 Future Impact
   Early Risk Detection: Spot high-risk employees
   Cost Efficiency: Save on hiring and training
   Workforce Stability: Enhance retention and planning
   HR Strategy: Data-driven decision making

✅ Conclusion
The project successfully developed a churn prediction model using real-world HR data. SVM yielded the best performance. The model offers a practical approach to improving employee retention and can be enhanced with more features and larger datasets in the future.
