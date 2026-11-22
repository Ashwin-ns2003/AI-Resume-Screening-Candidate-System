# AI-Powered Resume Screening & Candidate Selection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

A machine learning system that predicts whether a candidate should be **hired or rejected** from resume details (skills, certifications, experience, projects, education, job role and salary expectation), and **explains each decision** using LIME.

## Problem Statement
Screening a large number of resumes by hand is slow, inconsistent and prone to bias. This project automates the first screening step and shows recruiters *why* a candidate was selected or rejected, so the decision can be understood and justified.

## Dataset
- 1,000 candidate records with 11 columns: skills, experience (years), education, certifications, job role, recruiter decision, salary expectation, projects count and AI score
- Target: `Recruiter Decision` (Hire = 1, Reject = 0)
- Source: [add dataset name and link here]

## Approach
1. **Data cleaning:** filled missing `Certifications` (274 rows) with "No Certification"; standardised column names
2. **EDA:** class balance, salary and score distributions, education vs decision, experience vs score, correlation heatmap
3. **Feature engineering:**
   - Text: skills + certifications combined and converted with **TF-IDF** (top 1,000 terms)
   - Numeric: experience years, projects count, salary expectation
   - Categorical: education and job role, **one-hot encoded**
   - Dropped `resume_id`, `name` and `ai_score` so the model does not learn from identifiers or a score that leaks the outcome
4. **Modelling:** 80/20 train-test split, then trained and compared three classifiers
5. **Explainability:** **LIME** shows which features pushed each prediction towards Hire or Reject
6. **Demo app:** an interactive **Gradio** interface that takes candidate details and returns the decision with its explanation

## Results
Evaluated on a held-out test set of 200 resumes:

| Model | Accuracy |
|---|---|
| **Logistic Regression** | **98.0%** |
| Random Forest | 96.5% |
| Decision Tree | 96.0% |

Logistic Regression (final model) classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Reject (0) | 0.94 | 0.98 | 0.96 | 46 |
| Hire (1) | 0.99 | 0.98 | 0.99 | 154 |
| Weighted avg | 0.98 | 0.98 | 0.98 | 200 |

Confusion matrix: 45 rejects and 151 hires correctly classified; 1 reject wrongly predicted as hire and 3 hires wrongly predicted as reject.

## Tools and Technologies
Python, Pandas, NumPy, Scikit-learn, TF-IDF, Logistic Regression, Random Forest, Decision Tree, LIME, Gradio, Matplotlib, Seaborn

## How to Run
1. Click the **Open in Colab** button at the top of the notebook
2. Upload the dataset CSV to the Colab session and make sure the file path in the `pd.read_csv(...)` cell matches it
3. Run all cells in order. The last code cell launches the Gradio app

## Limitations and Future Improvements
- The dataset is small (1,000 records) and evaluated with a single train-test split. Cross-validation would give a more reliable estimate
- Use a scikit-learn `Pipeline` so TF-IDF and encoding are fitted on training data only
- Replace TF-IDF with sentence embeddings (e.g. BERT) so similar skills are understood by meaning
- Add PDF resume upload and automatic skill extraction
- Check the model for bias before any real-world use. It should assist recruiters, not replace them
