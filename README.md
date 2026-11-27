# Genetic Data Compression & Pattern Visualization System

A Streamlit dashboard and Flask REST API that preprocess a genetic-disorder dataset and compare unsupervised PCA with supervised LDA.

## 1. Project structure

```text
genetic_data_compression_project/
├── app.py                 # Streamlit dashboard
├── api.py                 # Flask API
├── src/
│   ├── __init__.py
│   └── analysis.py        # Shared preprocessing, PCA and LDA logic
├── data/
│   └── genetic_disorders.csv   # Add your Kaggle CSV here
├── outputs/
├── requirements.txt
├── Procfile
├── runtime.txt
└── README.md
```

## 2. Download the dataset

1. Open https://www.kaggle.com/datasets/aibuzz/predict-the-genetic-disorders-datasetof-genomes/data
2. Sign in to Kaggle and download the dataset.
3. Extract the ZIP file.
4. Copy the relevant CSV into the project's `data` folder.
5. Rename it to `genetic_disorders.csv`.

The dashboard also lets you upload a CSV directly. Dataset files are excluded from Git by `.gitignore`; do not commit identifiable patient data to a public repository.

## 3. Open in VS Code

1. Extract this project ZIP.
2. Open VS Code.
3. Select **File → Open Folder** and choose `genetic_data_compression_project`.
4. Open Terminal → New Terminal.

## 4. Install and run (Windows)

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

## 5. Run Flask API (separate terminal)

Activate the same environment, then run:

```bash
python api.py
```

API health check: `http://127.0.0.1:5000/health`

Example request (PowerShell or a REST client):

```powershell
$body = @{
  target_column = "Genetic Disorder"
  n_components = 5
  records = @(
    @{ age = 24; symptom = "Yes"; marker = 1.4; "Genetic Disorder" = "Class A" },
    @{ age = 31; symptom = "No"; marker = 2.1; "Genetic Disorder" = "Class B" },
    @{ age = 28; symptom = "Yes"; marker = 1.8; "Genetic Disorder" = "Class A" }
  )
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:5000/analyze -Method Post -ContentType "application/json" -Body $body
```

Replace the example fields and target-column name with columns from your actual CSV. For meaningful PCA, provide enough rows and at least two varying feature dimensions.

## 6. Deploy Streamlit using GitHub

1. Create a new **private** GitHub repository (recommended for sensitive research data).
2. Push this project folder to the repository. Do not upload patient-level data or secrets.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
5. Deploy. Upload the CSV in the deployed app, or use a suitable private data source.

Git commands from the VS Code terminal:

```bash
git init
git add .
git commit -m "Initial genetic data compression project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own GitHub repository URL. Streamlit Community Cloud deploys the Streamlit dashboard; the Flask API is a separate service and needs its own host if you want it publicly accessible.

## 7. Implementation notes

- Missing numeric values: median imputation.
- Missing categorical values: most-frequent imputation.
- Categorical values: one-hot encoding.
- Numeric values: standard scaling.
- PCA: unsupervised; the class/target column is excluded from its input.
- LDA: supervised; uses the selected target labels and can produce at most `number of classes - 1` discriminant dimensions.
- The recommended PCA component count is the first component count reaching the selected cumulative-variance threshold, within the number of components calculated.

## 8. Critical thinking answer: PCA vs LDA

PCA does not use disorder labels. It finds directions that preserve the largest overall variance in the encoded and standardized data. These directions may reflect age, demographic differences, measurement variation, or other factors rather than the disorder categories.

LDA uses known disorder labels. It searches for directions that maximize separation between classes relative to variation within each class. Therefore, LDA may create a clearer visual separation of known classes even when those classes do not account for the largest overall variation.

PCA is useful for general compression and exploratory structure. LDA is useful for supervised class-discrimination exploration. LDA's apparent separation is not proof of diagnostic performance: validate with held-out data, assess class imbalance and label quality, and avoid data leakage. Neither plot should be treated as a medical diagnosis.

## Important

This is an educational research prototype, not a validated clinical tool. Use de-identified data and follow your institution's privacy, security, and ethics requirements.
