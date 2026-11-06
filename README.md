# NLP Lab — Interactive NLP Experiment Platform

A local-first React + FastAPI application built around the four experiments in the supplied NLP Case Study – Module 4.

## Included experiments

1. **Telecom Embedding Comparison** — text/star-rating classification with TF-IDF, Word2Vec, FastText and Doc2Vec representations; Logistic Regression, Random Forest and SVM classifiers; accuracy, macro-F1, classification report and confusion matrix.
2. **IKN Sentiment Classification** — positive/negative sentiment classification with the same representation/classifier comparison workflow.
3. **Word-to-Sentence Embeddings** — compare TF-IDF and Word2Vec sentence vectors, plus optional Sentence-Transformers embeddings; cosine similarity, Spearman correlation when human scores are supplied, and PCA projection.
4. **Healthcare Word2Vec + Bi-LSTM** — department classification from patient complaint text using Word2Vec + a bidirectional LSTM, with TF-IDF + SVM baseline, per-class metrics, confusion matrix and training curves.

The uploaded report provides the experiment framing and reference results. This application trains on the dataset you upload; it does **not** hard-code the report's numbers as newly measured results. Healthcare outputs are educational NLP predictions, not medical advice or diagnoses.

## Requirements

- Python 3.10–3.12 recommended
- Node.js 20+
- 4 GB RAM minimum; 8 GB+ recommended for neural experiments
- Internet is not required after dependencies are installed. Optional transformer models may need a first-time model download.

## Start locally

### 1. Backend

```bash
cd backend
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs

Optional model extras:

```bash
pip install -r requirements-optional.txt
```

`requirements-optional.txt` adds TensorFlow (for the actual Bi-LSTM) and Sentence-Transformers (for contextual sentence embeddings). Gensim is included in the main requirements for Word2Vec/FastText/Doc2Vec.

### 2. Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite (normally http://localhost:5173). The frontend calls the API at `http://127.0.0.1:8000`.

## Dataset formats

Upload CSV, XLSX, XLS or TXT files from the Dataset Manager.

- **Classification experiments:** one row per example. Select a text column and a label column. For Telecom, the label can be a star rating (for example `1`–`5`). For IKN, labels should represent sentiment (for example `positive` / `negative`). For Healthcare, labels should be department names.
- **Sentence embeddings:** upload a CSV with `sentence1` and `sentence2`; optionally include a numeric `score` column containing human similarity ratings. You can also enter sentence pairs manually.
- TXT files are read as one text example per non-empty line; classification labels must then be supplied in a CSV/XLSX file.

## Important implementation notes

- Models are trained on the uploaded dataset and evaluated with a stratified train/test split when possible.
- Small datasets may not contain enough examples per class for a reliable split. The API returns a clear validation error rather than pretending to produce a meaningful score.
- Word2Vec, FastText and Doc2Vec are trained on the uploaded corpus. They are not pretrained embeddings.
- BERT/Sentence-Transformers is optional and may require a model download.
- TensorFlow is optional to keep the default installation lighter. The Healthcare Bi-LSTM endpoint explains how to install the optional dependency if it is missing.
- Experiment history and uploaded datasets are stored locally in `backend/data/`. This is a single-user local demo; add authentication, access controls, storage quotas and secure deployment configuration before exposing it publicly.
- Uploaded files are restricted by extension and size (25 MB default). Do not upload confidential patient records or personally identifiable data.

## API overview

- `GET /api/health`
- `GET /api/experiments`
- `POST /api/datasets/upload`
- `GET /api/datasets`
- `POST /api/experiments/classification`
- `POST /api/experiments/sentences`
- `POST /api/experiments/healthcare`
- `GET /api/history`
- `GET /api/history/{id}`
- `GET /api/reports/{id}.pdf`

## Suggested cloud deployment

Deploy the FastAPI backend and React static frontend separately. Set a production CORS allowlist, persistent storage for `backend/data`, HTTPS, authentication, rate limits, upload scanning and a secrets manager. For large model workloads, use a worker queue and persistent model/artifact storage. Do not expose the development server directly to the public internet.
