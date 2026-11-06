from __future__ import annotations
import io, os, json, uuid, shutil, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .ml import run_classification, run_sentence_experiment, run_healthcare

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
UPLOADS = DATA / "uploads"
RESULTS = DATA / "results"
REPORTS = DATA / "reports"
for p in (UPLOADS, RESULTS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NLP Lab API", version="1.0.0", description="Interactive NLP experiments based on NLP Case Study – Module 4.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

EXPERIMENTS = [
    {"id":"telecom","name":"Telecom Embedding Comparison","description":"Compare text representations and classifiers for star-rating prediction.","kind":"classification"},
    {"id":"ikn","name":"IKN Sentiment Classification","description":"Classify public comments into sentiment categories and compare models.","kind":"classification"},
    {"id":"sentences","name":"Word-to-Sentence Embeddings","description":"Compare sentence representations, semantic similarity and embedding projections.","kind":"similarity"},
    {"id":"healthcare","name":"Healthcare Word2Vec + Bi-LSTM","description":"Classify patient complaint text into hospital departments using Word2Vec + Bi-LSTM.","kind":"healthcare"},
]

def load_history():
    path = DATA / "history.json"
    if not path.exists(): return []
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return []

def save_result(result):
    rid = result["id"]
    (RESULTS / f"{rid}.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    hist = load_history()
    hist.insert(0, {k: result.get(k) for k in ("id","experiment","created_at","title","summary","status")})
    (DATA / "history.json").write_text(json.dumps(hist[:100], ensure_ascii=False), encoding="utf-8")

def read_dataset(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv": return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"): return pd.read_excel(path)
    if suffix == ".txt":
        return pd.DataFrame({"text":[x.strip() for x in path.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]})
    raise HTTPException(400, "Unsupported file type.")

class ClassificationRequest(BaseModel):
    dataset_id: str
    experiment: str = Field(pattern="^(telecom|ikn)$")
    text_column: str
    label_column: str
    representation: str = "tfidf"
    classifier: str = "svm"
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    random_state: int = 42

class SentenceRequest(BaseModel):
    sentences: list[str] = []
    dataset_id: Optional[str] = None
    sentence1_column: str = "sentence1"
    sentence2_column: str = "sentence2"
    score_column: Optional[str] = "score"
    representation: str = "tfidf"

class HealthcareRequest(BaseModel):
    dataset_id: str
    text_column: str
    label_column: str
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    epochs: int = Field(default=12, ge=2, le=50)
    random_state: int = 42

@app.get("/api/health")
def health():
    return {"status":"ok","service":"NLP Lab API","time":datetime.now(timezone.utc).isoformat()}

@app.get("/api/experiments")
def experiments():
    return EXPERIMENTS

@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    name = Path(file.filename or "dataset").name
    suffix = Path(name).suffix.lower()
    if suffix not in (".csv",".xlsx",".xls",".txt"):
        raise HTTPException(400, "Upload a CSV, XLSX, XLS or TXT file.")
    content = await file.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(413, "File exceeds the 25 MB upload limit.")
    did = uuid.uuid4().hex[:12]
    target = UPLOADS / f"{did}{suffix}"
    target.write_bytes(content)
    try:
        df = read_dataset(target)
    except Exception as e:
        target.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read dataset: {e}")
    return {"dataset_id":did,"filename":name,"rows":int(len(df)),"columns":[str(c) for c in df.columns],"preview":df.head(8).replace({np.nan:None}).to_dict(orient="records")}

@app.get("/api/datasets")
def datasets():
    out=[]
    for p in UPLOADS.iterdir():
        if p.is_file():
            try:
                df=read_dataset(p)
                out.append({"dataset_id":p.stem,"filename":p.name,"rows":len(df),"columns":[str(c) for c in df.columns]})
            except Exception: pass
    return out

def dataset_path(dataset_id):
    matches=list(UPLOADS.glob(f"{dataset_id}.*"))
    if not matches: raise HTTPException(404,"Dataset not found. Upload it again.")
    return matches[0]

@app.post("/api/experiments/classification")
def classification(req: ClassificationRequest):
    df=read_dataset(dataset_path(req.dataset_id))
    try:
        result=run_classification(df, req.text_column, req.label_column, req.representation, req.classifier, req.test_size, req.random_state)
        result.update({"id":uuid.uuid4().hex[:12],"experiment":req.experiment,"title":"Telecom Embedding Comparison" if req.experiment=="telecom" else "IKN Sentiment Classification","created_at":datetime.now(timezone.utc).isoformat(),"status":"completed"})
        result["summary"]=f'{result["metrics"]["accuracy"]*100:.1f}% accuracy · {result["metrics"]["macro_f1"]:.3f} macro-F1'
        save_result(result)
        return result
    except ValueError as e: raise HTTPException(422,str(e))
    except Exception as e: raise HTTPException(500,f"Experiment failed: {e}")

@app.post("/api/experiments/sentences")
def sentences(req: SentenceRequest):
    try:
        if req.dataset_id:
            df=read_dataset(dataset_path(req.dataset_id))
            result=run_sentence_experiment(df=df, s1=req.sentence1_column, s2=req.sentence2_column, score=req.score_column, representation=req.representation, sentences=None)
        else:
            result=run_sentence_experiment(df=None, s1=None, s2=None, score=None, representation=req.representation, sentences=req.sentences)
        result.update({"id":uuid.uuid4().hex[:12],"experiment":"sentences","title":"Word-to-Sentence Embeddings","created_at":datetime.now(timezone.utc).isoformat(),"status":"completed"})
        result["summary"]=f'{result["pair_count"]} sentence pair(s) analyzed'
        save_result(result)
        return result
    except ValueError as e: raise HTTPException(422,str(e))
    except Exception as e: raise HTTPException(500,f"Experiment failed: {e}")

@app.post("/api/experiments/healthcare")
def healthcare(req: HealthcareRequest):
    df=read_dataset(dataset_path(req.dataset_id))
    try:
        result=run_healthcare(df,req.text_column,req.label_column,req.test_size,req.epochs,req.random_state)
        result.update({"id":uuid.uuid4().hex[:12],"experiment":"healthcare","title":"Healthcare Word2Vec + Bi-LSTM","created_at":datetime.now(timezone.utc).isoformat(),"status":"completed"})
        result["summary"]=f'{result["metrics"]["accuracy"]*100:.1f}% test accuracy · {result["metrics"]["macro_f1"]:.3f} macro-F1'
        save_result(result)
        return result
    except ImportError as e: raise HTTPException(501,str(e))
    except ValueError as e: raise HTTPException(422,str(e))
    except Exception as e: raise HTTPException(500,f"Experiment failed: {e}")

@app.get("/api/history")
def history(): return load_history()

@app.get("/api/history/{result_id}")
def history_item(result_id: str):
    p=RESULTS/f"{result_id}.json"
    if not p.exists(): raise HTTPException(404,"Result not found.")
    return json.loads(p.read_text(encoding="utf-8"))

@app.get("/api/reports/{result_id}.pdf")
def report(result_id: str):
    from .reports import create_pdf
    p=RESULTS/f"{result_id}.json"
    if not p.exists(): raise HTTPException(404,"Result not found.")
    result=json.loads(p.read_text(encoding="utf-8"))
    target=REPORTS/f"{result_id}.pdf"
    create_pdf(result,target)
    return FileResponse(target,media_type="application/pdf",filename=f"nlp-lab-{result_id}.pdf")
