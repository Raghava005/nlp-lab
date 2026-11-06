import re, io, base64, warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

def clean_text(x):
    x=str(x).lower()
    x=re.sub(r"https?://\S+|www\.\S+"," ",x)
    x=re.sub(r"[^a-z0-9\s']"," ",x)
    return re.sub(r"\s+"," ",x).strip()

def tokens_for(texts):
    return [clean_text(t).split() or ["empty"] for t in texts]

def metric_pack(y_true,y_pred,labels):
    acc=float(accuracy_score(y_true,y_pred))
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
    cm=confusion_matrix(y_true,y_pred,labels=labels)
    report=classification_report(y_true,y_pred,labels=labels,output_dict=True,zero_division=0)
    return {"accuracy":acc,"macro_precision":float(p),"macro_recall":float(r),"macro_f1":float(f),
            "labels":[str(x) for x in labels],"confusion_matrix":cm.tolist(),
            "classification_report":{str(k):{a:float(b) for a,b in v.items()} if isinstance(v,dict) else float(v) for k,v in report.items()}}

def pooled_vectors(token_lists, model, dim):
    out=[]
    for toks in token_lists:
        vecs=[model.wv[w] for w in toks if w in model.wv]
        out.append(np.mean(vecs,axis=0) if vecs else np.zeros(dim))
    return np.asarray(out)

def representation_features(texts, rep):
    if rep=="tfidf":
        vec=TfidfVectorizer(max_features=12000,ngram_range=(1,2),min_df=1,sublinear_tf=True)
        return vec.fit_transform(texts), {"name":"TF-IDF","features":int(len(vec.get_feature_names_out()))}
    toks=tokens_for(texts); dim=100
    if rep=="word2vec":
        model=Word2Vec(sentences=toks,vector_size=dim,window=5,min_count=1,workers=1,epochs=30,seed=42)
        return pooled_vectors(toks,model,dim), {"name":"Word2Vec","dimensions":dim}
    if rep=="fasttext":
        model=FastText(sentences=toks,vector_size=dim,window=5,min_count=1,workers=1,epochs=30,seed=42)
        return pooled_vectors(toks,model,dim), {"name":"FastText","dimensions":dim}
    if rep=="doc2vec":
        docs=[TaggedDocument(words=t, tags=[str(i)]) for i,t in enumerate(toks)]
        model=Doc2Vec(documents=docs,vector_size=dim,window=5,min_count=1,workers=1,epochs=40,seed=42)
        return np.asarray([model.dv[str(i)] for i in range(len(toks))]), {"name":"Doc2Vec","dimensions":dim}
    if rep=="sentence-transformer":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install optional dependencies with: pip install -r requirements-optional.txt")
        model=SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts,show_progress_bar=False), {"name":"Sentence-Transformers (MiniLM)","dimensions":384}
    raise ValueError("Unknown representation. Choose tfidf, word2vec, fasttext, doc2vec or sentence-transformer.")

def classifier_model(name):
    if name=="svm": return SVC(kernel="linear",class_weight="balanced",random_state=42)
    if name=="logistic": return LogisticRegression(max_iter=1500,class_weight="balanced")
    if name=="random-forest": return RandomForestClassifier(n_estimators=250,class_weight="balanced",random_state=42,n_jobs=-1)
    raise ValueError("Unknown classifier. Choose svm, logistic or random-forest.")

def run_classification(df,text_col,label_col,representation,classifier,test_size,seed):
    if text_col not in df or label_col not in df: raise ValueError("Selected text or label column does not exist.")
    work=df[[text_col,label_col]].dropna().copy()
    work[text_col]=work[text_col].astype(str).map(clean_text)
    work=work[(work[text_col].str.len()>0)]
    if work[label_col].nunique()<2: raise ValueError("The label column must contain at least two classes.")
    counts=work[label_col].value_counts()
    if counts.min()<2: raise ValueError("Each class needs at least two examples for a stratified train/test split.")
    if len(work)<10: raise ValueError("Upload at least 10 labeled rows for a useful experiment.")
    X_text=work[text_col].tolist(); y=work[label_col].astype(str).to_numpy()
    X_train_text,X_test_text,y_train,y_test=train_test_split(X_text,y,test_size=test_size,random_state=seed,stratify=y)
    # Fit representations only on training text to avoid vocabulary leakage.
    rep_info={"name":representation,"dimensions":100 if representation in ("word2vec","fasttext","doc2vec") else None}
    if representation=="tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer=TfidfVectorizer(max_features=12000,ngram_range=(1,2),min_df=1,sublinear_tf=True)
        X_train=vectorizer.fit_transform(X_train_text); X_test=vectorizer.transform(X_test_text)
    else:
        # Train embedding model on train corpus; test documents are inferred/averaged using same model.
        toks_train=tokens_for(X_train_text); toks_test=tokens_for(X_test_text); dim=100
        if representation=="word2vec":
            model=Word2Vec(sentences=toks_train,vector_size=dim,window=5,min_count=1,workers=1,epochs=30,seed=seed)
            X_train=pooled_vectors(toks_train,model,dim); X_test=pooled_vectors(toks_test,model,dim)
        elif representation=="fasttext":
            model=FastText(sentences=toks_train,vector_size=dim,window=5,min_count=1,workers=1,epochs=30,seed=seed)
            X_train=pooled_vectors(toks_train,model,dim); X_test=pooled_vectors(toks_test,model,dim)
        elif representation=="doc2vec":
            docs=[TaggedDocument(words=t,tags=[str(i)]) for i,t in enumerate(toks_train)]
            model=Doc2Vec(documents=docs,vector_size=dim,window=5,min_count=1,workers=1,epochs=40,seed=seed)
            X_train=np.asarray([model.dv[str(i)] for i in range(len(toks_train))])
            X_test=np.asarray([model.infer_vector(t,epochs=30) for t in toks_test])
        elif representation=="sentence-transformer":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError: raise ImportError("Install optional dependencies with: pip install -r requirements-optional.txt")
            model=SentenceTransformer("all-MiniLM-L6-v2")
            X_train=model.encode(X_train_text); X_test=model.encode(X_test_text)
        else: raise ValueError("Unsupported representation.")
    model=classifier_model(classifier)
    model.fit(X_train,y_train); pred=model.predict(X_test)
    labels=sorted(np.unique(y).tolist())
    metrics=metric_pack(y_test,pred,labels)
    return {"metrics":metrics,"representation":rep_info,"classifier":classifier,"dataset":{"rows_used":len(work),"train_rows":len(y_train),"test_rows":len(y_test),"classes":labels},
            "predictions":[{"actual":str(a),"predicted":str(b),"correct":bool(a==b)} for a,b in zip(y_test,pred)],
            "charts":{"class_distribution":[{"label":str(k),"count":int(v)} for k,v in work[label_col].value_counts().items()]},
            "notes":["Results are calculated from the uploaded dataset and selected train/test split.","Embeddings are trained or inferred using the training split only."]}

def run_sentence_experiment(df,s1,s2,score,representation,sentences):
    if df is not None:
        for c in (s1,s2):
            if c not in df: raise ValueError(f"Column '{c}' was not found.")
        pairs=df[[s1,s2]+([score] if score and score in df else [])].dropna(subset=[s1,s2])
        a=pairs[s1].astype(str).tolist(); b=pairs[s2].astype(str).tolist()
        human=pairs[score].astype(float).tolist() if score and score in pairs else None
    else:
        if not sentences or len(sentences)<2: raise ValueError("Enter at least two sentences. Sentence pairs are compared in adjacent order.")
        a=sentences[:-1]; b=sentences[1:]; human=None
    if not a: raise ValueError("No valid sentence pairs were found.")
    all_text=a+b
    if representation=="tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec=TfidfVectorizer(ngram_range=(1,2))
        mat=vec.fit_transform(all_text).toarray()
    elif representation=="word2vec":
        toks=tokens_for(all_text); model=Word2Vec(sentences=toks,vector_size=100,min_count=1,workers=1,epochs=40,seed=42)
        mat=pooled_vectors(toks,model,100)
    elif representation=="sentence-transformer":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError: raise ImportError("Install optional dependencies with: pip install -r requirements-optional.txt")
        mat=SentenceTransformer("all-MiniLM-L6-v2").encode(all_text)
    else: raise ValueError("Choose tfidf, word2vec or sentence-transformer.")
    n=len(a); A=mat[:n]; B=mat[n:]
    def cosine(x,y):
        den=np.linalg.norm(x)*np.linalg.norm(y)
        return float(np.dot(x,y)/den) if den else 0.0
    sims=[cosine(x,y) for x,y in zip(A,B)]
    rho=None
    if human and len(human)>1:
        val=spearmanr(human,sims).statistic
        rho=float(val) if np.isfinite(val) else None
    pca_data=[]
    if len(mat)>=2:
        coords=PCA(n_components=2).fit_transform(mat)
        pca_data=[{"x":float(v[0]),"y":float(v[1]),"label":f"Sentence {i+1}"} for i,v in enumerate(coords)]
    return {"representation":{"name":representation,"dimensions":int(mat.shape[1])},"pair_count":n,"mean_cosine_similarity":float(np.mean(sims)),
            "spearman_correlation":rho,"pairs":[{"sentence1":x,"sentence2":y,"cosine_similarity":s} for x,y,s in zip(a,b,sims)],
            "pca":pca_data,"notes":["Cosine similarity measures geometric closeness, not truth or universal semantic equivalence.","PCA is a 2D projection and can distort high-dimensional distances."]}

def run_healthcare(df,text_col,label_col,test_size,epochs,seed):
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError:
        raise ImportError("TensorFlow is required for the real Bi-LSTM. Activate your backend environment and run: pip install -r requirements-optional.txt")
    if text_col not in df or label_col not in df: raise ValueError("Selected text or department label column does not exist.")
    work=df[[text_col,label_col]].dropna().copy()
    work[text_col]=work[text_col].astype(str).map(clean_text)
    work=work[work[text_col].str.len()>0]
    counts=work[label_col].astype(str).value_counts()
    if len(counts)<2 or counts.min()<3: raise ValueError("Each department needs at least 3 examples; use a larger balanced dataset.")
    if len(work)<20: raise ValueError("Upload at least 20 labeled patient-description rows.")
    texts=work[text_col].tolist(); labels=work[label_col].astype(str).tolist()
    Xtr,Xte,ytr,yte=train_test_split(texts,labels,test_size=test_size,random_state=seed,stratify=labels)
    # Fit tokenizer and Word2Vec on training split only.
    tok=Tokenizer(num_words=12000,oov_token="<OOV>"); tok.fit_on_texts(Xtr)
    vocab=min(12000,len(tok.word_index)+1); maxlen=min(60,max(12,int(np.percentile([len(x.split()) for x in Xtr],95))))
    tr=pad_sequences(tok.texts_to_sequences(Xtr),maxlen=maxlen,padding="post",truncating="post")
    te=pad_sequences(tok.texts_to_sequences(Xte),maxlen=maxlen,padding="post",truncating="post")
    w2v=Word2Vec(sentences=tokens_for(Xtr),vector_size=100,window=5,min_count=1,workers=1,epochs=35,seed=seed)
    emb=np.random.normal(0,0.05,(vocab,100)).astype("float32")
    for word,idx in tok.word_index.items():
        if idx<vocab and word in w2v.wv: emb[idx]=w2v.wv[word]
    enc=LabelEncoder(); ytr_i=enc.fit_transform(ytr); yte_i=enc.transform(yte)
    tf.random.set_seed(seed); np.random.seed(seed)
    model=Sequential([Embedding(vocab,100,weights=[emb],input_length=maxlen,trainable=True),Bidirectional(LSTM(48)),Dropout(0.3),Dense(32,activation="relu"),Dense(len(enc.classes_),activation="softmax")])
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    hist=model.fit(tr,ytr_i,validation_split=0.2,epochs=epochs,batch_size=16,verbose=0,callbacks=[EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)])
    probs=model.predict(te,verbose=0); pred=probs.argmax(axis=1)
    metrics=metric_pack(yte_i,pred,list(range(len(enc.classes_))))
    metrics["labels"]=[str(x) for x in enc.classes_]
    # TF-IDF + SVM baseline
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec=TfidfVectorizer(max_features=10000,ngram_range=(1,2))
    xtr=vec.fit_transform(Xtr); xte=vec.transform(Xte)
    svm=SVC(kernel="linear",class_weight="balanced"); svm.fit(xtr,ytr); svm_pred=svm.predict(xte)
    baseline=metric_pack(yte,svm_pred,sorted(set(labels)))
    return {"metrics":metrics,"baseline_metrics":baseline,"dataset":{"rows_used":len(work),"train_rows":len(Xtr),"test_rows":len(Xte),"classes":[str(x) for x in enc.classes_]},
            "training":{"epochs":[i+1 for i in range(len(hist.history["loss"]))],"loss":[float(x) for x in hist.history["loss"]],"val_loss":[float(x) for x in hist.history["val_loss"]],"accuracy":[float(x) for x in hist.history["accuracy"]],"val_accuracy":[float(x) for x in hist.history["val_accuracy"]]},
            "predictions":[{"actual":str(a),"predicted":str(enc.classes_[b]),"correct":bool(enc.classes_[b]==a)} for a,b in zip(yte,pred)],
            "notes":["Word2Vec is trained on the training split; its vectors initialize the Bi-LSTM embedding layer.","This is an educational text classification demonstration, not a clinical decision-support system."]}
