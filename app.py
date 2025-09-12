# app.py
import streamlit as st
from io import BytesIO
import fitz  # PyMuPDF
import re
import requests
from bs4 import BeautifulSoup
import arxiv
import feedparser
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from typing import Tuple, List


st.set_page_config(page_title="Research Assistant", layout="wide")


@st.cache_resource
def load_models():
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")       # embeddings
    # generation model (used for summarization & Q/A). Flan-T5 small is free/open.
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    gen_pipe = pipeline("text2text-generation", model=gen_model, tokenizer=tokenizer, device=-1)
    return emb_model, gen_pipe

emb_model, gen_pipe = load_models()

# -----------------------
# Text / PDF helpers
# -----------------------
from typing import Tuple, List

def extract_text_from_pdf(uploaded_file) -> Tuple[str, List[str]]:
    """Return full text and list of per-page texts."""
    file_bytes = uploaded_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for p in range(doc.page_count):
        page = doc.load_page(p)
        pages.append(page.get_text("text"))
    full_text = "\n".join(pages)
    return full_text, pages


def chunk_text(text, max_chars=1200, overlap=200):
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + max_chars, L)
        chunk = text[start:end]
        # try to end at sentence boundary
        if end < L:
            last_period = chunk.rfind('. ')
            if last_period != -1 and last_period > int(max_chars*0.4):
                end = start + last_period + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = max(end - overlap, end)  # move forward, allow overlap
    return chunks

def extract_keywords(text, top_n=10):
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    if len(sentences) == 0:
        return []
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(sentences)
    sums = X.sum(axis=0)
    terms = vectorizer.get_feature_names_out()
    scores = [(terms[i], float(sums[0, i])) for i in range(len(terms))]
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    keywords = [t for t, s in scores_sorted[:top_n]]
    return keywords

# -----------------------
# Summarization & gen
# -----------------------
def summarize_chunk(chunk, gen_pipe, max_length=120):
    prompt = f"summarize in 3 concise bullet points:\n\n{chunk}"
    out = gen_pipe(prompt, max_length=max_length, do_sample=False)
    summary = out[0]["generated_text"]
    return summary.strip()

def summarize_text_long(text, gen_pipe, chunk_size=1200):
    chunks = chunk_text(text, max_chars=chunk_size, overlap=200)
    chunk_summaries = []
    for c in chunks:
        s = summarize_chunk(c, gen_pipe, max_length=120)
        chunk_summaries.append(s)
    combined = "\n\n".join(chunk_summaries)
    # final compression
    final = gen_pipe(f"summarize in 5 concise bullet points:\n\n{combined}", max_length=180, do_sample=False)[0]["generated_text"]
    return final.strip(), chunk_summaries

# -----------------------
# FAISS index helpers
# -----------------------
def build_faiss_index(texts, emb_model):
    # texts: list[str]
    embeddings = emb_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # normalize for cosine similarity using inner product index
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings

def search_faiss(index, query, emb_model, texts, k=4):
    q_emb = emb_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append({"text": texts[idx], "score": float(score), "index": int(idx)})
    return results

# -----------------------
# Fetch research sources
# -----------------------
def fetch_arxiv(query, max_results=5):
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    items = []
    for r in search.results():
        items.append({
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "summary": r.summary,
            "published": r.published.isoformat(),
            "pdf_url": r.pdf_url,
            "entry_id": r.entry_id
        })
    return items

def fetch_url_text(url):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        # get main text: all <p>
        paras = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        text = "\n".join(paras)
        return text[:20000]  # limit size
    except Exception as e:
        return ""

def fetch_rss_entries(rss_url, max_entries=5):
    feed = feedparser.parse(rss_url)
    items = []
    for i, e in enumerate(feed.entries[:max_entries]):
        items.append({
            "title": e.get("title"),
            "summary": e.get("summary", ""),
            "link": e.get("link")
        })
    return items

# -----------------------
# Generation using retrieved context
# -----------------------
def answer_from_context(prompt, retrieved_texts, gen_pipe, max_length=200):
    # build small context (limit total chars)
    joined = "\n\n".join(retrieved_texts)
    if len(joined) > 8000:
        joined = joined[:8000]
    instruction = f"Use the following documents to answer the question. If answer is not present, say 'Not found in the provided sources.'\n\nDocuments:\n{joined}\n\nQuestion: {prompt}"
    out = gen_pipe(instruction, max_length=max_length, do_sample=False)[0]["generated_text"]
    return out.strip()

# -----------------------
# Streamlit UI
# -----------------------
st.title("🤖 CerebralQuest AI – a quest for knowledge powered by AI")

tab = st.sidebar.radio("Tool", ["PDF Summarizer", "Research Chatbot", "About / Tips"])

if tab == "PDF Summarizer":
    st.header("PDF Summarizer — bullet points + highlighted keywords")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    summary_style = st.selectbox("Summary length", ["Detailed (10-12 bullets)"])
    n_keywords = st.slider("Number of keywords to extract", 3, 20, 8)

    if uploaded:
        with st.spinner("Extracting PDF text..."):
            full_text, pages = extract_text_from_pdf(uploaded)
        st.success("PDF loaded — pages: " + str(len(pages)))
        if st.button("Generate summary"):
            with st.spinner("Summarizing (may take 20-90s first time while models download)..."):
                # choose chunk size by desired length
                chunk_size = 1200 if summary_style == "Short (3 bullets)" else 1700 if summary_style == "Medium (5 bullets)" else 2300
                final_summary, chunk_summaries = summarize_text_long(full_text, gen_pipe, chunk_size=chunk_size)
                keywords = extract_keywords(full_text, top_n=n_keywords)

            st.subheader("🔹 Summary (generated)")
            # display as bullet list (split sentences / lines)
            bullets = re.split(r'[\n\r]+', final_summary)
            if len(bullets) == 1:
                # try splitting by sentences
                bullets = re.split(r'(?<=[\.\?\!])\s+', final_summary)
            for b in bullets:
                if b.strip():
                    st.markdown(f"- {b.strip()}")

            st.subheader("🔸 Extracted keywords")
            st.write(", ".join(keywords))

            # highlight keywords inside the summary (simple)
            highlighted = final_summary
            for kw in keywords[:15]:
                # naive highlight (case-insensitive)
                highlighted = re.sub(fr"(?i)\b({re.escape(kw)})\b", r"<mark>\1</mark>", highlighted)
            st.markdown("**Summary with keywords highlighted:**", unsafe_allow_html=True)
            st.markdown(f"<div style='line-height:1.6'>{highlighted}</div>", unsafe_allow_html=True)

            # download button
            st.download_button("Download summary (.txt)", data=final_summary, file_name="summary.txt", mime="text/plain")

elif tab == "Research Chatbot":
    st.header("Research Chatbot — fetch latest research (arXiv/RSS/URLs) and ask")
    st.write("Select sources, enter topic, optionally add custom prompt. The app will fetch recent docs, index them, and answer from those sources.")

    col1, col2 = st.columns([2,1])
    with col1:
        topic = st.text_input("Topic / query (e.g., 'graph neural networks')", value="")
        custom_prompt = st.text_area("Optional: add a specific question or instruction (e.g., 'Summarize latest results and list top 3 papers')", value="")
        sources = st.multiselect("Sources to use", ["arXiv (recommended)", "RSS feed(s)", "Custom URLs (comma-separated)"], default=["arXiv (recommended)"])
        arxiv_count = st.slider("How many arXiv results to fetch", 1, 20, 5)
    with col2:
        st.info("Tip: use arXiv for CS/ML/Physics preprints. For news use RSS from journals/blogs. For specific sites, paste URLs in the 'Custom URLs' box below.")

    urls_input = st.text_input("Custom URLs (comma-separated)", value="")
    rss_input = st.text_input("RSS feed URLs (comma-separated)", value="")

    if st.button("Fetch & Answer"):
        all_texts = []
        metadata = []

        with st.spinner("Fetching selected sources..."):
            if "arXiv (recommended)" in sources and topic.strip():
                try:
                    arx = fetch_arxiv(topic, max_results=arxiv_count)
                    for a in arx:
                        text = a["summary"] or ""
                        # include title to help retrieval
                        doc_text = f"TITLE: {a['title']}\n\n{a['summary']}\n\nPDF: {a.get('pdf_url','')}"
                        all_texts.append(doc_text)
                        metadata.append({"source":"arXiv", "title": a["title"], "link": a.get("pdf_url","")})
                except Exception as e:
                    st.warning("arXiv fetch failed: " + str(e))

            if "RSS feed(s)" in sources and rss_input.strip():
                for feed in [f.strip() for f in rss_input.split(",") if f.strip()]:
                    items = fetch_rss_entries(feed, max_entries=5)
                    for it in items:
                        doc_text = f"TITLE: {it.get('title')}\n\n{it.get('summary')}\n\nLINK: {it.get('link')}"
                        all_texts.append(doc_text)
                        metadata.append({"source":"RSS", "title": it.get("title"), "link": it.get("link")})

            if "Custom URLs (comma-separated)" in sources and urls_input.strip():
                for u in [u.strip() for u in urls_input.split(",") if u.strip()]:
                    txt = fetch_url_text(u)
                    if txt:
                        doc_text = f"URL: {u}\n\n{txt}"
                        all_texts.append(doc_text)
                        metadata.append({"source":"URL", "title": u, "link": u})

        if len(all_texts) == 0:
            st.warning("No documents fetched. Make sure topic is entered (for arXiv) or URLs / RSS are provided.")
        else:
            with st.spinner("Indexing documents (creating embeddings)..."):
                index, embeddings = build_faiss_index(all_texts, emb_model)

            user_query = custom_prompt.strip() if custom_prompt.strip() else f"Provide a short summary of recent findings on: {topic}"
            with st.spinner("Retrieving relevant documents..."):
                retrieved = search_faiss(index, user_query, emb_model, all_texts, k=6)
                retrieved_texts = [r["text"] for r in retrieved]
            with st.spinner("Generating answer from retrieved docs..."):
                answer = answer_from_context(user_query, retrieved_texts, gen_pipe, max_length=240)

            st.subheader("Answer from retrieved sources")
            st.write(answer)

            st.subheader("Top documents used (by similarity score)")
            for i, r in enumerate(retrieved):
                st.markdown(f"**{i+1}.** (score: {r['score']:.3f}) — Source snippet:")
                st.write(r['text'][:1000] + ("..." if len(r['text'])>1000 else ""))

            st.download_button("Download answer (.txt)", data=answer, file_name="research_answer.txt", mime="text/plain")

elif tab == "About / Tips":
    st.header("About & Tips")
    st.markdown("""
- This app uses free models:
  - Embeddings: `all-MiniLM-L6-v2` (sentence-transformers)  
  - Generation & summarization: `google/flan-t5-small` (transformers).
- **Limitations**: flan-t5-small is light but not as accurate as large proprietary models. Long documents are chunked and summarized — this reduces hallucination but isn't perfect.
- **Data sources**: arXiv is free; scraping other websites may be limited by site rules. Respect robots.txt and site TOS.
- **If models are slow**: you can run this on Google Colab (free GPU) or upgrade models later.
""")

st.caption("Built with ❤️ — ask me if you want this exported into modular files or deployed to Streamlit Cloud.")
