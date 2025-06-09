import os
import re
import requests
import whois
import wikipedia
import google.generativeai as genai
import spacy
import torch
import numpy as np
from PIL import Image, ImageChops
from datetime import datetime, timezone
from urllib.parse import urlparse, quote
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration, pipeline
from io import BytesIO
import pytesseract

# ─── Configuration ────────────────────────────────────────────────────────────
NEWS_API_KEY   = "d5a6c90096714e2f92318d2733624f12"
GOOGLE_API_KEY = "AIzaSyCy3t04bcR5kB2f127hljQ7Qkksy9lCaiw"
GOOGLE_CX      = "0245f63ba154c42d9"
GEMINI_API_KEY = "AIzaSyBiIkugHzr82XG9zUWXthVU291HnuoU_SA"
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ─── Load Models ──────────────────────────────────────────────────────────────
sbert = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

# ─── Text Utilities ───────────────────────────────────────────────────────────
def extract_domain(url): return urlparse(url).netloc.lower().removeprefix("www.")

def dynamic_trust_score(domain):
    score = 0.1
    score += {"gov": 0.3, "edu": 0.3, "org": 0.2, "net": 0.1}.get(domain.split(".")[-1], 0.05)
    try:
        w = whois.whois(domain)
        cd = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
        if cd: score += min((datetime.now(timezone.utc) - cd.replace(tzinfo=timezone.utc)).days / 365 * 0.02, 0.3)
    except: pass
    try:
        if wikipedia.search(domain): score += 0.2
    except: pass
    return round(max(0.1, min(score, 1.0)), 2)

def fetch_articles(query):
    articles = []
    try:
        r1 = requests.get(f"https://newsapi.org/v2/everything?q={quote(query)}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}")
        articles += [{"title": a["title"], "url": a["url"], "desc": a.get("description", ""), "publishedAt": a.get("publishedAt")} for a in r1.json().get("articles", [])]
    except: pass
    try:
        r2 = requests.get(f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&q={quote(query)}")
        articles += [{"title": i["title"], "url": i["link"], "desc": i.get("snippet", ""), "publishedAt": None} for i in r2.json().get("items", [])]
    except: pass
    return [dict(t) for t in {tuple(d.items()) for d in articles}]

def extract_keywords(text):
    doc = nlp(text)
    ents = {ent.text.lower() for ent in doc.ents}
    words = {tok.lemma_.lower() for tok in doc if tok.pos_ in ("NOUN", "PROPN", "VERB") and not tok.is_stop}
    return ents | words

def extract_date_or_year(text):
    patterns = [r'\b(\d{4}-\d{1,2}-\d{1,2})\b', r'\b(\d{1,2}/\d{1,2}/\d{4})\b']
    for p in patterns:
        m = re.search(p, text)
        if m: return datetime.fromisoformat(m.group(1).replace("/", "-"))
    y = re.search(r'\b(19|20)\d{2}\b', text)
    return datetime(int(y.group()), 1, 1) if y else None

def verify_with_gemini(claim):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat = model.start_chat()
        response = chat.send_message(f"Fact-check this claim: “{claim}”. Provide a one-sentence verdict and a confidence score (0–1).")
        return response.text.strip()
    except:
        return "⚠️ AI verification unavailable."

def determine_final_verdict(claim_text):
    claim_keywords = extract_keywords(claim_text)
    if not claim_keywords:
        return [], "⚠️ Invalid claim."
    claim_date = extract_date_or_year(claim_text)
    query = " ".join(claim_keywords) + (f" {claim_date.year}" if claim_date else "")
    articles = fetch_articles(query)
    now = datetime.now(timezone.utc)
    claim_emb = sbert.encode(claim_text, convert_to_tensor=True)
    scored = []

    for art in articles:
        emb = sbert.encode(f"{art['title']} {art['desc']}", convert_to_tensor=True)
        sim = float(util.cos_sim(claim_emb, emb).item())
        try:
            pub_date = datetime.fromisoformat((art["publishedAt"] or "").replace("Z", "+00:00"))
            recency = max(0.5, 1.0 - (now - pub_date).days / 14)
        except:
            recency = 0.75
        trust = dynamic_trust_score(extract_domain(art["url"]))
        overlap = len(claim_keywords & extract_keywords(art["title"]))
        scored.append((art["title"], art["url"], sim, recency, trust, overlap))

    scored.sort(key=lambda x: x[2] * x[3] * x[4], reverse=True)
    high_conf = [(t, u, s, r, tr) for t, u, s, r, tr, o in scored if s > 0.6 and tr > 0.8]
    if len(high_conf) >= 2:
        return high_conf, "✅ Multiple high-trust sources confirm this claim as TRUE."
    if not scored:
        return [], verify_with_gemini(claim_text)

    # new scoring logic
    reliable = [x for x in scored if x[2] > 0.7]  # similarity > 0.7
    if len(reliable) >= 2:
        return reliable[:3], "✅ Multiple articles confirm this claim as TRUE."

    refutes = [x for x in scored if x[2] < 0.4]
    if len(refutes) >= 2:
        return refutes[:3], "❌ Multiple articles contradict the claim. Likely FALSE."

    return scored[:3], verify_with_gemini(claim_text)

# ─── Image Consistency Tools ──────────────────────────────────────────────────
def image_text_consistency(image_path, texts):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)
    logits = clip_model(**inputs).logits_per_image
    return logits.softmax(dim=1).detach().numpy()[0]

def error_level_analysis(image_path):
    image = Image.open(image_path).convert('RGB')
    buf = BytesIO(); image.save(buf, format='JPEG', quality=90)
    ela = Image.open(buf); diff = ImageChops.difference(image, ela)
    return max([ex[1] for ex in diff.getextrema()])

def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path).convert('RGB')).strip()

def generate_caption(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = caption_processor(img, return_tensors="pt")
    out = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)

def classify_news_text(text):
    out = classifier(text)[0]
    return "✅ REAL News" if out['label'] == "REAL" else "❌ FAKE News", out['score']

# ─── Main Fake News System ────────────────────────────────────────────────────
def fake_news_detector(input_text=None, image_path=None):
    results = {}

    if image_path:
        ela_score = error_level_analysis(image_path)
        image_text = extract_text_from_image(image_path)
        if not image_text:
            image_text = generate_caption(image_path)
        label, score = classify_news_text(image_text)
        consistency = image_text_consistency(image_path, [input_text or image_text, "This is fake", "Unrelated image"])
        results["image_verdict"] = label
        results["ela_score"] = ela_score
        results["consistency_probs"] = consistency.tolist()
        results["image_text_used"] = image_text

    if input_text:
        articles, verdict = determine_final_verdict(input_text)
        results["text_verdict"] = verdict
        results["evidence_articles"] = articles

    return results
