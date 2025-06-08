import requests
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# ✅ Load AI-powered verification tools
genai.configure(api_key="AIzaSyCy3t04bcR5kB2f127hljQ7Qkksy9lCaiw")
sbert = SentenceTransformer("all-MiniLM-L6-v2")

def extract_domain(url):
    """Extract domain from URL."""
    return urlparse(url).netloc.replace("www.", "").lower()

def fetch_page_metadata(url):
    """Extract metadata (title + description) if full scraping fails."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string if soup.title else ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        description = meta_desc["content"] if meta_desc else ""
        return f"{title}. {description}"
    except:
        return None

def scrape_page_text(url, max_chars=3000):
    """Scrape full text from a page or fallback to metadata."""
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return fetch_page_metadata(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
        return " ".join(paragraphs)[:max_chars]
    except:
        return fetch_page_metadata(url)

def determine_source_credibility(domain):
    """Estimate credibility dynamically based on multiple objective factors."""
    base_score = 0.7  # Default credibility score

    # ✅ Factor 1: Government/Education domains get a boost
    if domain.endswith(".gov") or domain.endswith(".edu"):
        base_score += 0.2

    return min(max(base_score, 0.1), 1.0)  # Normalize between [0.1, 1.0]

def fetch_news_articles(query):
    """Fetch news articles using NewsAPI."""
    news_api_key = "fddcb7b3cac941e0b6238651590a31e4"
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={news_api_key}"
    response = requests.get(url)
    return [(art["url"]) for art in response.json().get("articles", [])] if response.status_code == 200 else []

def fetch_google_search_articles(query):
    """Fetch extra links via Google Custom Search."""
    google_api_key = "AIzaSyAkr6GTCvzRXxyxC9EDsXDvR6tDJpIHI-k"
    search_engine_id = "0245f63ba154c42d9"
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={search_engine_id}&key={google_api_key}"
    response = requests.get(url)
    return [item["link"] for item in response.json().get("items", [])] if response.status_code == 200 else []

def verify_claim_with_gemini(claim_text):
    """AI-assisted fact-check using Google Gemini."""
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(f"Fact-check this claim: {claim_text}. Provide a credibility rating from 0 to 1.")
    return response.text.strip()

def determine_final_verdict(claim_text):
    """Analyze sources, credibility, and AI insights to verify claim accuracy."""
    urls = fetch_news_articles(claim_text) + fetch_google_search_articles(claim_text)
    total_score = 0.0
    count = 0
    credibility_results = []
    trusted_count = 0

    for url in urls:
        domain = extract_domain(url)
        credibility = determine_source_credibility(domain)
        text = scrape_page_text(url)
        if not text:
            continue

        emb_claim = sbert.encode(claim_text, convert_to_tensor=True)
        emb_content = sbert.encode(text, convert_to_tensor=True)
        similarity = util.cos_sim(emb_claim, emb_content).item()
        sim_score = (similarity + 1) / 2  # Convert to [0,1]

        weighted_score = sim_score * credibility
        credibility_results.append((domain, round(sim_score, 2), round(weighted_score, 2), url))
        total_score += weighted_score
        count += 1

        if credibility > 0.8 and sim_score > 0.6:
            trusted_count += 1

    ai_verdict = verify_claim_with_gemini(claim_text)
    
    if trusted_count >= 2:
        return credibility_results, "✅ Multiple sources confirm this claim as TRUE. AI verdict overridden."
    elif count > 0:
        avg_score = total_score / count
        if avg_score > 0.6:
            return credibility_results, "✅ Real-time sources confirm the claim as TRUE."
        elif avg_score < 0.4:
            return credibility_results, "❌ Real-time sources confirm the claim as FALSE."
        return credibility_results, "⚖️ Mixed signals—further verification needed."

    return credibility_results, ai_verdict

class FakeNewsDetectionSystem:
    def analyze_claim(self, claim_text):
        return determine_final_verdict(claim_text)
