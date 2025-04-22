import os
import sys
import json
import logging
import importlib
from typing import List, Dict, Any
from mimetypes import guess_extension
from urllib.parse import urlparse, urlunparse

import streamlit as st
from dotenv import load_dotenv
from pocketgroq import GroqProvider
from groq import Groq, APIError
import validators
from requests.exceptions import RequestException
import html2text
from bs4 import BeautifulSoup
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Replace with your actual path

# --- Constants ---
MAX_CONTEXT_LENGTH = 100000  # Limit for session state context
MAX_PAGES_LIMIT = 50  # Limit for crawling pages
MODEL_NAME = "llama-3.3-70b-versatile"  # Supported model

# Set up logging
logging.basicConfig(level=logging.INFO, filename="groqgazer.log")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY environment variable is not set. Please provide a valid API key in the .env file.")
    st.stop()

# Set up Streamlit page with logo (must be first Streamlit command)
st.set_page_config(page_title="GrokGazer", layout="wide", page_icon="logo.png")

# Custom CSS for modern UI inspired by Spotify
st.markdown(
    """
    <style>
    /* Sidebar styling with gradient */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #121212 0%, #1db954 100%);
        color: #ffffff;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content h2 {
        text-align: center;
        font-size: 1.5em;
        font-weight: 700;
        color: #1db954;
        text-transform: uppercase;
    }
    .sidebar .stRadio > label {
        color: #ffffff !important;
        font-size: 1.1em;
    }
    .sidebar .stButton>button {
        background: linear-gradient(90deg, #1db954, #191414);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 20px;
        transition: transform 0.3s ease;
    }
    .sidebar .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 10px #1db954;
    }
    .stExpander {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
    }
    .stExpander div[role="button"] {
        color: #1db954 !important;
    }

    /* Main content styling */
    .main .block-container {
        background: #1a1a1a;
        color: #e0e0e0;
        padding: 20px;
        border-radius: 10px;
    }
    .stTitle {
        color: #1db954 !important;
        font-size: 2.5em !important;
        text-align: center;
        font-weight: 800;
        text-transform: uppercase;
    }
    .stSubheader {
        color: #ffffff !important;
        font-size: 1.5em !important;
        font-weight: 600;
    }
    .stTextArea, .stFileUploader, .stDownloadButton {
        background: #2a2a2a;
        border: 1px solid #444;
        border-radius: 10px;
        color: #e0e0e0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1db954, #191414);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 20px;
        transition: transform 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 10px #1db954;
    }
    .stTabs [data-baseweb="tab"] {
        background: #2a2a2a !important;
        color: #e0e0e0 !important;
        padding: 10px 20px !important;
        border-radius: 10px 10px 0 0 !important;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #1db954 !important;
        color: #121212 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #1db954 !important;
        color: #121212 !important;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("GrokGazer: Smart Web Intelligence Explorer")
st.markdown("Unlock insights from the web using PocketGroq's crawling and scraping tools with Groq AI analysis.")

# Module check logic
def check_import(module_name: str) -> str | None:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return None
    except ImportError as e:
        return str(e)

required_modules = ['pocketgroq', 'groq', 'html2text', 'bs4', 'pdfplumber', 'PIL', 'pytesseract']
import_errors = {module: check_import(module) for module in required_modules}

if any(import_errors.values()):
    st.error("Some required modules could not be imported:")
    for module, error in import_errors.items():
        if error:
            st.error(f"{module}: {error}")
    st.error("Please check your installation and requirements.txt file.")
    st.write("Python version:", sys.version)
    st.write("Python path:", sys.executable)
    st.stop()

# Initialize GroqProvider and Groq client
groq_provider: GroqProvider | None = None
groq_client: Groq | None = None
try:
    groq_provider = GroqProvider(api_key=api_key)
    groq_client = Groq(api_key=api_key)
    logger.info(f"Initialized Groq client with model: {MODEL_NAME}")
    st.success("Groq services initialized successfully.")
except Exception as e:
    st.error(f"Error initializing Groq services: {str(e)}")
    st.stop()

# --- Helper Functions ---
def normalize_url(url: str) -> str:
    """Normalize URL to its base domain (scheme + netloc)."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL")
        normalized = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
        logger.debug(f"Normalized URL: {url} -> {normalized}")
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing URL {url}: {str(e)}")
        raise ValueError(f"Invalid URL: {str(e)}")

def convert_html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown using html2text."""
    try:
        h = html2text.HTML2Text()
        h.ignore_links = False
        markdown = h.handle(html)
        logger.info("Successfully converted HTML to Markdown")
        return markdown
    except Exception as e:
        logger.error(f"Error converting HTML to Markdown: {str(e)}")
        return f"Error: {str(e)}"

def extract_structured_data(html: str) -> Dict[str, Any]:
    """Extract structured data (title, meta description, etc.) from HTML using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        structured_data = {
            "title": soup.title.string if soup.title else "No title",
            "meta_description": "",
            "headings": []
        }
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            structured_data["meta_description"] = meta_desc['content']
        headings = soup.find_all(['h1', 'h2', 'h3'])
        structured_data["headings"] = [h.get_text(strip=True) for h in headings]
        logger.info("Successfully extracted structured data")
        return structured_data
    except Exception as e:
        logger.error(f"Error extracting structured data: {str(e)}")
        return {"error": f"Error: {str(e)}"}

def scrape_url(url: str, formats: List[str] = ["markdown", "html"]) -> Dict[str, Any]:
    """
    Scrape a single URL and return content in specified formats.

    Args:
        url: The URL to scrape.
        formats: List of output formats (e.g., ["markdown", "html"]).

    Returns:
        Dictionary containing scraped content or an error message.
    """
    try:
        if not validators.url(url):
            return {"error": "Invalid URL. Please enter a valid URL starting with http:// or https://."}
        html_result = groq_provider.enhanced_web_tool.scrape_page(url, formats=["html"])
        if not isinstance(html_result, dict) or "html" not in html_result:
            return {"error": "Failed to retrieve HTML content"}
        
        result = {"html": html_result["html"]}
        if "markdown" in formats:
            result["markdown"] = convert_html_to_markdown(result["html"])
        if "structured_data" in formats:
            result["structured_data"] = extract_structured_data(result["html"])
        
        logger.info(f"Successfully scraped {url} with markdown: {len(result.get('markdown', '')) > 0}")
        return result
    except RequestException as e:
        logger.error(f"Network error while scraping {url}: {str(e)}")
        return {"error": f"Network error while scraping {url}: {str(e)}"}
    except ValueError as e:
        logger.error(f"Invalid URL or format for {url}: {str(e)}")
        return {"error": f"Invalid URL or format: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error while scraping {url}: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}

def crawl_website(url: str, max_depth: int, max_pages: int, formats: List[str] = ["markdown", "html"],
                  exclude_paths: str = "", include_paths: str = "", ignore_sitemap: bool = False,
                  allow_backwards_links: bool = False) -> List[Dict[str, Any]]:
    """
    Crawl a website and return content from multiple pages.

    Args:
        url: The starting URL for crawling.
        max_depth: Maximum depth to crawl.
        max_pages: Maximum number of pages to crawl.
        formats: List of output formats.
        exclude_paths: Comma-separated paths to exclude.
        include_paths: Comma-separated paths to include.
        ignore_sitemap: Whether to ignore the sitemap.
        allow_backwards_links: Whether to allow crawling backwards links.

    Returns:
        List of dictionaries containing crawled content or error messages.
    """
    try:
        if not validators.url(url):
            return [{"error": "Invalid URL. Please enter a valid URL starting with http:// or https://."}]
        if max_pages > MAX_PAGES_LIMIT:
            max_pages = MAX_PAGES_LIMIT
            logger.warning(f"Max pages limited to {MAX_PAGES_LIMIT}")
        groq_provider.enhanced_web_tool.max_depth = max_depth
        groq_provider.enhanced_web_tool.max_pages = max_pages
        groq_provider.enhanced_web_tool.ignore_sitemap = ignore_sitemap
        groq_provider.enhanced_web_tool.allow_backwards_links = allow_backwards_links
        if exclude_paths:
            groq_provider.enhanced_web_tool.exclude_paths = [path.strip() for path in exclude_paths.split(",")]
        if include_paths:
            groq_provider.enhanced_web_tool.include_paths = [path.strip() for path in include_paths.split(",")]
        
        html_results = groq_provider.enhanced_web_tool.crawl(url, formats=["html"])
        results = []
        for html_result in html_results:
            result = {"url": html_result.get("url", "N/A"), "html": html_result.get("html", "")}
            if "markdown" in formats:
                result["markdown"] = convert_html_to_markdown(result["html"])
            if "structured_data" in formats:
                result["structured_data"] = extract_structured_data(result["html"])
            results.append(result)
        
        logger.info(f"Successfully crawled {url} with {len(results)} pages")
        return results
    except RequestException as e:
        logger.error(f"Network error while crawling {url}: {str(e)}")
        return [{"error": f"Network error while crawling {url}: {str(e)}"}]
    except ValueError as e:
        logger.error(f"Invalid URL or parameters for {url}: {str(e)}")
        return [{"error": f"Invalid URL or parameters: {str(e)}"}]
    except Exception as e:
        logger.error(f"Unexpected error while crawling {url}: {str(e)}")
        return [{"error": f"Unexpected error: {str(e)}"}]

def summarize_content(content: str) -> str:
    """
    Summarize the provided content using Groq API.

    Args:
        content: Text to summarize.

    Returns:
        Summary text or an error message.
    """
    try:
        logger.info(f"Summarizing with model: {MODEL_NAME}")
        prompt = f"Summarize the following content in 100 words or less:\n\n{content[:4000]}"
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        summary = response.choices[0].message.content.strip()
        logger.info("Successfully summarized content")
        return summary
    except APIError as e:
        error_msg = str(e).lower()
        if "model" in error_msg and "not found" in error_msg or "decommissioned" in error_msg:
            logger.error(f"Model {MODEL_NAME} is not supported: {str(e)}")
            return f"Error: Model {MODEL_NAME} is not supported. Please check https://console.groq.com/docs/deprecations for alternatives."
        logger.error(f"Groq API error summarizing content: {str(e)}")
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Error summarizing content: {str(e)}")
        return f"Error: {str(e)}"

def extract_keywords(content: str) -> List[str]:
    """
    Extract keywords from the provided content using Groq API.

    Args:
        content: Text to analyze.

    Returns:
        List of keywords or an error message.
    """
    try:
        logger.info(f"Extracting keywords with model: {MODEL_NAME}")
        prompt = f"Extract the top 5 keywords from the following content:\n\n{content[:4000]}"
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        keywords = response.choices[0].message.content.strip().split(", ")
        logger.info("Successfully extracted keywords")
        return keywords if keywords != [""] else ["No keywords found."]
    except APIError as e:
        error_msg = str(e).lower()
        if "model" in error_msg and "not found" in error_msg or "decommissioned" in error_msg:
            logger.error(f"Model {MODEL_NAME} is not supported: {str(e)}")
            return [f"Error: Model {MODEL_NAME} is not supported. Please check https://console.groq.com/docs/deprecations for alternatives."]
        logger.error(f"Groq API error extracting keywords: {str(e)}")
        return [f"Error: {str(e)}"]
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        return [f"Error: {str(e)}"]

def answer_question(context: str, question: str) -> str:
    """
    Answer a question based on the provided context using Groq API.

    Args:
        context: Context text.
        question: Question to answer.

    Returns:
        Answer text or an error message.
    """
    try:
        logger.info(f"Answering question with model: {MODEL_NAME}")
        prompt = f"Based on the following context, answer the question:\n\nContext: {context[:4000]}\n\nQuestion: {question}"
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"Successfully answered question: {question}")
        return answer
    except APIError as e:
        error_msg = str(e).lower()
        if "model" in error_msg and "not found" in error_msg or "decommissioned" in error_msg:
            logger.error(f"Model {MODEL_NAME} is not supported: {str(e)}")
            return f"Error: Model {MODEL_NAME} is not supported. Please check https://console.groq.com/docs/deprecations for alternatives."
        logger.error(f"Groq API error answering question: {str(e)}")
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return f"Error: {str(e)}"

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown('<h2>🌐 Navigation</h2>', unsafe_allow_html=True)
    scraping_type = st.radio("Choose Mode", ["Scrape", "Crawl", "Multimodal"], key="mode_switch", on_change=lambda: st.session_state.update({"displayed_result": None, "qa_context": ""}))

    st.markdown("---")
    st.markdown('<h2>⚙️ Options</h2>', unsafe_allow_html=True)

    url = st.text_input("Website URL", placeholder="https://example.com", key="url_input")

    if scraping_type == "Crawl":
        max_depth = st.number_input("Max Depth", min_value=1, value=3, key="max_depth")
        max_pages = st.number_input("Max Pages", min_value=1, value=10, key="max_pages")

    formats = st.multiselect("Output Formats", ["markdown", "html", "structured_data"], default=["markdown", "html"], key="formats")
    exclude_paths = st.text_input("Exclude Paths", placeholder="blog/,about/", key="exclude_paths")
    include_paths = st.text_input("Include Only Paths", placeholder="articles/", key="include_paths")
    ignore_sitemap = st.checkbox("Ignore Sitemap", key="ignore_sitemap")
    allow_backwards_links = st.checkbox("Allow Backwards Links", key="allow_backwards_links")

    # Clear Cache Button
    if st.button("🗑️ Clear Cache", key="clear_cache"):
        if "qa_context" in st.session_state:
            del st.session_state.qa_context
            st.success("Cache cleared successfully!")
            logger.info("Cache cleared")

    # About Section Moved to Sidebar
    with st.expander("📘 About GrokGazer"):
        st.markdown("""
        **GrokGazer** is a multimodal web intelligence explorer powered by PocketGroq and Groq AI.

        - 🌐 Scrape or crawl any site
        - 📂 Output in Markdown, HTML, or structured data
        - 🧠 AI-powered summarization, keyword extraction, and Q&A (using Groq's llama-3.3-70b-versatile)
        - 📄 Multimodal analysis of PDFs, text, and images

        Built with ❤️ using [Streamlit](https://streamlit.io), [PocketGroq](https://pocketgroq.com), and [Groq](https://groq.com)
        """)

    # Q&A Section Moved to Sidebar
    if st.session_state.get("qa_context", ""):
        st.markdown("## 💬 Ask Questions")
        user_question = st.text_input("Ask your question here...", placeholder="What is the main topic of the content?", key="question_input")
        if st.button("Get Answer", key="get_answer"):
            with st.spinner("Thinking..."):
                try:
                    answer = answer_question(st.session_state.qa_context, user_question)
                    st.markdown("### 🤖 Answer")
                    st.success(answer)
                    logger.info(f"Successfully answered question: {user_question}")
                except Exception as e:
                    st.error(f"Failed to get answer: {str(e)}")
                    logger.error(f"Failed to answer question: {e}")
        st.write(f"Debug: qa_context length = {len(st.session_state.qa_context)}")  # Debug check
        st.rerun()  # Force re-render to ensure sidebar updates

    st.markdown('</div>', unsafe_allow_html=True)

# --- Main UI ---
st.markdown('<div class="main-content">', unsafe_allow_html=True)
if scraping_type == "Scrape":
    st.subheader("🔍 Scrape a Single URL")
    if st.button("Run Scrape", key="run_scrape"):
        with st.spinner("Scraping..."):
            result = scrape_url(url, formats)
            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state.displayed_result = result
                if "markdown" in formats and "markdown" in result and result["markdown"]:
                    st.session_state.qa_context = result["markdown"][:MAX_CONTEXT_LENGTH]
                    logger.info(f"Set qa_context with length: {len(st.session_state.qa_context)}")
                st.success("Scrape successful!")
                tabs = st.tabs(["Markdown", "HTML", "Structured Data"])
                if "markdown" in formats and "markdown" in result:
                    with tabs[0]:
                        st.text_area("Markdown Output", result["markdown"], height=250, key="markdown_output")
                        st.markdown("### 🧠 Summary")
                        summary = summarize_content(result["markdown"])
                        st.write(summary)
                        st.markdown("### 🧠 Extracted Keywords")
                        keywords = extract_keywords(result["markdown"])
                        st.write(", ".join(keywords))
                if "html" in formats and "html" in result:
                    with tabs[1]:
                        st.code(result["html"], language="html")
                if "structured_data" in formats and "structured_data" in result:
                    with tabs[2]:
                        st.json(result["structured_data"])
                json_result = json.dumps(result, indent=2)
                st.download_button("Download JSON", json_result, "scrape_result.json", key="download_scrape")

elif scraping_type == "Crawl":
    st.subheader("🕷️ Crawl a Website")
    if st.button("Run Crawl", key="run_crawl"):
        with st.spinner("Crawling..."):
            results = crawl_website(url, max_depth, max_pages, formats, exclude_paths, include_paths, ignore_sitemap, allow_backwards_links)
            if any("error" in r for r in results):
                for result in results:
                    if "error" in result:
                        st.error(result["error"])
            else:
                st.session_state.displayed_result = results
                if "markdown" in formats:
                    combined_text = "\n".join([r.get("markdown", "") for r in results if "markdown" in r and r.get("markdown", "")])[:MAX_CONTEXT_LENGTH]
                    if combined_text:
                        st.session_state.qa_context = combined_text
                        logger.info(f"Set qa_context with length: {len(st.session_state.qa_context)}")
                st.success("Crawl successful!")
                for i, result in enumerate(results, 1):
                    st.markdown(f"### Page {i}")
                    st.write(f"🔗 URL: {result.get('url', 'N/A')}")
                    tabs = st.tabs(["Markdown", "HTML", "Structured Data"])
                    if "markdown" in formats and "markdown" in result:
                        with tabs[0]:
                            st.text_area(f"Markdown ({i})", result["markdown"], height=200, key=f"markdown_crawl_{i}")
                            st.markdown("### 🧠 Summary")
                            summary = summarize_content(result["markdown"])
                            st.write(summary)
                            st.markdown("### 🧠 Extracted Keywords")
                            keywords = extract_keywords(result["markdown"])
                            st.write(", ".join(keywords))
                    if "html" in formats and "html" in result:
                        with tabs[1]:
                            st.code(result["html"], language="html")
                    if "structured_data" in formats and "structured_data" in result:
                        with tabs[2]:
                            st.json(result["structured_data"])
                    st.markdown("---")
                json_result = json.dumps(results, indent=2)
                st.download_button("Download Crawl Results", json_result, "crawl_result.json", key="download_crawl")

elif scraping_type == "Multimodal":
    st.subheader("📤 Upload and Analyze Documents or Images")
    uploaded_file = st.file_uploader("Upload a PDF, TXT, or Image", type=["pdf", "txt", "png", "jpg", "jpeg"], key="file_uploader")
    extracted_text = ""

    if uploaded_file is not None:
        with st.spinner("Extracting content..."):
            file_extension = guess_extension(uploaded_file.type)
            try:
                if file_extension == ".pdf":
                    import pdfplumber
                    with pdfplumber.open(uploaded_file) as pdf:
                        for page in pdf.pages:
                            extracted_text += page.extract_text() or ""
                elif file_extension == ".txt":
                    extracted_text = uploaded_file.read().decode("utf-8")
                elif file_extension in [".png", ".jpg", ".jpeg"]:
                    from PIL import Image
                    import pytesseract
                    image = Image.open(uploaded_file)
                    extracted_text = pytesseract.image_to_string(image)
                else:
                    st.error(f"Unsupported file type: {file_extension}")
                    logger.error(f"Unsupported file type: {file_extension}")
            except Exception as e:
                st.error(f"Failed to extract content: {str(e)}")
                logger.error(f"Failed to extract content: {str(e)}")

        if extracted_text:
            st.session_state.displayed_result = {"extracted_text": extracted_text}
            st.session_state.qa_context = extracted_text[:MAX_CONTEXT_LENGTH]
            logger.info(f"Set qa_context with length: {len(st.session_state.qa_context)}")
            st.success("✅ Text extracted successfully!")
            st.text_area("Extracted Content", extracted_text, height=250, key="extracted_content")

            if st.button("🔎 Analyze", key="analyze_button"):
                with st.spinner("Analyzing..."):
                    try:
                        summary_result = summarize_content(extracted_text)
                        keywords_result = extract_keywords(extracted_text)

                        st.markdown("### 📌 Summary")
                        st.markdown(summary_result)

                        st.markdown("### 🗝️ Keywords")
                        st.write(", ".join(keywords_result))

                        results = {
                            "summary": summary_result,
                            "keywords": keywords_result,
                            "extracted_text": extracted_text
                        }
                        st.download_button("📥 Download Analysis", json.dumps(results, indent=2), "analysis.json", key="download_analysis")
                        logger.info("Successfully analyzed uploaded file")
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        logger.error(f"Analysis failed: {str(e)}")

# Session state management
if "displayed_result" not in st.session_state:
    st.session_state.displayed_result = None
if "qa_context" not in st.session_state:
    st.session_state.qa_context = ""

st.markdown('</div>', unsafe_allow_html=True)