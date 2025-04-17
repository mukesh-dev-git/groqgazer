import os
import sys
import json
import logging
import importlib
from typing import List, Dict, Any
from mimetypes import guess_extension

import streamlit as st
from dotenv import load_dotenv
from pocketgroq import GroqProvider
from groq import Groq, APIError
import validators
from requests.exceptions import RequestException
import html2text
from bs4 import BeautifulSoup

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

# Set up Streamlit page
st.set_page_config(page_title="GrokGazer", layout="wide", page_icon="🧠")
st.title("🧠 GrokGazer: Smart Web Intelligence Explorer")
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
    st.success("Groq services initialized successfully.")
except Exception as e:
    st.error(f"Error initializing Groq services: {str(e)}")
    st.stop()

# --- Constants ---
MAX_CONTEXT_LENGTH = 100000  # Limit for session state context
MAX_PAGES_LIMIT = 50  # Limit for crawling pages
MODEL_NAME = "llama-3.3-70b-versatile"  # Supported model

# --- Helper Functions ---
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
        # PocketGroq only returns HTML
        html_result = groq_provider.enhanced_web_tool.scrape_page(url, formats=["html"])
        if not isinstance(html_result, dict) or "html" not in html_result:
            return {"error": "Failed to retrieve HTML content"}
        
        result = {"html": html_result["html"]}
        if "markdown" in formats:
            result["markdown"] = convert_html_to_markdown(result["html"])
        if "structured_data" in formats:
            result["structured_data"] = extract_structured_data(result["html"])
        
        logger.info(f"Successfully scraped {url}")
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
        
        # PocketGroq only returns HTML
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
        return {"error": f"Unexpected error: {str(e)}"}

def map_website(url: str) -> List[str]:
    """
    Map a website by searching for its pages.

    Args:
        url: The website URL to map.

    Returns:
        List of URLs or an error message.
    """
    try:
        if not validators.url(url):
            return ["Error: Invalid URL. Please enter a valid URL starting with http:// or https://."]
        results = groq_provider.web_search(f"site:{url}")
        logger.info(f"Successfully mapped {url}")
        return [result['url'] for result in results]
    except RequestException as e:
        logger.error(f"Network error while mapping {url}: {str(e)}")
        return [f"Error: Network error: {str(e)}"]
    except Exception as e:
        logger.error(f"Unexpected error while mapping {url}: {str(e)}")
        return [f"Error: {str(e)}"]

def summarize_content(content: str) -> str:
    """
    Summarize the provided content using Groq API.

    Args:
        content: Text to summarize.

    Returns:
        Summary text or an error message.
    """
    try:
        prompt = f"Summarize the following content in 100 words or less:\n\n{content[:4000]}"  # Limit input size
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
        prompt = f"Extract the top 5 keywords from the following content:\n\n{content[:4000]}"  # Limit input size
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
    st.header("🔍 Navigation")
    scraping_type = st.radio("Choose Mode", ["Scrape", "Crawl", "Map", "Multimodal", "About"], index=0)

    st.markdown("---")
    st.header("⚙️ Options")

    url = st.text_input("Website URL", placeholder="https://example.com")

    if scraping_type == "Crawl":
        max_depth = st.number_input("Max Depth", min_value=1, value=3)
        max_pages = st.number_input("Max Pages", min_value=1, value=10)

    formats = st.multiselect("Output Formats", ["markdown", "html", "structured_data"], default=["markdown", "html"])
    exclude_paths = st.text_input("Exclude Paths", placeholder="blog/,about/")
    include_paths = st.text_input("Include Only Paths", placeholder="articles/")
    ignore_sitemap = st.checkbox("Ignore Sitemap")
    allow_backwards_links = st.checkbox("Allow Backwards Links")

# --- Main UI ---
if scraping_type == "Scrape":
    st.subheader("🔎 Scrape a Single URL")
    if st.button("Run Scrape") and url:
        with st.spinner("Scraping..."):
            result = scrape_url(url, formats)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Scrape successful!")
                tabs = st.tabs(["Markdown", "HTML", "Structured Data"])
                if "markdown" in formats and "markdown" in result:
                    st.session_state.qa_context = result["markdown"][:MAX_CONTEXT_LENGTH]
                    with tabs[0]:
                        st.text_area("Markdown Output", result["markdown"], height=250)
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
                st.download_button("Download JSON", json_result, "scrape_result.json")

elif scraping_type == "Crawl":
    st.subheader("🕷️ Crawl a Website")
    if st.button("Run Crawl") and url:
        with st.spinner("Crawling..."):
            results = crawl_website(url, max_depth, max_pages, formats, exclude_paths, include_paths, ignore_sitemap, allow_backwards_links)
            for i, result in enumerate(results, 1):
                st.markdown(f"### Page {i}")
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.write(f"🔗 URL: {result.get('url', 'N/A')}")
                    tabs = st.tabs(["Markdown", "HTML", "Structured Data"])
                    if "markdown" in formats and "markdown" in result:
                        combined_text = "\n".join([r.get("markdown", "") for r in results if "markdown" in r])[:MAX_CONTEXT_LENGTH]
                        st.session_state.qa_context = combined_text
                        with tabs[0]:
                            st.text_area(f"Markdown ({i})", result["markdown"], height=200)
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
            st.download_button("Download Crawl Results", json_result, "crawl_result.json")

elif scraping_type == "Map":
    st.subheader("🗘️ Map a Website")
    if st.button("Run Mapping") and url:
        with st.spinner("Mapping..."):
            results = map_website(url)
            if results and "Error" not in results[0]:
                st.success("Mapping Complete")
                for link in results:
                    st.write(f"🔗 {link}")
                json_result = json.dumps(results, indent=2)
                st.download_button("Download Sitemap", json_result, "site_map.json")
            else:
                st.error(results[0])

elif scraping_type == "About":
    st.subheader("📘 About GrokGazer")
    st.markdown("""
    **GrokGazer** is a multimodal web intelligence explorer powered by PocketGroq and Groq AI.

    - 🔍 Scrape or crawl any site
    - 📁 Output in Markdown, HTML, or structured data
    - 🧠 AI-powered summarization, keyword extraction, and Q&A
    - 📄 Multimodal analysis of PDFs, text, and images

    Built with ❤️ using [Streamlit](https://streamlit.io), [PocketGroq](https://pocketgroq.com), and [Groq](https://groq.com)
    """)

elif scraping_type == "Multimodal":
    st.subheader("🧾 Upload and Analyze Documents or Images")
    uploaded_file = st.file_uploader("Upload a PDF, TXT, or Image", type=["pdf", "txt", "png", "jpg", "jpeg"])
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
            st.success("✅ Text extracted successfully!")
            st.text_area("Extracted Content", extracted_text, height=250)
            st.session_state.qa_context = extracted_text[:MAX_CONTEXT_LENGTH]

            if st.button("🔎 Analyze"):
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
                        st.download_button("📥 Download Analysis", json.dumps(results, indent=2), "analysis.json")
                        logger.info("Successfully analyzed uploaded file")
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        logger.error(f"Analysis failed: {str(e)}")

# Session state for Q&A
if "qa_context" not in st.session_state:
    st.session_state.qa_context = ""

# Q&A section
if scraping_type in ["Scrape", "Crawl", "Multimodal"] and st.session_state.get("qa_context", ""):
    st.markdown("## 💬 Ask Questions about the Extracted Content")
    user_question = st.text_input("Ask your question here...", placeholder="What is the main topic of the content?")
    if st.button("Get Answer") and user_question:
        with st.spinner("Thinking..."):
            try:
                answer = answer_question(st.session_state.qa_context, user_question)
                st.markdown("### 🤖 Answer")
                st.success(answer)
                logger.info(f"Successfully answered question: {user_question}")
            except Exception as e:
                st.error(f"Failed to get answer: {str(e)}")
                logger.error(f"Failed to answer question: {str(e)}")