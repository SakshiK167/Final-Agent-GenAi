"""
AI Research Brief Agent - Streamlit Application
Experiment 10: MLOps Deployment of a Task-Automation Agent
"""

import os
import re
import sys
import time
import logging
from typing import Any

import streamlit as st
from openai import OpenAI, AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError, APIError

# Local development .env loading (fallback if python-dotenv is present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# DuckDuckGo search import supporting both duckduckgo_search and ddgs naming
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

# ---------------------------------------------------------------------------
# Structured Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ai_research_agent")

# Documented default model identifier (users must verify availability on OpenRouter)
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct:free"


# ---------------------------------------------------------------------------
# Helper & Core Logic Functions (Modular & Testable)
# ---------------------------------------------------------------------------
def sanitize_topic(topic: str, max_len: int = 50) -> str:
    """Sanitize and truncate research topic for safe structured logging."""
    if not topic:
        return ""
    cleaned = re.sub(r"[\r\n\t]+", " ", topic).strip()
    return cleaned[:max_len]


def validate_topic(topic: str) -> str:
    """Validate user input topic. Raises ValueError if invalid."""
    if topic is None:
        raise ValueError("Research topic cannot be empty. Please enter a topic.")
    
    cleaned = topic.strip()
    if not cleaned:
        raise ValueError("Research topic cannot be empty. Please enter a topic.")
    
    if len(cleaned) > 300:
        cleaned = cleaned[:300]
        
    return cleaned


def get_openrouter_config() -> tuple[str, str]:
    """Retrieve OpenRouter API key and Model from environment variables."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL
    return api_key, model


def search_web(topic: str, max_results: int = 5) -> list[dict[str, str]]:
    """
    Search DuckDuckGo for relevant snippets and URLs.
    
    Returns a list of dicts: [{'title': ..., 'url': ..., 'snippet': ...}]
    """
    if DDGS is None:
        raise RuntimeError("DuckDuckGo search package (ddgs/duckduckgo-search) is not installed.")

    topic_clean = validate_topic(topic)
    logger.info("Search initiated | query='%s' | max_results=%d", sanitize_topic(topic_clean), max_results)
    
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(keywords=topic_clean, max_results=max_results))
    except Exception as exc:
        logger.error("Search failed | query='%s' | error_type='%s'", sanitize_topic(topic_clean), type(exc).__name__)
        raise RuntimeError(f"Web search encountered an error: {str(exc)}") from exc

    results: list[dict[str, str]] = []
    for item in raw_results:
        title = item.get("title", "").strip() or "Web Source"
        url = item.get("href") or item.get("link", "").strip()
        snippet = item.get("body") or item.get("snippet", "").strip()
        
        # Only retain entries with valid content and links
        if url and snippet:
            results.append({"title": title, "url": url, "snippet": snippet})

    logger.info("Search completed | query='%s' | results_retrieved=%d", sanitize_topic(topic_clean), len(results))
    return results


def parse_and_validate_bullets(raw_text: str) -> list[str]:
    """
    Parse LLM response and guarantee exactly three concise factual bullet points.
    Strips numbering, leading bullet symbols, and intro/outro noise.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("The language model returned an empty response.")

    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
    cleaned_bullets: list[str] = []

    prefixes = ("- ", "* ", "• ", "> ")

    for line in lines:
        stripped = line.strip()
        # Remove markdown bold/italic asterisks at line edges
        stripped = re.sub(r"^\*{1,2}(.*?)\*{1,2}", r"\1", stripped)
        # Remove numbered prefixes like '1. ', '2. ', '1) ', etc.
        stripped = re.sub(r"^\d+[\.\)]\s*", "", stripped)
        
        for prefix in prefixes:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):].strip()
        
        lower = stripped.lower()
        if any(lower.startswith(skip) for skip in (
            "here are", "here is", "in conclusion", "summary:", "key findings:", "brief:"
        )):
            continue

        if len(stripped) >= 15:  # Ignore fragments
            cleaned_bullets.append(stripped)

    # Exactly three bullets handling
    if len(cleaned_bullets) >= 3:
        return cleaned_bullets[:3]

    # If fewer than 3, split multi-sentence statements
    if 0 < len(cleaned_bullets) < 3:
        expanded: list[str] = []
        for bullet in cleaned_bullets:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", bullet) if len(s.strip()) > 15]
            if len(sentences) > 1:
                expanded.extend(sentences)
            else:
                expanded.append(bullet)
        
        if len(expanded) >= 3:
            return expanded[:3]

        # Fill remaining slots with factual constraint indicator
        while len(cleaned_bullets) < 3:
            cleaned_bullets.append("No additional distinct factual finding was verified within the retrieved search snippets.")
        return cleaned_bullets[:3]

    raise ValueError("Unable to extract valid factual findings from the model response.")


def generate_research_brief(
    topic: str,
    search_results: list[dict[str, str]],
    api_key: str,
    model: str,
    client: Any = None,
) -> list[str]:
    """
    Synthesize exactly three concise factual bullet points from search snippets
    using OpenRouter via the OpenAI Python SDK.
    """
    if not api_key:
        raise ValueError("OpenRouter API key is missing. Please set OPENROUTER_API_KEY.")
    
    if not search_results:
        raise ValueError("Cannot generate a brief without search results.")

    # Format search snippets cleanly for LLM input
    formatted_snippets = []
    for idx, res in enumerate(search_results, start=1):
        formatted_snippets.append(
            f"Source [{idx}]: {res['title']}\n"
            f"URL: {res['url']}\n"
            f"Snippet: {res['snippet']}"
        )
    snippets_block = "\n\n".join(formatted_snippets)

    system_prompt = (
        "You are an expert factual research analyst. Your duty is to produce an objective, concise research brief "
        "based strictly on the provided web search snippets.\n\n"
        "STRICT CONSTRAINTS:\n"
        "1. Produce EXACTLY three concise, factual bullet points.\n"
        "2. Base every fact ONLY on the provided search snippets. NEVER fabricate citations or extrapolate claims.\n"
        "3. If the search results do not support a conclusion, do not claim they do.\n"
        "4. Format the output strictly as three bullet points, each starting with '- '.\n"
        "5. Do NOT output intro, disclaimers, or outro."
    )

    user_prompt = (
        f"Research Topic: {topic}\n\n"
        f"Retrieved Search Snippets:\n{snippets_block}\n\n"
        "Generate exactly three concise factual bullet points based solely on the snippets above:"
    )

    if client is None:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://render.com",
                "X-Title": "AI Research Brief Agent",
            },
        )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        content = response.choices[0].message.content or ""
    except AuthenticationError as exc:
        logger.error("OpenRouter auth failed | model='%s' | error='%s'", model, str(exc))
        raise PermissionError("OpenRouter Authentication Failed: Please check your OPENROUTER_API_KEY.") from exc
    except RateLimitError as exc:
        logger.error("OpenRouter rate limit hit | model='%s' | error='%s'", model, str(exc))
        raise RuntimeError("OpenRouter Rate Limit Exceeded: Please wait a moment or check your account balance.") from exc
    except APITimeoutError as exc:
        logger.error("OpenRouter timeout | model='%s' | error='%s'", model, str(exc))
        raise TimeoutError("OpenRouter Request Timeout: The model took too long to respond.") from exc
    except APIConnectionError as exc:
        logger.error("OpenRouter connection error | model='%s' | error='%s'", model, str(exc))
        raise ConnectionError("Network Connection Error: Unable to reach OpenRouter API servers.") from exc
    except APIError as exc:
        logger.error("OpenRouter API error | model='%s' | error='%s'", model, str(exc))
        raise RuntimeError(f"OpenRouter API Error: {str(exc)}") from exc
    except Exception as exc:
        logger.error("OpenRouter unexpected error | model='%s' | error='%s'", model, str(exc))
        raise RuntimeError(f"Unexpected generation error: {str(exc)}") from exc

    return parse_and_validate_bullets(content)


# ---------------------------------------------------------------------------
# Streamlit Web Application Interface
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="AI Research Brief Agent",
        page_icon="🔬",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    api_key, model = get_openrouter_config()

    # Sidebar: MLOps & Environment Settings
    with st.sidebar:
        st.markdown("### ⚙️ System Status")
        
        # API Key Indicator
        if api_key:
            st.success("✅ **API Key:** Configured (Environment)")
        else:
            st.error("❌ **API Key:** Missing (`OPENROUTER_API_KEY`)")
            st.caption("Set `OPENROUTER_API_KEY` in `.env` locally or in Render environment variables.")

        # Model Indicator
        st.info(f"🧠 **Model:** `{model}`")
        st.caption("Configured via `OPENROUTER_MODEL`. Verify model status on [OpenRouter Models](https://openrouter.ai/models).")

        st.markdown("---")
        st.markdown("### 🧪 Experiment 10")
        st.markdown(
            "**Topic:** MLOps Deployment of a Task-Automation Agent\n\n"
            "- **Stack:** Streamlit + DDGS + OpenRouter\n"
            "- **CI/CD:** GitHub Actions + Pytest\n"
            "- **Container:** Docker + Render Cloud"
        )
        st.markdown("---")
        st.caption("🔒 *Antigravity MLOps Lab • Safe & Secret-Free*")

    # Header Section
    st.title("🔬 AI Research Brief Agent")
    st.markdown(
        "Enter any research topic below. The agent searches the live web, extracts authentic source snippets, "
        "and synthesizes **exactly three factual key findings** backed by citations."
    )

    # Example Topic Pills / Quick Buttons
    st.markdown("**💡 Quick Examples:**")
    example_topics = [
        "James Webb Space Telescope Exoplanet Discoveries",
        "Solid-State Battery Breakthroughs in EV Technology",
        "Recent Clinical Applications of CRISPR Gene Editing",
        "Quantum Computing Quantum Advantage Benchmarks",
    ]
    
    col1, col2 = st.columns(2)
    selected_example = None
    for i, topic_example in enumerate(example_topics):
        col = col1 if i % 2 == 0 else col2
        if col.button(f"📌 {topic_example}", key=f"ex_{i}", use_container_width=True):
            selected_example = topic_example

    # Topic Input Field
    default_text = selected_example if selected_example else ""
    user_topic = st.text_input(
        "Enter your research topic:",
        value=default_text,
        placeholder="e.g., Perovskite solar cells commercialization status",
        max_chars=300,
        help="Be as specific as possible for higher-quality research results.",
    )

    generate_clicked = st.button("🚀 Generate Brief", type="primary", use_container_width=True)

    if generate_clicked:
        # 1. Input Validation
        try:
            clean_topic = validate_topic(user_topic)
        except ValueError as err:
            st.warning(f"⚠️ {str(err)}")
            return

        # 2. Check API Key
        if not api_key:
            st.error(
                "🔑 **OpenRouter API Key Missing**\n\n"
                "Please configure the `OPENROUTER_API_KEY` environment variable in your `.env` file "
                "or on your Render dashboard before generating briefs."
            )
            return

        start_time = time.perf_counter()
        logger.info("Request started | topic='%s' | length=%d", sanitize_topic(clean_topic), len(clean_topic))

        # 3. Processing with Loading Spinner
        with st.spinner("🔍 Step 1/2: Searching the live web for verified source snippets..."):
            search_start = time.perf_counter()
            try:
                search_results = search_web(clean_topic, max_results=5)
            except Exception as err:
                logger.error("Search failure | topic='%s' | err='%s'", sanitize_topic(clean_topic), str(err))
                st.error(f"🔍 **Search Error:** {str(err)}")
                return
            search_duration = time.perf_counter() - search_start

        if not search_results:
            st.info("ℹ️ No relevant web search results were found for this topic. Please try rephrasing with broader keywords.")
            return

        with st.spinner(f"🧠 Step 2/2: Synthesizing factual brief with OpenRouter ({model})..."):
            llm_start = time.perf_counter()
            try:
                bullets = generate_research_brief(clean_topic, search_results, api_key, model)
            except PermissionError as err:
                st.error(f"🔒 {str(err)}")
                return
            except TimeoutError as err:
                st.error(f"⏱️ {str(err)}")
                return
            except ConnectionError as err:
                st.error(f"🌐 {str(err)}")
                return
            except Exception as err:
                st.error(f"❌ {str(err)}")
                return
            llm_duration = time.perf_counter() - llm_start

        total_duration = time.perf_counter() - start_time
        logger.info(
            "Request finished | topic='%s' | search_sec=%.2f | llm_sec=%.2f | total_sec=%.2f",
            sanitize_topic(clean_topic),
            search_duration,
            llm_duration,
            total_duration,
        )

        # 4. Display Results
        st.markdown("---")
        st.subheader("📋 Key Research Findings")
        
        # Display 3 Numbered Findings
        for idx, bullet in enumerate(bullets, start=1):
            st.markdown(
                f"""
                <div style="background-color: #F1F5F9; border-left: 4px solid #2563EB; padding: 12px 16px; border-radius: 4px; margin-bottom: 12px;">
                    <strong style="color: #1E3A8A; font-size: 15px;">Finding {idx}:</strong>
                    <p style="margin: 4px 0 0 0; color: #1E293B; font-size: 14.5px; line-height: 1.5;">{bullet}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Display Source Links
        st.markdown("### 🔗 Verified Sources Consulted")
        st.caption("Findings are strictly grounded in these retrieved web citations:")
        for idx, res in enumerate(search_results, start=1):
            st.markdown(f"**{idx}.** [{res['title']}]({res['url']})")

        # Performance & Timing Badge
        st.markdown("---")
        st.success(
            f"⏱️ **Brief synthesized in {total_duration:.2f}s** "
            f"(Search: {search_duration:.2f}s • LLM Synthesis: {llm_duration:.2f}s • Model: `{model}`)"
        )


if __name__ == "__main__":
    main()
