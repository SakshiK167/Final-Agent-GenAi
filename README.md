# Experiment 10: MLOps Deployment of a Task-Automation Agent
## 🔬 AI Research Brief Agent

A production-ready, beginner-friendly MLOps project that automates the generation of factual research briefs. The agent accepts any research topic, retrieves live web search snippets via DuckDuckGo (`ddgs`), and synthesizes **exactly three concise factual findings** using an OpenRouter-compatible language model via the OpenAI Python SDK. Citations are displayed as verified clickable links, and no facts or citations are fabricated.

---

## 📑 Table of Contents
- [Architecture & Workflow](#-architecture--workflow)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Local Setup & Development (Windows PowerShell)](#-local-setup--development-windows-powershell)
- [Running Automated Tests](#-running-automated-tests)
- [Containerization with Docker](#-containerization-with-docker)
- [CI/CD Pipeline (GitHub Actions)](#-cicd-pipeline-github-actions)
- [Production Deployment on Render](#-production-deployment-on-render)
- [Security & Environment Variables](#-security--environment-variables)
- [MLOps Best Practices Implemented](#-mlops-best-practices-implemented)

---

## 🏛️ Architecture & Workflow

```
[ User Input (Topic) ]
         │
         ▼
[ Input Sanitization & Validation ]
         │
         ▼
[ DuckDuckGo Live Search (DDGS) ] ──▶ Returns: Title, URL, Snippet
         │
         ▼
[ Factual Prompt Construction ] (Zero Hallucination Grounding)
         │
         ▼
[ OpenRouter API (OpenAI SDK) ] ──▶ Evaluated with OpenRouter Model
         │
         ▼
[ Response Extraction & 3-Bullet Rule Enforcement ]
         │
         ▼
[ Streamlit UI Output: 3 Findings + Grounded Citations + Latency Metric ]
```

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language & Runtime** | Python 3.11 | Modern, lightweight, and stable Python environment |
| **Frontend Framework** | Streamlit | Responsive, clean user interface tailored for rapid ML prototyping |
| **Web Search** | DuckDuckGo Search (`ddgs`) | Free live web search without requiring external search API keys |
| **LLM Inference** | OpenAI SDK + OpenRouter | OpenRouter endpoint (`https://openrouter.ai/api/v1`) with flexible model selection |
| **Testing** | pytest | Unit test suite with 100% mocked dependencies (no API fees or live internet needed) |
| **Containerization** | Docker | Minimal `python:3.11-slim` multi-stage styled image with non-root security |
| **CI/CD** | GitHub Actions | Automated lint/test execution and Docker build verification on push/PR |
| **Cloud Hosting** | Render | Docker-based Web Service deployment on Render's free tier with auto-deploy |

---

## 📁 Repository Structure

```
exp-10 agent/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow (Test & Docker build)
├── .streamlit/
│   └── config.toml             # Streamlit server and theme configuration
├── tests/
│   ├── __init__.py
│   └── test_app.py             # 24 unit tests covering validation, parsing, errors, and mocks
├── .dockerignore               # Prevents secrets, virtual environments, and caches from entering image
├── .env.example                # Example template for required environment variables
├── .gitignore                  # Git ignore rules protecting keys, bytecode, and virtual envs
├── Dockerfile                  # Production container definition configured for Render dynamic PORT
├── README.md                   # Comprehensive project documentation
├── app.py                      # Complete Streamlit application and core logic functions
└── requirements.txt            # Compatible pinned Python package dependencies
```

---

## 💻 Local Setup & Development (Windows PowerShell)

Follow these exact steps in **Windows PowerShell** to run the agent locally:

### Step 1: Open the Project Directory
```powershell
cd "c:\Users\Sushant Patil\Downloads\exp-10 agent"
```

### Step 2: Create a Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
> *Tip: If you encounter an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and re-run the activation script.*

### Step 4: Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables
Copy the `.env.example` file to create a local `.env` file:
```powershell
Copy-Item .env.example .env
```
Edit `.env` or set the environment variables in your active PowerShell session:
```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-your-actual-openrouter-key"
$env:OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
$env:PORT = "8501"
```
> **Important:** Always check [OpenRouter Models](https://openrouter.ai/models) to verify that your chosen model is currently online and active. Free-tier examples include:
> - `meta-llama/llama-3.1-8b-instruct:free`
> - `google/gemini-2.0-flash-exp:free`
> - `openai/gpt-4o-mini`

### Step 6: Launch the Streamlit Application
```powershell
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 🧪 Running Automated Tests

The test suite runs entirely offline using mocks for both DuckDuckGo search and OpenRouter API calls. It validates input edge cases, error conditions, and the three-bullet guarantee.

Run the tests with `pytest`:
```powershell
pytest -v
```

Expected output:
```
tests/test_app.py::test_validate_topic_valid PASSED
tests/test_app.py::test_validate_topic_empty PASSED
tests/test_app.py::test_validate_topic_whitespace PASSED
tests/test_app.py::test_validate_topic_none PASSED
tests/test_app.py::test_validate_topic_truncates_long PASSED
tests/test_app.py::test_sanitize_topic_removes_newlines_and_truncates PASSED
tests/test_app.py::test_get_openrouter_config_defaults PASSED
tests/test_app.py::test_get_openrouter_config_custom PASSED
tests/test_app.py::test_search_web_success PASSED
tests/test_app.py::test_search_web_handles_empty_or_malformed PASSED
tests/test_app.py::test_search_web_handles_exception PASSED
tests/test_app.py::test_parse_and_validate_bullets_dash_format PASSED
tests/test_app.py::test_parse_and_validate_bullets_numbered_format PASSED
tests/test_app.py::test_parse_and_validate_bullets_excess_items PASSED
tests/test_app.py::test_parse_and_validate_bullets_padding PASSED
tests/test_app.py::test_parse_and_validate_bullets_empty_raises PASSED
tests/test_app.py::test_generate_brief_success PASSED
tests/test_app.py::test_generate_brief_missing_api_key PASSED
tests/test_app.py::test_generate_brief_missing_search_results PASSED
tests/test_app.py::test_generate_brief_auth_error PASSED
tests/test_app.py::test_generate_brief_rate_limit_error PASSED
tests/test_app.py::test_generate_brief_timeout_error PASSED
tests/test_app.py::test_generate_brief_connection_error PASSED
tests/test_app.py::test_application_integrity_and_exports PASSED
============================== 24 passed in ~1.3s ==============================
```

---

## 🐳 Containerization with Docker

The container is built on `python:3.11-slim`, drops privileges to a non-root `appuser`, includes a healthcheck, and dynamically adapts to the port allocated by Render.

### Build Docker Image
```powershell
docker build -t ai-research-brief-agent .
```

### Run Docker Container Locally
Pass your OpenRouter API key as an environment variable (or use `--env-file .env`):
```powershell
docker run -d --name research-agent -p 8501:8501 -e OPENROUTER_API_KEY="your-openrouter-api-key" -e OPENROUTER_MODEL="meta-llama/llama-3.1-8b-instruct:free" ai-research-brief-agent
```
Navigate to: **`http://localhost:8501`**

### Check Container Health & Logs
```powershell
# View running container status (includes health status)
docker ps

# Inspect logs
docker logs -f research-agent

# Stop and clean up container
docker stop research-agent
docker rm research-agent
```

---

## 🔄 CI/CD Pipeline (GitHub Actions)

The `.github/workflows/ci.yml` pipeline triggers automatically on every `push` and `pull_request` targeting the `main` or `master` branch.

### Workflow Stages:
1. **`test` Job:**
   - Checks out the code.
   - Sets up Python 3.11 with pip caching.
   - Installs dependencies from `requirements.txt`.
   - Executes the 24-test `pytest` suite offline without requiring external credentials.
2. **`docker-build` Job (Runs only if tests pass):**
   - Sets up Docker Buildx.
   - Builds the Docker image to guarantee Dockerfile syntax and container assembly integrity.
   - Does **not** publish or require any Docker Hub credentials.

---

## 🚀 Production Deployment on Render

Render provides free Docker-based Web Service hosting with automatic deploys directly connected to GitHub.

### Step 1: Push Code to GitHub
Initialize Git and push your repository to GitHub from Windows PowerShell:
```powershell
git init
git add .
git commit -m "feat: initial commit for Experiment 10 AI Research Brief Agent"
git branch -M main
git remote add origin https://github.com/<your-github-username>/<your-repo-name>.git
git push -u origin main
```

### Step 2: Obtain an OpenRouter API Key
1. Go to [OpenRouter.ai](https://openrouter.ai/) and sign in.
2. Navigate to **Keys** (`https://openrouter.ai/keys`) and click **Create Key**.
3. Copy the key (starts with `sk-or-v1-...`).
4. Browse [OpenRouter Models](https://openrouter.ai/models) and confirm your chosen model (e.g. `meta-llama/llama-3.1-8b-instruct:free`).

### Step 3: Create a Web Service on Render
1. Log into your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** ➡️ **Web Service**.
3. Select **Build and deploy from a Git repository** and connect your GitHub repository.
4. Configure the Web Service settings:
   - **Name:** `ai-research-brief-agent` (or your preferred name)
   - **Region:** Choose the region closest to you (e.g., Oregon, Frankfurt, Singapore)
   - **Branch:** `main`
   - **Runtime:** **Docker** (Render automatically detects your `Dockerfile`)
   - **Instance Type:** **Free**
5. Configure Environment Variables under the **Environment Variables** section:
   | Key | Value | Notes |
   | :--- | :--- | :--- |
   | `OPENROUTER_API_KEY` | `sk-or-v1-xxxxxxxxxxxx` | Secret API key (never commit to Git) |
   | `OPENROUTER_MODEL` | `meta-llama/llama-3.1-8b-instruct:free` | Any active OpenRouter model |
6. Click **Create Web Service**.

### Step 4: Verification & Auto-Deploy
- Render will automatically trigger a Docker build.
- Streamlit will bind to `0.0.0.0` and Render's injected `$PORT`.
- Once the status displays **Live**, access your public HTTPS application URL (e.g., `https://ai-research-brief-agent.onrender.com`).
- Any future push to the `main` branch will automatically trigger GitHub Actions CI and deploy to Render!

---

## 🔒 Security & Environment Variables

- **Zero Secret Exposure:** API keys are read exclusively from environment variables (`OPENROUTER_API_KEY`).
- **No Secrets in Docker:** Docker build arguments and layers never contain secrets.
- **Git Protection:** `.gitignore` actively prevents `.env`, `.streamlit/secrets.toml`, and virtual environments from ever being staged or committed.
- **Sanitized Logging:** Python logging records sanitized queries (truncated to 50 characters with line-breaks removed) and execution latency without ever printing API tokens or full system prompts.

---

## 📊 MLOps Best Practices Implemented

| MLOps Pillar | Implementation |
| :--- | :--- |
| **Reproducibility** | Compatible pinned dependencies in `requirements.txt` and pinned base Docker image `python:3.11-slim` |
| **Grounding & Trust** | Prompt strictly limits responses to retrieved search snippets; citations are verified URLs |
| **Defensive Architecture** | Full validation on user topic input, model responses, and gracefully typed exceptions |
| **Observability** | Structured execution logs tracking request start, latency breakdown (search vs. LLM), and error types |
| **Continuous Integration** | Automated pytest execution and Docker build verification on every commit via GitHub Actions |
| **Zero-Downtime Deployment**| Render container deployment with built-in healthcheck (`/_stcore/health`) |
