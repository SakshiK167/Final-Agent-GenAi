# ------------------------------------------------------------------------------
# Experiment 10: MLOps Deployment of a Task-Automation Agent
# Base Image: Python 3.11 slim for minimal size, security, and fast builds
# ------------------------------------------------------------------------------
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered stdout/stderr logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

# Set working directory
WORKDIR /app

# Install system dependencies (curl required for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group for security compliance
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy dependency specification and install Python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and configuration
COPY .streamlit /app/.streamlit
COPY app.py /app/app.py

# Set proper ownership for non-root execution
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose default Streamlit port (Render dynamically sets PORT at runtime)
EXPOSE 8501

# Health check to ensure the Streamlit web service is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Execute Streamlit binding to 0.0.0.0 and Render dynamic $PORT
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
