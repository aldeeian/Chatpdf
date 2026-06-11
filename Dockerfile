# ChatPDF — Streamlit UI in a container
#
# Build:  docker build -t chatpdf .
# Run:    docker run -p 8501:8501 -e GROQ_API_KEY=your_key chatpdf
# Open:   http://localhost:8501

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker layer caching kicks in on code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "streamlitui.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
