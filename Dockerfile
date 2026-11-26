# CPU image for the Streamlit demo:  docker build -t driver-distraction .  &&  docker run -p 8501:8501 driver-distraction
FROM python:3.12-slim
WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./
COPY .streamlit ./.streamlit
COPY demo_samples ./demo_samples
COPY eda ./eda
COPY results_multi ./results_multi
RUN useradd -m app && mkdir -p logs && chown -R app /app
USER app
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "demo_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
