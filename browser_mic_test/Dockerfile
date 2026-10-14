FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2

WORKDIR /workspace/browser_mic_test

# Match the CPU checkpoint's tested PyTorch version. Other runtime packages
# come from the small, pinned app requirements file.
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.14.1

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py streaming.py ./
COPY static ./static

RUN mkdir -p /workspace/household_sound_model/artifacts \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /workspace
USER appuser

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
