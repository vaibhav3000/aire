FROM python:3.11-slim

WORKDIR /app
COPY . .

# Install the package itself so the aire-gate CLI is available in the image.
RUN pip install --no-cache-dir -e ".[dev]"

CMD ["python", "-m", "pytest", "tests/", "-q"]
