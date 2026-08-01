# Start from an official, lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy just the requirements file first (explained below)
COPY requirements.txt .

# Install all our Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the NLTK data our preprocessing needs
RUN python -m nltk.downloader punkt punkt_tab stopwords

# Now copy the rest of our project code
COPY . .

# Tell Docker this container listens on port 8000
EXPOSE 8000

# The command that runs when the container starts
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]