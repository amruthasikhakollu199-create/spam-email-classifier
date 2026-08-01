# Start from the same lightweight Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy just the requirements file first (same caching benefit as before)
COPY requirements.txt .

# Install all our Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of our project code
COPY . .

# Tell Docker this container listens on port 8501 (Streamlit's default)
EXPOSE 8501

# The command that runs when the container starts
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]