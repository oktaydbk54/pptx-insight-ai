# Base image
FROM python:3.8

# Working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . /app

# Command to run the application
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]