# Use the official Python 3.10.11 image as the base image
FROM python:3.10.11-slim

WORKDIR /app

COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create the data directory (where JSON files will be stored)
RUN mkdir -p /app/data

# Expose port 5000
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
