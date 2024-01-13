# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy models
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download ru_core_news_sm

# Make port 5000 available to the world outside this container
EXPOSE 80

# Copy .env file into the container (if you prefer keeping it separate)
#COPY .venv .venv

# At the end of your Dockerfile
#CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:80", "app:app"]

# Run app.py when the container launches
CMD ["python", "./app.py", "0.0.0.0:80"]