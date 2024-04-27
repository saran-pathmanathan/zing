# Use an official Python runtime as a parent image
FROM python:3.10.11

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirement.txt

# Use gunicorn as the entrypoint, and run the app on the $PORT environment variable
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app