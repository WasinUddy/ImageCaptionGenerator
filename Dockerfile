FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /app

# Copy require files
COPY requirements.txt .
COPY model.py .
COPY app.py .
COPY saved_data saved_data
COPY saved_model saved_model

# Install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Run app
CMD ["python", "app.py"]