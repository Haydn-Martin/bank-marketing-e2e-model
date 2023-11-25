# Set the verstion of python to run.
ARG PYTHON_VERSION=3.11.3
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Set the working directory.
WORKDIR /app

# Copy dependences as a separate step.
COPY requirements.txt /app/requirements.txt

# Download dependencies as a separate step.
RUN python -m pip install -r requirements.txt

# Copy the source code into the container.
COPY /bank_marketing_e2e_model .

# MyPy for static type checking
RUN mypy --explicit-package-bases --ignore-missing-imports /app/src/

# Run Pylint for linting
RUN pylint /app

# Run unit tests
RUN python -m unittest discover /app/tests

# Expose the port that the application listens on.
EXPOSE 8501

# Run the application.
CMD streamlit run main.py
