# Define your virtual environment and app
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
APP = app.py

# Install dependencies and run tests
.PHONY: install
install:
	python3 -m venv $(VENV)  # Create a virtual environment
	$(PIP) install --upgrade pip  # Upgrade pip
	$(PIP) install -r requirements.txt  # Install dependencies
	$(VENV)/bin/pytest -v  # Run tests using pytest in verbose mode

# Run the Dash application (only if tests pass)
.PHONY: run
run:
	$(PYTHON) $(APP)  # Run the Dash application
