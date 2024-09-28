# Define your virtual environment and app
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
APP = app.py

# Install dependencies and run tests if they exist
.PHONY: install
install:
	python3 -m venv $(VENV)  # Create a virtual environment
	$(PIP) install --upgrade pip  # Upgrade pip
	$(PIP) install -r requirements.txt  # Install dependencies
	# Check for test files and only run pytest if they exist
	if ls test_*.py *_test.py 2>/dev/null | grep -q .; then \
		echo "Test files found, running pytest..."; \
		$(VENV)/bin/pytest -v; \
	else \
		echo "No test files found, skipping pytest."; \
	fi

# Run the Dash application
.PHONY: run
run:
	$(PYTHON) $(APP)  # Run the Dash application

# Clean up virtual environment
.PHONY: clean
clean:
	rm -rf $(VENV)  # Remove the virtual environment

# Reinstall all dependencies
.PHONY: reinstall
reinstall: clean install  # Clean and reinstall
