PYTHON=Code_python/venv/bin/python

run:
	$(PYTHON) Code_python/tryagain.py

web:
	cd Code_web && python3 -m http.server 8000

clean:
	rm -rf __pycache__ Code_python/__pycache__

