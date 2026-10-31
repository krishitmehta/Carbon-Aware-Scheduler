.PHONY: setup pipeline test app

setup:
	pip install -r requirements.txt -r requirements-dev.txt

pipeline:
	python ingest.py
	python clean_eda.py
	python features.py
	python forecast.py
	python backtest.py

test:
	python -m pytest -q

app:
	streamlit run app.py
