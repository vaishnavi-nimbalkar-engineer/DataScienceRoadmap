default:
	cat justfile

setup:
	venv setup python 3.12
	python -m pip install uv

install:
	uv pip install -r pyproject.toml

serve:
	streamlit run main.py

run: serve

