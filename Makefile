.PHONY : build sync tests

requirements.txt : requirements.in setup.py
	pip-compile -v

sync : requirements.txt
	pip-sync

tests :
	pytest -v -s --cov=cost_based_selection --cov-report=html --cov-report=term

build : tests
