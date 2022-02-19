.PHONY : sync tests

requirements.txt : requirements.in
	pip-compile -v

sync : requirements.txt
	pip-sync

tests :
	pytest -v --cov=cost_based_selection --cov-report=html --cov-report=term
