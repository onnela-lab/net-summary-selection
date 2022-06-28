.PHONY : build sync tests workspace/simulations workspace/simulations/ba

# Helper function to split by special character.
wordx = $(word $2,$(subst $3, ,$1))

requirements.txt : requirements.in setup.py
	pip-compile -v

sync : requirements.txt
	pip-sync

tests :
	OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest -v -s \
		--cov=cost_based_selection --cov-report=html --cov-report=term --disable-warnings

build : lint tests

lint :
	flake8
