.SECONDEXPANSION :
.PHONY : build sync tests workspace/simulations workspace/simulations/ba

# Helper function to split by special character.
wordx = $(word $2,$(subst $3, ,$1))

requirements.txt : requirements.in setup.py
	pip-compile -v

sync : requirements.txt
	pip-sync

tests :
	pytest -v -s --cov=cost_based_selection --cov-report=html --cov-report=term

build : tests

# Simulation
# ==========

# General configuration
# ---------------------

SIMULATION_ROOT = workspace/simulations
NUM_NODES = 1000
BATCH_SIZE = 100
NUM_BATCHES_train = 50
NUM_BATCHES_test = 50
BATCH_INDICES_train = $(shell seq ${NUM_BATCHES_train})
BATCH_INDICES_test = $(shell seq ${NUM_BATCHES_test})
SEED_train =
SEED_test = 999
MODELS = ba dmX
SPLITS = train test

# Targets for different models
# ----------------------------

SIMULATIONS_ba_train = $(addprefix ${SIMULATION_ROOT}/ba/train/,${BATCH_INDICES_train:=.pkl})
SIMULATIONS_ba_test = $(addprefix ${SIMULATION_ROOT}/ba/test/,${BATCH_INDICES_test:=.pkl})
SIMULATIONS_ba = ${SIMULATIONS_ba_train} ${SIMULATIONS_ba_test}

SIMULATIONS_dmX_train = $(addprefix ${SIMULATION_ROOT}/dmX/train/,${BATCH_INDICES_train:=.pkl})
SIMULATIONS_dmX_test = $(addprefix ${SIMULATION_ROOT}/dmX/test/,${BATCH_INDICES_test:=.pkl})
SIMULATIONS_dmX = ${SIMULATIONS_dmX_train} ${SIMULATIONS_dmX_test}

SIMULATIONS = ${SIMULATIONS_ba} ${SIMULATIONS_dmX}

# The pattern will be {model}/{split}/{batch}
${SIMULATIONS} : ${SIMULATION_ROOT}/%.pkl : scripts/generate_reference_table.py
	NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python $< \
		--seed=${SEED_$(call wordx,$*,2,/)}$(call wordx,$*,3,/) $(call wordx,$*,1,/) ${NUM_NODES} \
		${BATCH_SIZE} $@

# Grouped targets
# ---------------

SIMULATIONS_models = $(addprefix ${SIMULATION_ROOT}/,${MODELS})
SIMULATIONS_models_splits = $(foreach m,${MODELS},$(foreach s,${SPLITS},${SIMULATION_ROOT}/${m}/${s}))
${SIMULATION_ROOT} : ${SIMULATIONS_models}
${SIMULATIONS_models} : ${SIMULATION_ROOT}/% : $$(addprefix $$@/,${SPLITS})
${SIMULATIONS_models_splits} : ${SIMULATION_ROOT}/% : \
	$$(addprefix $$@/,$${BATCH_INDICES_$$(call wordx,$$*,2,/):=.pkl})

