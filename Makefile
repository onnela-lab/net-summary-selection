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


# Ranking
# =======

METHODS = JMI JMIM mRMR reliefF pen_rf_importance weighted_rf_importance
PENALTIES = 0.0 0.0125 0.025 0.05 0.1 0.2 0.4 0.8 1.6 3.2 6.4 12.8 25.6 51.2
RANKING_ROOT = workspace/rankings
RANKING_TARGETS_models = $(addprefix ${RANKING_ROOT}/,${MODELS})
RANKING_TARGETS_models_methods = $(foreach m,${RANKING_TARGETS_models},$(foreach k,${METHODS},${m}/${k}))
RANKING_TARGETS = $(foreach mk,${RANKING_TARGETS_models_methods},$(foreach p,${PENALTIES},${mk}/${p}.pkl))

${RANKING_ROOT} : $(addprefix ${RANKING_ROOT}/,${MODELS})
${RANKING_TARGETS_models} : ${RANKING_ROOT}/% : $$(addprefix $$@/,$${METHODS})
${RANKING_TARGETS_models_methods} : ${RANKING_ROOT}/% : $$(addprefix $$@/,$${PENALTIES:=.pkl})

${RANKING_TARGETS} : ${RANKING_ROOT}/%.pkl : scripts/rank_features.py $${SIMULATIONS_$$(call wordx,$$*,1,/)_train}
	NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python $< --penalty=$(call wordx,$*,3,/) \
		$(call wordx,$*,2,/) $@ $(filter-out $<,$^)
