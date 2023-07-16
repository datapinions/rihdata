PYTHON = python3.9
LOGLEVEL = WARNING

YEAR := 2020

BUILD_DIR := ./build-$(YEAR)
WORKING_DIR := $(BUILD_DIR)/working
DATA_DIR := ./data-$(YEAR)
PARAMS_DIR := $(BUILD_DIR)/params
PLOT_DIR := ./plots-$(YEAR)
PRICE_PLOT_DIR := $(PLOT_DIR)/price-income
SHAP_PLOT_DIR := $(PLOT_DIR)/shap

GROUP_HISPANIC_LATINO = --group-hispanic-latino
N := 50

TOP_N_LIST_FILE := $(WORKING_DIR)/top_$(N)_$(YEAR)_cbsa.txt
RANKED_FILE :=  $(PARAMS_DIR)/ranked_$(N)_$(YEAR)_cbsa.csv

TOP_N := $(shell $(MAKE) -s -f make_list.mk PYTHON=$(PYTHON) TOP_N_LIST_FILE=$(TOP_N_LIST_FILE) N=$(N) YEAR=$(YEAR); cat $(TOP_N_LIST_FILE))

TOP_N_DATA := $(patsubst %,$(DATA_DIR)/%,$(TOP_N))
TOP_N_PARAMS := $(TOP_N_DATA:$(DATA_DIR)/%.geojson=$(PARAMS_DIR)/%.params.yaml)
TOP_N_LINREG := $(TOP_N_DATA:$(DATA_DIR)/%.geojson=$(PARAMS_DIR)/%.linreg.yaml)
TOP_N_PRICE_PLOT_DIRS := $(TOP_N_DATA:$(DATA_DIR)/%.geojson=$(PRICE_PLOT_DIR)/%/price-income.png)
TOP_N_SHAP_PLOT_DIRS := $(TOP_N_DATA:$(DATA_DIR)/%.geojson=$(SHAP_PLOT_DIR)/%)

.PHONY: all plots shap_plots price_plots data params linreg clean clean_plots ranked_file

all: ranked_file plots

plots: shap_plots price_plots

shap_plots: $(TOP_N_SHAP_PLOT_DIRS)

price_plots: $(TOP_N_PRICE_PLOT_DIRS)

params: $(TOP_N_PARAMS)

linreg: $(TOP_N_LINREG)

data: $(TOP_N_DATA)

ranked_file: $(RANKED_FILE)

clean: clean_plots
	rm -rf $(DATA_DIR)

clean_plots:
	rm -rf $(PLOT_DIR)

# Build data files, one for each of the top N CBSAs.

$(TOP_N_DATA) &:
	mkdir -p $(DATA_DIR)
	$(PYTHON) -m rih.datagen -c $(TOP_N) -v $(YEAR) -o $(DATA_DIR)

# How to go from a CBSA file to a parameter file.

$(PARAMS_DIR)/%.params.yaml: $(DATA_DIR)/%.geojson
	$(PYTHON) -m rih.treegress --log $(LOGLEVEL) -v $(YEAR) $(GROUP_HISPANIC_LATINO) -o $@ $<

$(RANKED_FILE): $(TOP_N_PARAMS) $(TOP_N_LINREG)
	mkdir -p $(@D)
	$(PYTHON) -m rih.rankscore -o $@ $(TOP_N_PARAMS)

# Linear regression for comparison.
$(PARAMS_DIR)/%.linreg.yaml: $(DATA_DIR)/%.geojson
	$(PYTHON) -m rih.linreg --log $(LOGLEVEL) -v $(YEAR) $(GROUP_HISPANIC_LATINO) -o $@ $<

# Price plot
$(PRICE_PLOT_DIR)/%/price-income.png: $(DATA_DIR)/%.geojson
	mkdir -p ${@D}
	$(PYTHON) -m rih.priceplot --log $(LOGLEVEL) -v $(YEAR) -o $@ $<

# Shap plot
$(SHAP_PLOT_DIR)/%: $(PARAMS_DIR)/%.params.yaml $(DATA_DIR)/%.geojson
	mkdir -p $@
	$(PYTHON) -m rih.shapplot --log $(LOGLEVEL) --background -v $(YEAR) $(GROUP_HISPANIC_LATINO) -p $(PARAMS_DIR)/$*.params.yaml -o $@ $(DATA_DIR)/$*.geojson
