# Copyright (c) 2023 - Darren Erik Vengroff

PYTHON = python3.9
LOGLEVEL = WARNING

# The year of U.S. Census ACS data that drives all of
# our analysis.
YEAR := 2021

# We will do our analysis on the top N CBASs. When testing
# and debugging, it is sometimes useful to override this
# with a smaller value on the command line. For example,
#
#    gmake N=3
#
# will only work with the three largest CBSAs.
N := 50

# The various directories where we will store intermediate
# and final results. Most are mentioned in .gitignore so
# their contents don't end up in the repository.
BUILD_DIR := ./build-$(YEAR)
WORKING_DIR := $(BUILD_DIR)/working
DATA_DIR := ./data-$(YEAR)
PARAMS_DIR := $(BUILD_DIR)/params
PLOT_DIR := ./plots-$(YEAR)
PRICE_PLOT_DIR := $(PLOT_DIR)/price-income
PRICE_FEATURE_PLOT_DIR := $(PLOT_DIR)/price-feature
SHAP_PLOT_DIR := $(PLOT_DIR)/shap

GROUP_HISPANIC_LATINO = --group-hispanic-latino

# The goal here is to construct a variable listing the top N
# CBSAs by population in the year specified by $(YEAR). But we
# don't want to download the data and do the analysis every time
# we invoke this Makefile. Instead, we want to cache the results
# in a file and read them from the file whenever possible. To
# accomplish this, we use another small makefile to ensure that
# the file is there. If it is, then the recursive make does
# nothing and we quickly have access to the contents of the file
# that tells us to the top N CBSAs. This variable then flows
# down through other variables in this Makefile that describe
# collections of files that should be generated for the top N
# CBSAs.
TOP_N_LIST_FILE := $(WORKING_DIR)/top_$(N)_$(YEAR)_cbsa.txt
TOP_N := $(shell $(MAKE) -s -f make_list.mk PYTHON=$(PYTHON) TOP_N_LIST_FILE=$(TOP_N_LIST_FILE) N=$(N) YEAR=$(YEAR); cat $(TOP_N_LIST_FILE))

# Paths for data for each of the top N CBSAs.
TOP_N_DATA := $(patsubst %,$(DATA_DIR)/%,$(TOP_N))

# Summary statistics about the data for each CBSA and overall
TOP_N_SUMMARY_STATS := $(TOP_N_DATA:%.geojson=%-summary.csv)
OVERALL_SUMMARY := $(DATA_DIR)/overall-summary-$(YEAR).csv

# These are additional files and directories derived from the list of top
# N data paths.
#
# Pattern-matching rules that come later will be used to generate
# the individual files listed in these variables.
TOP_N_PARAMS := $(TOP_N_DATA:$(DATA_DIR)/%.geojson=$(PARAMS_DIR)/%.params.yaml)
TOP_N_LINREG := $(TOP_N_DATA:$(DATA_DIR)/%.geojson=$(PARAMS_DIR)/%.linreg.yaml)
TOP_N_PRICE_PLOT_DIRS := $(TOP_N_DATA:$(DATA_DIR)/%.geojson=$(PRICE_PLOT_DIR)/%/price-income.png)
TOP_N_PRICE_FEATURE_PLOT_DIRS := $(TOP_N_DATA:$(DATA_DIR)/%.geojson=$(PRICE_FEATURE_PLOT_DIR)/%)
TOP_N_SHAP_PLOT_DIRS := $(TOP_N_DATA:$(DATA_DIR)/%.geojson=$(SHAP_PLOT_DIR)/%)

# An output file that ranks the performance of the model
# on the top N CBSAs. This is the top level output that
# our default targer `all` builds along with plots.
RANKED_FILE :=  $(PARAMS_DIR)/ranked_$(N)_$(YEAR)_cbsa.csv

.PHONY: all all_plots shap_plots price_plots paper_plots data summary params linreg clean clean_plots ranked_file

all: summary ranked_file all_plots

all_plots: shap_plots price_plots price_feature_plots

shap_plots: $(TOP_N_SHAP_PLOT_DIRS)

price_plots: $(TOP_N_PRICE_PLOT_DIRS)

price_feature_plots: $(TOP_N_PRICE_FEATURE_PLOT_DIRS)

params: $(TOP_N_PARAMS)

linreg: $(TOP_N_LINREG)

data: $(TOP_N_DATA)

summary: $(TOP_N_SUMMARY_STATS) $(OVERALL_SUMMARY)

ranked_file: $(RANKED_FILE)

clean: clean_plots
	rm -rf $(DATA_DIR)

clean_plots:
	rm -rf $(PLOT_DIR)

# Build data files, one for each of the top N CBSAs.
# Note that we use the &: rule syntax here. This was
# introduced in GNU Make 4.3. If you are using an older
# version, it will not properly recognize the intent
# of this rule.
$(TOP_N_DATA) &:
	mkdir -p $(DATA_DIR)
	$(PYTHON) -m rih.datagen -c $(TOP_N) -v $(YEAR) -o $(DATA_DIR)

# How to generate a file of summary stats for a data file.
%-summary.csv: %.geojson
	$(PYTHON) -m rih.summary --log $(LOGLEVEL) -o $@ $<

$(OVERALL_SUMMARY): $(TOP_N_DATA)
	$(PYTHON) -m rih.summary --log $(LOGLEVEL) -o $@ $(TOP_N_DATA)

# How to go from a data file for a single CBSA to a parameter file.
# for the same CBSA.
$(PARAMS_DIR)/%.params.yaml: $(DATA_DIR)/%.geojson
	mkdir -p $(@D)
	$(PYTHON) -m rih.treegress --log $(LOGLEVEL) -v $(YEAR) $(GROUP_HISPANIC_LATINO) -o $@ $<

# How to build the file that ranks the CBSAs by score. This is a
# summary file that is useful for undestanding which CBSAs fit
# well and which did not fit as well. It requires
# a parameter file from each of the top N CBSAs. It also requires
# a linear regression results file for each of them, since it puts
# these scores in the output file also.
$(RANKED_FILE): $(TOP_N_PARAMS) $(TOP_N_LINREG)
	mkdir -p $(@D)
	$(PYTHON) -m rih.rankscore -o $@ $(TOP_N_PARAMS)

# This is the rule to run a linear regression for a single CBSA.
# It reguires the data from that CBSA..
$(PARAMS_DIR)/%.linreg.yaml: $(DATA_DIR)/%.geojson
	$(PYTHON) -m rih.linreg --log $(LOGLEVEL) -v $(YEAR) $(GROUP_HISPANIC_LATINO) -o $@ $<

# Produce a plot of price vs. income for a single CBSA. All of
# the block groups in that CBSA are considered.
$(PRICE_PLOT_DIR)/%/price-income.png: $(DATA_DIR)/%.geojson
	mkdir -p ${@D}
	$(PYTHON) -m rih.priceplot --log $(LOGLEVEL) -v $(YEAR) -o $@ $<

# Produce a series of plots for the influence of each of several
# features on the output of the model for a single CBSA. All of
# the block groups in that CBSA are considered. Since the shap analysis
# is the slow part of this, and it produces values for all features at
# the same time, we organize the code so that one executable produces
# plots for all features.
$(SHAP_PLOT_DIR)/%: $(PARAMS_DIR)/%.params.yaml $(DATA_DIR)/%.geojson
	mkdir -p $@
	$(PYTHON) -m rih.shapplot --log $(LOGLEVEL) --background -v $(YEAR) $(GROUP_HISPANIC_LATINO) -p $(PARAMS_DIR)/$*.params.yaml -o $@ $(DATA_DIR)/$*.geojson
	touch $@

# Produce a series of scatter plots of price vs. the various race and
# ethnicity features.
$(PRICE_FEATURE_PLOT_DIR)/%: $(DATA_DIR)/%.geojson
	mkdir -p $@
	$(PYTHON) -m rih.featureplot --log $(LOGLEVEL) -v $(YEAR) $(GROUP_HISPANIC_LATINO) -o $@ $(DATA_DIR)/$*.geojson
	touch $@

# Special plots for the paper.
paper_plots: $(PLOT_DIR)/paper/Miami-Fort_Lauderdale-Pompano_Beach,_FL_Metro_Area/33100/750-15.png \
    $(PLOT_DIR)/paper/Miami-Fort_Lauderdale-Pompano_Beach,_FL_Metro_Area/33100/750-5.png \
    $(PLOT_DIR)/paper/Miami-Fort_Lauderdale-Pompano_Beach,_FL_Metro_Area/33100/500-40.png \
    $(PLOT_DIR)/paper/Miami-Fort_Lauderdale-Pompano_Beach,_FL_Metro_Area/33100/500-5.png \
    $(PLOT_DIR)/paper/Miami-Fort_Lauderdale-Pompano_Beach,_FL_Metro_Area/33100/500-1.png

$(PLOT_DIR)/paper/Miami-Fort_Lauderdale-Pompano_Beach,_FL_Metro_Area/33100/750-15.png:
	mkdir -p $(@D)
	$(PYTHON) -m rih.featureplot --log $(LOGLEVEL) -v $(YEAR) $(GROUP_HISPANIC_LATINO) \
    --feature frac_B03002_004E --highlight-feature-above 0.15 --emphasize-value-above 750000 \
    -o $(@D) -F $(@F) $(DATA_DIR)/Miami-Fort_Lauderdale-Pompano_Beach,_FL_Metro_Area/33100.geojson

$(PLOT_DIR)/paper/Miami-Fort_Lauderdale-Pompano_Beach,_FL_Metro_Area/33100/750-5.png:
	mkdir -p $(@D)
	$(PYTHON) -m rih.featureplot --log $(LOGLEVEL) -v $(YEAR) $(GROUP_HISPANIC_LATINO) \
    --feature frac_B03002_004E --highlight-feature-below 0.05 --emphasize-value-above 750000 \
    -o $(@D) -F $(@F) $(DATA_DIR)/Miami-Fort_Lauderdale-Pompano_Beach,_FL_Metro_Area/33100.geojson

$(PLOT_DIR)/paper/Miami-Fort_Lauderdale-Pompano_Beach,_FL_Metro_Area/33100/500-40.png:
	mkdir -p $(@D)
	$(PYTHON) -m rih.featureplot --log $(LOGLEVEL) -v $(YEAR) $(GROUP_HISPANIC_LATINO) \
    --feature frac_B03002_004E --highlight-feature-above 0.4 --emphasize-value-above 500000 \
    -o $(@D) -F $(@F) $(DATA_DIR)/Miami-Fort_Lauderdale-Pompano_Beach,_FL_Metro_Area/33100.geojson

$(PLOT_DIR)/paper/Miami-Fort_Lauderdale-Pompano_Beach,_FL_Metro_Area/33100/500-5.png:
	mkdir -p $(@D)
	$(PYTHON) -m rih.featureplot --log $(LOGLEVEL) -v $(YEAR) $(GROUP_HISPANIC_LATINO) \
    --feature frac_B03002_004E --highlight-feature-below 0.05 --emphasize-value-above 500000 \
    -o $(@D) -F $(@F) $(DATA_DIR)/Miami-Fort_Lauderdale-Pompano_Beach,_FL_Metro_Area/33100.geojson

$(PLOT_DIR)/paper/Miami-Fort_Lauderdale-Pompano_Beach,_FL_Metro_Area/33100/500-1.png:
	mkdir -p $(@D)
	$(PYTHON) -m rih.featureplot --log $(LOGLEVEL) -v $(YEAR) $(GROUP_HISPANIC_LATINO) \
    --feature frac_B03002_004E --highlight-feature-below 0.01 --emphasize-value-above 500000 \
    -o $(@D) -F $(@F) $(DATA_DIR)/Miami-Fort_Lauderdale-Pompano_Beach,_FL_Metro_Area/33100.geojson


