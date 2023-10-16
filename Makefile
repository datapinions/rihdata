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
PLOT_DIR := ./plots-$(YEAR)
PRICE_PLOT_DIR := $(PLOT_DIR)/price-income
PRICE_FEATURE_PLOT_DIR := $(PLOT_DIR)/price-feature

# Templates and related details for rendering the site.
HTML_TEMPLATE_DIR := ./templates
STATIC_HTML_DIR := ./static-html
SITE_DIR := $(BUILD_DIR)/site
SITE_IMAGE_DIR := $(SITE_DIR)/images

HTML_NAMES := impact.html
SITE_HTML := $(HTML_NAMES:%=$(SITE_DIR)/%)
HTML_TEMPLATES := $(HTML_NAMES:%.html=$(HTML_TEMPLATE_DIR)/%.html.j2)

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
PRICE_INCOME_FILE_NAME := Median-household-income-in-the-past-12-months-in-$(YEAR)-inflation-adjusted-dollars.png
TOP_N_PRICE_PLOTS := $(TOP_N_DATA:$(DATA_DIR)/%.geojson=$(PRICE_PLOT_DIR)/%/$(PRICE_INCOME_FILE_NAME))
TOP_N_PRICE_FEATURE_PLOT_DIRS := $(TOP_N_DATA:$(DATA_DIR)/%.geojson=$(PRICE_FEATURE_PLOT_DIR)/%)

.PHONY: all all_plots price_plots paper_plots data summary clean clean_plots dist_clean ranked_file

all: summary ranked_file all_plots

all_plots: price_plots price_feature_plots

price_plots: $(TOP_N_PRICE_PLOTS)

price_feature_plots: $(TOP_N_PRICE_FEATURE_PLOT_DIRS)

data: $(TOP_N_DATA)

summary: $(TOP_N_SUMMARY_STATS) $(OVERALL_SUMMARY)

ranked_file: $(RANKED_FILE)

site_html: $(SITE_HTML) $(SITE_PLOTS) $(SITE_IMAGE_DIR)/impact_charts $(SITE_IMAGE_DIR)/price_charts
	cp -r $(STATIC_HTML_DIR)/* $(SITE_DIR)

$(SITE_IMAGE_DIR)/price_charts: $(PRICE_FEATURE_PLOT_DIR) $(PRICE_PLOT_DIR)
	-rm -rf $@
	mkdir -p $@
	cp -r $(PRICE_FEATURE_PLOT_DIR)/* $@
	cp -r $(PRICE_PLOT_DIR)/* $@

clean: clean_plots
	rm -rf $(DATA_DIR) $(BUILD_DIR)

clean_plots:
	rm -rf $(PLOT_DIR)

dist_clean:
	rm -rf ./build-[12]??? ./data-[12]??? ./plots-[12]???

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

# Produce a plot of price vs. income for a single CBSA. All of
# the block groups in that CBSA are considered.
$(PRICE_PLOT_DIR)/%/$(PRICE_INCOME_FILE_NAME): $(DATA_DIR)/%.geojson
	mkdir -p ${@D}
	$(PYTHON) -m rih.priceplot --log $(LOGLEVEL) -v $(YEAR) -o $@ $<

# Produce a series of scatter plots of price vs. the various race and
# ethnicity features.
$(PRICE_FEATURE_PLOT_DIR)/%: $(DATA_DIR)/%.geojson
	mkdir -p $@
	$(PYTHON) -m rih.featureplot --log $(LOGLEVEL) -v $(YEAR) $(GROUP_HISPANIC_LATINO) -o $@ $(DATA_DIR)/$*.geojson
	touch $@

# How to render an HTML template for the site.
$(SITE_DIR)/%.html: $(HTML_TEMPLATE_DIR)/%.html.j2
	mkdir -p $(@D)
	$(PYTHON) -m rih.rendersite --log $(LOGLEVEL)  -v $(YEAR) -t $(TOP_N_LIST_FILE) -o $@ $<


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


