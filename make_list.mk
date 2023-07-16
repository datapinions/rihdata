# This is a makefile that helps us create the top N CBSA
# list that the main Makefile makes use of to create various
# files for each of these CBSAs. We do this in a separate
# file because we ran into some recursion issues when trying
# do do this in the main Makefile.

$(TOP_N_LIST_FILE):
	mkdir -p $(@D)
	$(PYTHON) -m rih.topn -n $(N) -v $(YEAR) > $@
