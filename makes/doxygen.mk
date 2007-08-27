DOXYGEN_GENERATED_DIR = html latex

# To bublish on a WWW server:
RSYNC = rsync --archive --hard-links --delete --force --partial --compress --verbose

# Where we want the documentation to be published:
PUBLISH_LOCATION := doxygen.pips.enstb.org:/var/www/pips/doxygen/$(PROJECT_NAME)

DEFAULT_DOXYGEN_CONFIG ?= $(ROOT)/makes/share/doxygen/Doxyfile

# Now some high end hackery since I cannot send a variable content with newlines to the shell:

# Put an eol char in this variable:
define eol_char


endef
# In what we transform an eol for shell communication:
bn=\\n

# Transform all the newlines to the 2 characters string '\n' :
DOXYGEN_PARAMETERS_WITHOUT_EOL := $(subst $(eol_char),$(bn),$(DOXYGEN_PARAMETERS))

# Add some global Doxygen parameters to the configration file:
DOXYGEN_PARAMETERS_WITHOUT_EOL += $(bn)PROJECT_NAME           = $(PROJECT_NAME)

ifdef GENERATE_TAGFILE_NAME
	# Tags are generated for each version, which is overkill, but KISS.
	DOXYGEN_PARAMETERS_WITHOUT_EOL += $(bn)GENERATE_TAGFILE=$(GENERATE_TAGFILE_NAME)
endif


# We generate 2 versions, one without callers/callees graphs, and another full-fledged heavy one:
doxygen : doxygen-plain doxygen-graph

doxygen-plain : OUTPUT_DIRECTORY       = plain
doxygen-plain : DOXYGEN_MORE_PARAMETERS = $(bn)OUTPUT_DIRECTORY       = $(OUTPUT_DIRECTORY)

doxygen-plain : do-doxygen

doxygen-graph : DOXYGEN_MORE_PARAMETERS = $(bn)OUTPUT_DIRECTORY       = graph$(bn)HAVE_DOT               = YES

doxygen-graph : do-doxygen

# Now do the reverse transformation once in the shell:
do-doxygen :
	( cat $(DEFAULT_DOXYGEN_CONFIG); echo "$(DOXYGEN_PARAMETERS_WITHOUT_EOL)$(DOXYGEN_MORE_PARAMETERS)" | sed s/\\\\n/$(bn)/g ) | doxygen -

ifdef PUBLISH_LOCATION

publish: make_destination_dir
	$(RSYNC) plain/html/ $(PUBLISH_LOCATION)/plain
	$(RSYNC) graph/html/ $(PUBLISH_LOCATION)/graph


# Just to avoid publish to complaining if not implemented in the including
# Makefile:
make_destination_dir :

endif
