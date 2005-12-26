# $Id$
#
# make stuff specific to the pips project

include $(ROOT)/makes/define_libraries.mk

CPPFLAGS += -DUTC_DATE='"$(UTC_DATE)"'

# issue about bad header files.
ifdef INC_TARGET

# force it again as the first pass was partly wrong
# because of cross dependencies between library headers
phase3: build-header-file .build_inc

endif

# build pips executables on request
$(ARCH)/pips:
	$(LINK) $@ $(ARCH)/$(PIPS_MAIN) -lpips $(PIPS_LIBS)

$(ARCH)/tpips:
	$(LINK) $@ $(TPIPS_LDFLAGS) \
		$(ARCH)/$(TPIPS_MAIN) -ltpips $(PIPS_LIBS) $(TPIPS_LIBS)

$(ARCH)/wpips:
	$(LINK) $@ $(WPIPS_LDFLAGS) \
		$(ARCH)/$(WPIPS_MAIN) -lwpips $(PIPS_LIBS) $(WPIPS_LIBS)

$(ARCH)/fpips:
	$(LINK) $@ $(FPIPS_LDFLAGS) \
		$(ARCH)/$(FPIPS_MAIN) -lfpips $(PIPS_LIBS) $(FPIPS_LIBS)

test: $(ARCH)/pips
ttest: $(ARCH)/tpips
wtest: $(ARCH)/wpips
ftest: $(ARCH)/fpips
