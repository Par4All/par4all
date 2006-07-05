# $Id$
#
# make stuff specific to the pips project

include $(ROOT)/makes/define_libraries.mk

CPPFLAGS += -DUTC_DATE='$(UTC_DATE)'

# issue about bad header files.
ifdef INC_TARGET

# force it again as the first pass was partly wrong
# because of cross dependencies between library headers
phase3: build-header-file .build_inc

endif

main.dir =	./$(ARCH)

# build pips executables on request
$(ARCH)/pips: $(ARCH)
	$(LINK) $@ $(main.dir)/$(PIPS_MAIN) -lpips $(PIPS_LIBS)

$(ARCH)/tpips: $(ARCH)
	$(LINK) $@ $(TPIPS_LDFLAGS) \
		$(main.dir)/$(TPIPS_MAIN) -ltpips $(PIPS_LIBS) $(TPIPS_LIBS)

$(ARCH)/wpips: $(ARCH)
	$(LINK) $@ $(WPIPS_LDFLAGS) \
		$(main.dir)/$(WPIPS_MAIN) -lwpips $(PIPS_LIBS) $(WPIPS_LIBS)

$(ARCH)/fpips: $(ARCH)
	$(LINK) $@ $(FPIPS_LDFLAGS) \
		$(main.dir)/$(FPIPS_MAIN) -lfpips $(PIPS_LIBS) $(FPIPS_LIBS)

# building a test executable in a library
test:; $(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/pips
ttest:;	$(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/tpips
wtest:;	$(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/wpips
ftest:;	$(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/fpips

# fix library dependency
ifdef LIB_TARGET
test ttest wtest ftest: $(ARCH)/$(LIB_TARGET)
endif
