# $Id$
#
# make stuff specific to the pips project

include $(ROOT)/makes/define_libraries.mk

CPPFLAGS += -DUTC_DATE='$(UTC_DATE)'

# issue about bad header files.
ifdef INC_TARGET

# force it again as the first pass was partly wrong
# because of cross dependencies between library headers
# however, do not repeat that every time...
phase3: .build_inc_second_pass

.build_inc_second_pass:
	$(MAKE) build-header-file .build_inc
	touch $@

clean: pips-phase3-clean

pips-phase3-clean:
	$(RM) .build_inc_second_pass

endif

main.dir =	./$(ARCH)

# build pips executables on request
$(ARCH)/pips:
	$(MAKE) $(ARCH)
	$(LINK) $@ $(main.dir)/$(PIPS_MAIN) -lpips $(PIPS_LIBS)

$(ARCH)/tpips:
	$(MAKE) $(ARCH)
	$(LINK) $@ $(TPIPS_LDFLAGS) \
		$(main.dir)/$(TPIPS_MAIN) -ltpips $(PIPS_LIBS) $(TPIPS_LIBS)

$(ARCH)/wpips:
	$(MAKE) $(ARCH)
	$(LINK) $@ $(WPIPS_LDFLAGS) \
		$(main.dir)/$(WPIPS_MAIN) -lwpips $(PIPS_LIBS) $(WPIPS_LIBS)

$(ARCH)/fpips:
	$(MAKE) $(ARCH)
	$(LINK) $@ $(FPIPS_LDFLAGS) \
		$(main.dir)/$(FPIPS_MAIN) -lfpips $(FPIPS_LIBS) $(PIPS_LIBS) 

# building a test executable in a library
test:; $(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/pips
ttest:;	$(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/tpips
wtest:;	$(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/wpips
ftest:;	$(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/fpips

ifdef LIB_TARGET
# fix local library dependency
$(ARCH)/pips: $(ARCH)/$(LIB_TARGET)
$(ARCH)/tpips: $(ARCH)/$(LIB_TARGET)
$(ARCH)/wpips: $(ARCH)/$(LIB_TARGET)
$(ARCH)/fpips: $(ARCH)/$(LIB_TARGET)
endif
