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

endif # INC_TARGET

ifdef BIN_TARGET
include $(ROOT)/makes/pips-bin.mk
endif # BIN_TARGET

ifdef LIB_TARGET
ifdef OLD_TEST 

# build pips executables on request?
include $(ROOT)/makes/pips-old.mk

ifndef BIN_TARGET
include $(ROOT)/makes/pips-bin.mk
endif # BIN_TARGET

else # not OLD_TEST

# simply link to actual executable, helpful for gdb
ifndef BIN_TARGET
$(ARCH)/tpips:
	$(RM) $@
	ln -s $(PIPS_ROOT)/bin/$@ $@

$(ARCH)/pips:
	$(RM) $@
	ln -s $(PIPS_ROOT)/bin/$@ $@

# full recompilation from a library
full: $(ARCH)/tpips $(ARCH)/pips
	$(MAKE) -C $(PIPS_ROOT) compile

# fast tpips recompilation
fast-tpips: $(ARCH)/tpips compile
	$(MAKE) -C $(PIPS_ROOT)/src/Passes/tpips compile

# fast pips recompilation
fast-pips: $(ARCH)/pips compile
	$(MAKE) -C $(PIPS_ROOT)/src/Passes/pips compile

# generate both pips and tpips, useful for validation
fast: fast-tpips fast-pips

# helper with old targets
test ttest ftest: 
	@echo -e "\a\n\ttry 'fast' (just link) or 'full' (recompilation)\n"

endif # BIN_TARGET
endif # OLD_TEST
endif # LIB_TARGET
