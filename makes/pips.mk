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

full-ttest: $(ARCH)/tpips
	$(MAKE) compile
	$(MAKE) -C $(PIPS_ROOT) compile

fast-ttest: $(ARCH)/tpips
	$(MAKE) compile
	$(MAKE) -C $(PIPS_ROOT)/src/Passes/tpips compile

ttest: fast-ttest

endif # BIN_TARGET
endif # OLD_TEST
endif # LIB_TARGET
