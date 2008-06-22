# $Id$
#
# make stuff specific to the pips project
#
# BEWARE! THIS IS OBSOLETE...
#

ifdef LIB_TARGET

$(ARCH)/pips: $(PIPS_ROOT)/lib/$(ARCH)/libpips.a
$(ARCH)/tpips: $(PIPS_ROOT)/lib/$(ARCH)/libtpips.a
$(ARCH)/wpips: $(PIPS_ROOT)/lib/$(ARCH)/libwpips.a

# building a test executable in a library
test:; $(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/pips
ttest:;	$(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/tpips
wtest:;	$(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/wpips
ftest:;	$(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/fpips

endif # LIB_TARGET
