# $Id$

# build pips executables on request
main.dir =	./$(ARCH)

$(ARCH)/pips:
	$(MAKE) $(ARCH)
	$(LINK) $@ $(main.dir)/$(PIPS_MAIN) -lpips $(addprefix -l,$(pips.libs))

$(ARCH)/tpips:
	$(MAKE) $(ARCH)
	$(LINK) $@ $(TPIPS_LDFLAGS) \
		$(main.dir)/$(TPIPS_MAIN) -ltpips $(addprefix -l,$(tpips.libs))

$(ARCH)/wpips:
	$(MAKE) $(ARCH)
	$(LINK) $@ $(WPIPS_LDFLAGS) \
		$(main.dir)/$(WPIPS_MAIN) -lwpips $(addprefix -l,$(wpips.libs))

$(ARCH)/fpips:
	$(MAKE) $(ARCH)
	$(LINK) $@ $(FPIPS_LDFLAGS) \
		$(main.dir)/$(FPIPS_MAIN) -lfpips $(addprefix -l,$(fpips.libs))

# all libraries as installed...
PIPSLIBS_LIBS	= \
	$(addsuffix .a, \
		$(addprefix $(PIPS_ROOT)/lib/$(ARCH)/lib,$(pipslibs.libs)))

NEWGEN_LIBS	= \
	$(addsuffix .a, \
		$(addprefix $(NEWGEN_ROOT)/lib/$(ARCH)/lib,$(newgen.libs)))

LINEAR_LIBS	= \
	$(addsuffix .a, \
		$(addprefix $(LINEAR_ROOT)/lib/$(ARCH)/lib,$(linear.libs)))

# add other pips dependencies...
$(ARCH)/pips \
$(ARCH)/tpips \
$(ARCH)/wpips \
$(ARCH)/fpips: \
	$(PIPSLIBS_LIBS) \
	$(NEWGEN_LIBS) \
	$(LINEAR_LIBS)

ifdef LIB_TARGET

# fix local library dependency
$(ARCH)/pips \
$(ARCH)/tpips \
$(ARCH)/wpips \
$(ARCH)/fpips: \
	$(ARCH)/$(LIB_TARGET)

endif # LIB_TARGET
