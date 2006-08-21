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

# building a test executable in a library
test:; $(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/pips
ttest:;	$(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/tpips
wtest:;	$(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/wpips
ftest:;	$(MAKE) main.dir=$(PIPS_ROOT)/lib/$(ARCH) $(ARCH)/fpips

ifdef LIB_TARGET
# fix local library dependency
$(ARCH)/pips \
$(ARCH)/tpips \
$(ARCH)/wpips \
$(ARCH)/fpips: \
	$(ARCH)/$(LIB_TARGET)
endif

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

$(ARCH)/pips: $(PIPS_ROOT)/lib/$(ARCH)/libpips.a
$(ARCH)/tpips: $(PIPS_ROOT)/lib/$(ARCH)/libtpips.a
$(ARCH)/wpips: $(PIPS_ROOT)/lib/$(ARCH)/libwpips.a
