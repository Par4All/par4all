#
# $Id$
# 
# Source, header and object files used to build the library.
# Do not include the main program source file.

LIB_CFILES	= \
	proper_effects_engine.c \
	rw_effects_engine.c \
	in_effects_engine.c \
	out_effects_engine.c \
	mappings.c \
	unary_operators.c \
	binary_operators.c \
	utils.c \
	prettyprint.c \
	intrinsics.c

# 	interprocedural.c

LIB_HEADERS	= effects-generic-local.h

LIB_OBJECTS	= $(LIB_CFILES:%.c=%.o)

PIPS_NEW_LIBS	= \
	-ltop-level -lpipsmake -lwp65 -lhpfc -ltransformations -lmovements \
	-lbootstrap -lcallgraph -licfg -lchains -lcomplexity -lconversion \
	-lprettyprint -latomizer -lprgm_mapping -lscheduling -lreindexing \
	-larray_dfg -lpaf-util -lstatic_controlize -lsyntax \
	-leffects-simple -leffects-convex -leffects-generic \
	-lalias-classes -lcomp_sections -lcontrol -lsemantics -lcontinuation \
	-lrice -lricedg -lpipsdbm -ltransformer -lpip -lri-util \
	-lproperties -ltext-util -lmisc -lproperties -lreductions -lflint \
	$(NEWGEN_LIBS) $(LINEAR_LIBS) -lm -lrx

#
# While developping generic effects.
#

ntest: $(ARCH)/pips
nttest: $(ARCH)/tpips

$(ARCH)/pips: all
	$(LINK) $(ARCH)/pips -lpips $(PIPS_NEW_LIBS) 

$(ARCH)/tpips: all
	$(LINK) $(ARCH)/tpips -ltpips $(PIPS_NEW_LIBS) $(TPIPS_ADDED_LIBS)
