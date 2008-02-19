# Push the make to 2 subdirectories, one for the NewGen program itself,
# and one for the NewGen libC that is interesting for programs using
# NewGen infrastructure.

TARGETS = newgen libgenC
NEWGEN_SOURCE_DIR = ../../../genC
export NEWGEN_SOURCE_DIR
# Forward any command to the specified directories:
.DEFAULT:
	for d in $(TARGETS) ; do $(MAKE) -C $$d $(FWD_MKFLAGS); done

#include ../../../makes/doxygen.mk
