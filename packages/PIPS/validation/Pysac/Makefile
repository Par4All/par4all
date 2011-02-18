# $Id$

# the parallel validation does not work
# because of collisions in the local directory,
# e.g. basics0.o a.out
#PARALLEL_VALIDATION = 1
include ../validate.mk

clean: local-clean
local-clean:
	$(RM) -r d.out .PYPS*.tmp .w0.tmp .loop_tiling.tmp .exception.tmp .effects.tmp \
	PYPS[-_a-zA-Z0-9][-_a-zA-Z0-9][-_a-zA-Z0-9][-_a-zA-Z0-9][-_a-zA-Z0-9][-_a-zA-Z0-9]
