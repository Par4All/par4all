# $Id$
# some local add-on for pips compilation

clean: local-clean
build: local-clean
compile: local-clean

recompile: local-clean
	$(MAKE) -C src

# set ARCH macro
MAKE.d	= ./makes
include $(MAKE.d)/arch.mk

FINDsrc	= find ./src -name '.svn' -type d -prune -o

local-clean:
	# remove executables to know if the compilation failed
	$(RM) bin/$(ARCH)/* lib/$(ARCH)/lib*.a lib/$(ARCH)/*.o
	$(FINDsrc) -name .build_lib.$(ARCH) -print0 | xargs -0 rm -f
