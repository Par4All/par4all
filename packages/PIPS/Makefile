# $Id$
# simplistic forward makefile
# useful targets include: tags clean build unbuild rebuild...

default: compile

# can generate a tar
VERSION=0.1

.DEFAULT:
	$(MAKE) -C newgen $@
	$(MAKE) -C linear $@
	$(MAKE) -C pips $@

