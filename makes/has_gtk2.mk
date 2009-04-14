# $Id$

ifndef has_gtk2_done

# first check that pkg-config is available? (comes with gnome?)
has_pkgcfg := $(shell type pkg-config > /dev/null 2>&1 && echo ok)

ifeq ($(has_pkgcfg),ok)

has_gtk2   := $(shell pkg-config --exists gtk+-2.0 && echo ok)

ifneq ($(has_gtk2),ok)

# no pkg-config => no gpips

$(warning "skipping gpips compilation, gtk2 is not available")
PIPS_NO_GPIPS	= 1

endif # has_gtk2

else # has_pkgcfg not ok

$(warning "skipping gpips compilation, pkg-config not found")
PIPS_NO_GPIPS	= 1

endif # has_pkgcfg

has_gtk2_done	= 1

endif # has_gtk2_done
