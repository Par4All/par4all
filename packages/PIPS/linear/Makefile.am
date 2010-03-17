SUBDIRS=src

# pkgconfig stuff
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA= linearlibs.pc

# config.h stuff 
# not used yet
#include_HEADERS= linear_config.h
#dist_noinst_HEADERS=_config.h
#linear_config.h:config.h _config.h
#	$(SED) -r -e 's/#define ([A-Za-z_]+) (.*)/#ifndef \1\n #define \1 \2\n#endif/g' $^ > $@

# aclocal stuff
ACLOCAL_AMFLAGS=-I makes/m4
