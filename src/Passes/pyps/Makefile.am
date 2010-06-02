BUILT_SOURCES = pypips_wrap.c  pypips_aux.c
dist_noinst_DATA=_pyps.py pypips.i.in pypips.c pipsmakerc2python.pl

dist_bin_SCRIPTS =	ipyps

dist_pkgpython_PYTHON = pyps.py pipscc.py pii.py
nodist_pkgpython_PYTHON=pypips.py
pkgpyexec_LTLIBRARIES = _pypips.la
_pypips_la_SOURCES = pypips_aux.c
nodist__pypips_la_SOURCES =pypips_wrap.c

_pypips_la_LDFLAGS = -module $(PYTHON_LDFLAGS)
_pypips_la_LIBADD = ../../Libs/libpipslibs.la $(PYTHON_EXTRALIBS) $(LINEARLIBS_LIBS) $(NEWGENLIBS_LIBS)
_pypips_la_CPPFLAGS = \
	$(PYTHON_CPPFLAGS)\
	-I$(top_srcdir)/src/Documentation/constants \
	-I../../Documentation/pipsmake \
	-I../../Documentation/newgen \
	-I../../Libs/misc\
	-I../../Libs/newgen\
	-I../../Libs/ri-util\
	-I../../Libs/top-level\
	-I../../Libs/pipsmake\
	-I../../Libs/pipsdbm\
	-I../../Libs/properties\
	-I../../Libs/transformations\
	$(LINEARLIBS_CFLAGS)\
	$(NEWGENLIBS_CFLAGS)

pypips_wrap.c : pypips.i pypips.h
	$(SWIG) -python -o pypips_wrap.c-tmp pypips.i
	cat pypips.h pypips_wrap.c-tmp > pypips_wrap.c
	rm -f pypips_wrap.c-tmp


pypips.i:pypips.i.in pypips.h
	sed -e 's/SWIG_MODULE_NAME/pypips/' $(srcdir)/pypips.i.in pypips.h > pypips.i

pypips.h:pypips.c Makefile
	echo "#ifdef HAVE_CONFIG_H" > pypips.h
	echo '#include "pips_config.h"' >> pypips.h
	echo '#endif' >> pypips.h
	$(CPROTO) -evcf2 -E "$(CPP) $(_pypips_la_CPPFLAGS) $(INCLUDES) $(DEFAULT_INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) -DCPROTO_IS_PROTOTYPING" $(srcdir)/pypips.c >> pypips.h
	echo "#include <unistd.h>" >> pypips.h
	echo '#include "linear.h"' >> pypips.h
	echo '#include "genC.h"' >> pypips.h
	echo '#include "ri.h"' >> pypips.h
	echo '#include "text.h"' >> pypips.h
	echo '#include "pipsmake.h"' >> pypips.h
	echo '#include "top-level.h"' >> pypips.h
	echo '#include "transformations.h"' >> pypips.h
	echo "#undef _POSIX_C_SOURCE" >> pypips.h
	echo "#undef assert" >> pypips.h
	echo "extern char *activate(char *);" >> pypips.h
	echo "extern char *module_loops(const char*,const char*);" >> pypips.h


pypips_aux.c:pypips.h pypips.c
	cat pypips.h $(srcdir)/pypips.c > pypips_aux.c

p2p_cmd= $(srcdir)/pipsmakerc2python.pl $(top_srcdir)/src/Documentation/pipsmake/pipsmake-rc.tex ../../Documentation/pipsmake/properties.rc ../../Documentation/pipsmake/pipsdep.rc
pyps.py:$(p2p_cmd) _pyps.py 
	sed -n -e '1,/### loop_methods/ p' $(srcdir)/_pyps.py > pyps.py
	$(PERL) $(p2p_cmd) -loop >> pyps.py
	sed -n -e '/### loop_methods/,/### module_methods/ p' $(srcdir)/_pyps.py >> pyps.py
	$(PERL) $(p2p_cmd) -module >> pyps.py
	sed -n -e '/### module_methods/,/### modules_methods/ p' $(srcdir)/_pyps.py >> pyps.py
	$(PERL) $(p2p_cmd) -modules >> pyps.py
	sed -n -e '/### modules_methods/,$$ p' $(srcdir)/_pyps.py >> pyps.py

CLEANFILES=pypips_aux.c pypips.py pyps.py pypips_wrap.c pypips.i pypips.h
