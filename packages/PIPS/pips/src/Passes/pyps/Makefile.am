SUBDIRS  = doc drivers impl
BUILT_SOURCES = pypips_wrap.c  pypips_aux.c

dist_noinst_DATA=pypips.i.in pypips.c pipsmakerc2python.py pypips.h.in pips-make.in pypsconfig.py.in

dist_bin_SCRIPTS = pips-gcc pips-ar pips-make ipyps

pipsmakeenvdir = $(libdir)/pips-make
dist_pipsmakeenv_SCRIPTS = gcc ar

dist_pkgpython_PYTHON = pypsbase.py pypsex.py pyps.py pipscc.py pii.py pipsgcc.py pypsutils.py pyrops.py apypips.py pypsconfig.py
dist_python_DATA = pips.pth

if WITH_PYPS_EXTRA
	dist_pkgpython_PYTHON += pyrops.py
endif

nodist_pkgpython_PYTHON=pypips.py
pkgpyexec_LTLIBRARIES = _pypips.la
_pypips_la_SOURCES = pypips_aux.c
nodist__pypips_la_SOURCES =pypips_wrap.c

_pypips_la_LDFLAGS = -module $(PYTHON_LDFLAGS)
_pypips_la_LIBADD = ../../Libs/libpipslibs.la $(PYTHON_EXTRALIBS) $(LINEARLIBS_LIBS) $(NEWGENLIBS_LIBS)
_pypips_la_CPPFLAGS = \
	-I../../Libs/gnulib \
	$(PYTHON_CPPFLAGS)\
	-I$(top_srcdir)/src/Documentation/constants \
	-I../../Documentation/pipsmake \
	-I../../Documentation/newgen \
	-I../../Libs/misc\
	-I../../Libs/newgen\
	-I../../Libs/ri-util\
	-I../../Libs/text-util\
	-I../../Libs/top-level\
	-I../../Libs/pipsmake\
	-I../../Libs/pipsdbm\
	-I../../Libs/properties\
	-I../../Libs/transformations\
	$(LINEARLIBS_CFLAGS)\
	$(NEWGENLIBS_CFLAGS)

pypips_wrap.c : pypips.i pypips.h
	$(AM_V_GEN)$(SWIG) -python -o pypips_wrap.c-tmp pypips.i
	$(AM_V_GEN)cat pypips.h pypips_wrap.c-tmp > pypips_wrap.c
	@rm -f pypips_wrap.c-tmp


pypips.i:pypips.i.in pypips.h
	$(AM_V_GEN)sed -e 's/SWIG_MODULE_NAME/pypips/' $(srcdir)/pypips.i.in pypips.h > pypips.i

pypips.h:pypips.c pypips.h.in Makefile
	$(AM_V_GEN)sed -n -e '1,/\/\*CPROTO pyips\.h/ p' $(srcdir)/pypips.h.in > pypips.h
	$(AM_V_GEN)$(CPROTO) -evcf2 -E "$(CPP) $(_pypips_la_CPPFLAGS) $(INCLUDES) $(DEFAULT_INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) -DCPROTO_IS_PROTOTYPING" $(srcdir)/pypips.c >> pypips.h
	$(AM_V_GEN)sed -n -e '/\/\*CPROTO pyips\.h/,$$ p' $(srcdir)/pypips.h.in >> pypips.h


pypips_aux.c:pypips.h pypips.c
	$(AM_V_GEN)cat pypips.h $(srcdir)/pypips.c > pypips_aux.c

p2p_cmd= $(srcdir)/pipsmakerc2python.py $(top_srcdir)/src/Documentation/pipsmake/pipsmake-rc.tex ../../Documentation/pipsmake/properties.rc ../../Documentation/pipsmake/pipsdep.rc
pyps.py:$(p2p_cmd) Makefile
	$(AM_V_GEN)echo '# -*- coding: utf-8 -*-' > pyps.py
	$(AM_V_GEN)echo "from pypsbase import *" >> pyps.py
	$(AM_V_GEN)$(PYTHON) $(p2p_cmd) -loop >> pyps.py
	$(AM_V_GEN)$(PYTHON) $(p2p_cmd) -module >> pyps.py
	$(AM_V_GEN)$(PYTHON) $(p2p_cmd) -modules >> pyps.py
	$(AM_V_GEN)$(PYTHON) $(p2p_cmd) -properties >> pyps.py

pips-make: pips-make.in
	$(AM_V_GEN)sed -e "s+@BINPATH@+$(pipsmakeenvdir)+" $(srcdir)/pips-make.in > pips-make

pypsconfig.py: Makefile pypsconfig.py.in
	$(AM_V_GEN)cp $(srcdir)/pypsconfig.py.in pypsconfig.py
	$(AM_V_GEN)echo 'pypsruntime = PypsRuntimeDir("$(pypsruntimedir)")' >> pypsconfig.py

CLEANFILES=pypips_aux.c pypips.py pyps.py pypips_wrap.c pypips.i pypips.h pypsconfig.py
