BUILT_SOURCES = pypips_wrap.c  pypips_aux.c
dist_noinst_DATA=_pyps.py pypips.i.in pypips.c pipsmakerc2python.pl

dist_pkgpython_PYTHON = pyps.py pypips.py pipscc.py pii.py
pkgpyexec_LTLIBRARIES = _pypips.la
_pypips_la_SOURCES = pypips_aux.c 
dist__pypips_la_SOURCES =pypips_wrap.c

_pypips_la_LDFLAGS = -module $(PYTHON_LDFLAGS)
_pypips_la_LIBADD = ../../Libs/libpipslibs.la $(PYTHON_EXTRALIBS) $(LINEARLIBS_LIBS) $(NEWGENLIBS_LIBS)
_pypips_la_CPPFLAGS = \
	$(PYTHON_CPPFLAGS)\
	$(LINEARLIBS_CFLAGS)\
	$(NEWGENLIBS_CFLAGS)\
	-I$(top_srcdir)/src/Documentation/constants \
	-I../../Documentation/pipsmake \
	-I../../Documentation/newgen \
	-I../../Libs/misc\
	-I../../Libs/newgen\
	-I../../Libs/ri-util\
	-I../../Libs/top-level\
	-I../../Libs/pipsmake\
	-I../../Libs/pipsdbm\
	-I../../Libs/properties

pypips_wrap.c : pypips.i pypips.h
	$(SWIG) -python -o $@-tmp pypips.i
	cat pypips.h $@-tmp > $@
	rm -f $@-tmp


pypips.i:pypips.i.in pypips.h
	sed -e 's/SWIG_MODULE_NAME/pypips/' $(srcdir)/pypips.i.in pypips.h > $@

pypips.h:pypips.c
	$(CPROTO) $(srcdir)/pypips.c > $@
	echo "extern char *activate(char *);" >> $@
	echo "extern char *module_loops(const char*,const char*);" >> $@
	echo "#include <unistd.h>" >> $@

pypips_aux.c:pypips.h pypips.c
	cat pypips.h $(srcdir)/pypips.c > $@

p2p_cmd= $(srcdir)/pipsmakerc2python.pl $(top_srcdir)/src/Documentation/pipsmake/pipsmake-rc.tex ../../Documentation/pipsmake/properties.rc ../../Documentation/pipsmake/pipsdep.rc
pyps.py:$(p2p_cmd) _pyps.py 
	sed -n -e '1,/### loop_methods/ p' $(srcdir)/_pyps.py > $@
	$(PERL) $(p2p_cmd) -loop >> $@
	sed -n -e '/### loop_methods/,/### module_methods/ p' $(srcdir)/_pyps.py >> $@
	$(PERL) $(p2p_cmd) -module >> $@
	sed -n -e '/### module_methods/,/### modules_methods/ p' $(srcdir)/_pyps.py >> $@
	$(PERL) $(p2p_cmd) -modules >> $@
	sed -n -e '/### modules_methods/,$$ p' $(srcdir)/_pyps.py >> $@

CLEANFILES=pypips_aux.c pypips.py pyps.py pypips_wrap.c pypips.i pypips.h
