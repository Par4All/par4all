SUBDIRS  = 
BUILT_SOURCES = phpips_wrap.c  phpips_aux.c

dist_noinst_DATA=phpips.i.in phpips.c pipsmakerc2php.php phpips.h.in pips-make.in

dist_bin_SCRIPTS = 

pipsmakeenvdir = $(libdir)/pips-make
dist_pipsmakeenv_SCRIPTS = gcc ar

dist_pkgPHP_PHP = phpipsbase.php
dist_PHP_DATA = pips.pth

nodist_pkgPHP_PHP=phpips.php
pkgphpexec_LTLIBRARIES = _phpips.la
_phpips_la_SOURCES = phpips_aux.c
nodist__phpips_la_SOURCES =phpips_wrap.c

_phpips_la_LDFLAGS = -module $(PHP_LDFLAGS)
_phpips_la_LIBADD = ../../Libs/libpipslibs.la $(PHP_EXTRALIBS) $(LINEARLIBS_LIBS) $(NEWGENLIBS_LIBS)
_phpips_la_CPPFLAGS = \
	-I../../Libs/gnulib \
	`php-config --includes` \
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

phpips_wrap.c : phpips.i phpips.h
	$(AM_V_GEN)$(SWIG) -php -o phpips_wrap.c-tmp phpips.i
	$(AM_V_GEN)cat phpips.h phpips_wrap.c-tmp > phpips_wrap.c
	@rm -f phpips_wrap.c-tmp


phpips.i:phpips.i.in phpips.h
	$(AM_V_GEN)sed -e 's/SWIG_MODULE_NAME/phpips/' $(srcdir)/phpips.i.in phpips.h > phpips.i

phpips.h:phpips.c phpips.h.in Makefile
	$(AM_V_GEN)sed -n -e '1,/\/\*CPROTO phpips\.h/ p' $(srcdir)/phpips.h.in > phpips.h
	$(AM_V_GEN)$(CPROTO) -evcf2 -E "$(CPP) $(_phpips_la_CPPFLAGS) $(INCLUDES) $(DEFAULT_INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) -DCPROTO_IS_PROTOTYPING" $(srcdir)/phpips.c >> phpips.h
	$(AM_V_GEN)sed -n -e '/\/\*CPROTO phpips\.h/,$$ p' $(srcdir)/phpips.h.in >> phpips.h


phpips_aux.c:phpips.h phpips.c
	$(AM_V_GEN)cat phpips.h $(srcdir)/phpips.c > phpips_aux.c

p2p_cmd= $(srcdir)/pipsmakerc2php.php $(top_srcdir)/src/Documentation/pipsmake/pipsmake-rc.tex ../../Documentation/pipsmake/properties.rc ../../Documentation/pipsmake/pipsdep.rc
phpips.php:$(p2p_cmd) Makefile
	$(AM_V_GEN)echo '# -*- coding: utf-8 -*-' > phpips.php
	$(AM_V_GEN)echo "include('phpipsbase');" >> phpips.php
	$(AM_V_GEN)$(PHP) $(p2p_cmd) -loop >> phpips.php
	$(AM_V_GEN)$(PHP) $(p2p_cmd) -module >> phpips.php
	$(AM_V_GEN)$(PHP) $(p2p_cmd) -modules >> phpips.php
	$(AM_V_GEN)$(PHP) $(p2p_cmd) -properties >> phpips.php

pips-make: pips-make.in
	$(AM_V_GEN)sed -e "s+@BINPATH@+$(pipsmakeenvdir)+" $(srcdir)/pips-make.in > pips-make

phpipsconfig.php:Makefile
	$(AM_V_GEN)echo 'phpipsruntime="$(phpipsruntimedir)"' > phpipsconfig.php

CLEANFILES=phpips_aux.c phpips.php phpips.php phpips_wrap.c phpips.i phpips.h phpipsconfig.php
