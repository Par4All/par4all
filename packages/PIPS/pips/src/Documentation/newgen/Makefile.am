#
# Copyright 1989-2010 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#

include $(srcdir)/newgen.mk

dist_noinst_DATA=$(F_tex) unstructured.idraw newgen_domain.sty
if WITH_DOC
doc_DATA=$(F_pdf)
endif

dist_noinst_SCRIPTS=remove-latex-comments
noinst_LTLIBRARIES=libnewgenaux.la
libnewgenaux_la_SOURCES=$(F_c)
BUILT_SOURCES=specs.h $(F_c)
include $(top_srcdir)/makes/latex.mk
SUFFIXES+=.newgen 

AM_CPPFLAGS=$(NEWGENLIBS_CFLAGS)

include_HEADERS=all_newgen_headers.h specs.h $(F_h)

all_newgen_headers.h: specs.h Makefile
	$(AM_V_GEN) sed -n 's,^\(.*\)_spec.*,#include "\1.h",p' specs.h > $@

specs.h:Makefile $(F_newgen)
	$(AM_V_GEN) PATH=`dirname $(NEWGEN)`:$$PATH $(NEWGEN) -c $(F_newgen) | $(MAKE_ALL_SPECS) > specs.h

.tex.newgen:
	$(AM_V_GEN)$(srcdir)/remove-latex-comments $<

$(F_c):specs.h

# Convert idraw/eps files.
ri.pdf:unstructured.pdf


CLEANFILES=$(F_c) $(include_HEADERS) *.newgen $(F_h) *.spec $(F_pdf) unstructured.pdf

