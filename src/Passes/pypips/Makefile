# $Id$
#
# Copyright 1989-2009 MINES ParisTech
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

PYTHON_MODULE=pypips
PYTHON_INCLUDE=/usr/include/python2.5
all:_$(PYTHON_MODULE).so pyps.py


# common stuff
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

PROJECT = pips
include $(ROOT)/makes/main.mk

# swig stuff

TMP_PREFIX=the_
PYPS_STDOUT=pypsout

$(PYTHON_MODULE).tmp:$(PIPS_ROOT)/src/Passes/tpips/tp_yacc.y
	sed -n -e '/%{/,/%}/ p' $< | sed -e '1 d' -e '$$ d' > $@

$(TMP_PREFIX)$(PYTHON_MODULE).c:$(PYTHON_MODULE).tmp $(PYTHON_MODULE).c
	cat $^ >> $@


$(PYTHON_MODULE).h:$(TMP_PREFIX)$(PYTHON_MODULE).c
	$(PROTOIZE) $< > $@
	echo "extern char *activate(char *phase);" >> $@

$(PYTHON_MODULE).I:$(PYTHON_MODULE).i $(PYTHON_MODULE).h
	sed -e 's/SWIG_MODULE_NAME/$(PYTHON_MODULE)/' $^ > $@

$(PYTHON_MODULE)_wrap.c:$(PYTHON_MODULE).I
	swig -python $<

_$(PYTHON_MODULE).so:$(PYTHON_MODULE)_wrap.o $(TMP_PREFIX)$(PYTHON_MODULE).o
	$(LD) -shared $^ $(LDFLAGS) $(LDOPT) $(addprefix -l,tpips readline termcap $(pips.libs)) -o $@

CPPFLAGS+=-I$(PYTHON_INCLUDE)

%.o:%.c
	$(COMPILE) $< -o $@

clean:
	rm -f *.o *.tmp $(TMP_PREFIX)* *.I *.H $(PYTHON_MODULE)_wrap.c *.pyc $(PYTHON_MODULE).h

distclean:clean
	rm -f $(PYTHON_MODULE).py _$(PYTHON_MODULE).so pyps.py

check:sample.py all
	python -c "import pypips"
	python -c "import pyps"
	PAGER=cat ./$<

checkall:xpc.py
	python ./xpc.py --module=foo --sources=carto.c,get.c --algo=genetic --nbgen=2 --CPPFLAGS=-DSIZE=100

DOCSTRING=%%@UserManualDocumentation:
pyps.py: ../../Documentation/pipsmake/pipsmake-rc.tex _pyps.py pipsmakerc2python.pl
	sed -n -e '1,/### helpers/ p' _pyps.py > $@
	./pipsmakerc2python.pl $< >> $@
	sed -n -e '/### helpers/,$$ p' _pyps.py >> $@
