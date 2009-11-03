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
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

# swig stuff

TMP_PREFIX=the_
PYPS_STDOUT=pypsout

$(PYTHON_MODULE).tmp: $(ROOT)/src/Passes/tpips/tp_yacc.y
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
	rm -f $(PYTHON_MODULE).py _$(PYTHON_MODULE).so pyps.py pipscc.py

pipscc.py:pipscc
	ln -s $< $@

check:sample.py all pipscc.py pii.py
	python -c "import pypips"
	python -c "import pyps"
	python -c "import pipscc"
	python -c "import pii"
	PAGER=cat ./$<
	./pipscc -c test.c
	PAGER=cat ./pipscc test.o -o thetest
	rm -f thetest test.o
	python pii.py -c test.c
	PAGER=cat python pii.py test.o -o thetest


checkall:xpc.py
	python ./xpc.py --module=foo --sources=carto.c,get.c --algo=genetic --nbgen=2 --CPPFLAGS=-DSIZE=100

DOCSTRING=%%@UserManualDocumentation:
pyps.py: pipsmakerc2python.pl ../../Documentation/pipsmake/pipsmake-rc.tex $(ETC.d)/properties.rc $(ETC.d)/pipsdep.rc _pyps.py 
	sed -n -e '1,/### loop_methods/ p' _pyps.py > $@
	./$^ -loop >> $@
	sed -n -e '/### loop_methods/,/### module_methods/ p' _pyps.py >> $@
	./$^ -module >> $@
	sed -n -e '/### module_methods/,/### modules_methods/ p' _pyps.py >> $@
	./$^ -modules >> $@
	sed -n -e '/### modules_methods/,$$ p' _pyps.py >> $@
