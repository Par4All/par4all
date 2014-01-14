# ^
# $Id$
#
# Copyright 1989-2014 MINES ParisTech
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

# common stuff
ROOT    = ../../..
include $(ROOT)/makes/has_swig.mk

PYTHON_MODULE=pypips
PYTHON_INCLUDE=`$(PYTHON_CONFIG) --cflags`
PYPS_STDOUT=pypsout

DYNLIB_TARGET=_$(PYTHON_MODULE).so
PY_TARGET=pyps.py $(PYTHON_MODULE).py pipscc.py pii.py

BIN_TARGET=pipscc

INSTALL_EXE = ipyps

DYNLIB_OBJECTS=the_$(PYTHON_MODULE)_wrap.o $(PYTHON_MODULE).o

PROJECT = pips
include $(ROOT)/makes/main.mk

LDFLAGS+=-L$(LIB.d) -L$(LIB.d)/$(ARCH) $(addprefix -l, $(pips.libs)) `$(PYTHON_CONFIG) --ldflags`

# install hook
install: pyps-install
pyps-install: setup.py
	$(INSTALL) -d $(PY.d)
	  $(CMP) $(ARCH)/$(DYNLIB_TARGET) $(PY.d)/$(DYNLIB_TARGET) || \
	    $(INSTALL) -m 644 $(ARCH)/$(DYNLIB_TARGET) $(PY.d)/$(DYNLIB_TARGET) 

# swig stuff

$(PYTHON_MODULE).h: $(PYTHON_MODULE).c
	$(PROTOIZE) $< > $@
	echo "extern char *activate(char *phase);" >> $@
	echo "extern int delete_workspace(char *phase);" >> $@

$(PYTHON_MODULE).i:$(PYTHON_MODULE).i.in $(PYTHON_MODULE).h
	sed -e 's/SWIG_MODULE_NAME/$(PYTHON_MODULE)/' $^ > $@

$(PYTHON_MODULE)_wrap.c:$(PYTHON_MODULE).i
	swig -python $<

the_$(PYTHON_MODULE)_wrap.c:$(PYTHON_MODULE).h $(PYTHON_MODULE)_wrap.c
	cat $^ > $@

$(PYTHON_MODULE).py:$(PYTHON_MODULE).i
	swig -python $<

CPPFLAGS+=-I$(PYTHON_INCLUDE)

%.o:%.c
	$(COMPILE) $< -o $@

clean:
	rm -f *.o *.tmp *.H the_$(PYTHON_MODULE)_wrap.c $(PYTHON_MODULE)_wrap.c \
	  *.pyc $(PYTHON_MODULE).h pypips.i
	rm -rf build 

distclean: clean
	rm -f $(PYTHON_MODULE).py _$(PYTHON_MODULE).so pyps.py *.i

$(ARCH)/pipscc:pipscc.py
	cp $< $@
	chmod u+x $@

check: sample.py all pipscc.py pii.py
	PYTHONPATH=$(PY.d) python -c "import pypips"
	PYTHONPATH=$(PY.d) python -c "import pyps"
	PYTHONPATH=$(PY.d) python -c "import pipscc"
	PYTHONPATH=$(PY.d) python -c "import pii"
	PYTHONPATH=$(PY.d) PAGER=cat ./$<
	PYTHONPATH=$(PY.d) PAGER=cat ./test.py
	PYTHONPATH=$(PY.d) PAGER=cat ./test2.py
	PYTHONPATH=$(PY.d) ./pipscc -c test.c
	PYTHONPATH=$(PY.d) PAGER=cat ./pipscc test.o -o thetest
	rm -f thetest test.o
	PYTHONPATH=$(PY.d) python pii.py -c test.c
	PYTHONPATH=$(PY.d) PAGER=cat python pii.py test.o -o thetest


checkall:xpc.py
	PYTHONPATH=$(PY.d) python ./xpc.py --module=foo --sources=carto.c,get.c --algo=genetic --nbgen=2 --CPPFLAGS=-DSIZE=100


pyps.py: pipsmakerc2python.pl ../../Documentation/pipsmake/pipsmake-rc.tex $(ETC.d)/properties.rc $(ETC.d)/pipsdep.rc _pyps.py 
	sed -n -e '1,/### loop_methods/ p' _pyps.py > $@
	./$^ -loop >> $@
	sed -n -e '/### loop_methods/,/### module_methods/ p' _pyps.py >> $@
	./$^ -module >> $@
	sed -n -e '/### module_methods/,/### modules_methods/ p' _pyps.py >> $@
	./$^ -modules >> $@
	sed -n -e '/### modules_methods/,$$ p' _pyps.py >> $@

#no dependency on setup.py.in to regenerate the file every time it is needed
setup.py:
	$(MAKE) $(PYTHON_MODULE).i
	sed -e 's,@CPPFLAGS@,$(CPPFLAGS),' -e 's,@LDFLAGS@,$(LDFLAGS),' $@.in > $@
