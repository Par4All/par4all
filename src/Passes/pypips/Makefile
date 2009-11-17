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

# common stuff
ROOT    = ../../..
include $(ROOT)/makes/has_swig.mk

PYTHON_MODULE=pypips
PYTHON_INCLUDE=`$(PYTHON_CONFIG) --cflags`
PYPS_STDOUT=pypsout

DYNLIB_TARGET=_$(PYTHON_MODULE).so
PY_TARGET=pyps.py $(PYTHON_MODULE).py 

LDFLAGS+=-L$(LIB.d) -L$(LIB.d)/$(ARCH) $(addprefix -l, $(pips.libs)) `$(PYTHON_CONFIG) --ldflags`

DYNLIB_OBJECTS=$(PYTHON_MODULE)_wrap.o $(PYTHON_MODULE).o


PROJECT = pips
include $(ROOT)/makes/main.mk

# install hook
install:pyps-install
pyps-install:
	$(INSTALL) -d $(PY.d)
	MAJOR_VERSION="`echo '$(VERSION)' | cut -d '.' -f 1`";\
	  $(CMP) $(ARCH)/$(DYNLIB_TARGET) $(PY.d)/$(DYNLIB_TARGET).$$MAJOR_VERSION || \
	    $(INSTALL) -m 644 $(ARCH)/$(DYNLIB_TARGET) $(PY.d)/$(DYNLIB_TARGET).$$MAJOR_VERSION ; \
		$(RM) $(PY.d)/$(DYNLIB_TARGET) ;\
		cd $(PY.d) && ln -s $(DYNLIB_TARGET).$$MAJOR_VERSION $(DYNLIB_TARGET) && cd -

# swig stuff

$(PYTHON_MODULE).h:$(PYTHON_MODULE).c
	$(PROTOIZE) $< > $@
	echo "extern char *activate(char *phase);" >> $@

$(PYTHON_MODULE).I:$(PYTHON_MODULE).i $(PYTHON_MODULE).h
	sed -e 's/SWIG_MODULE_NAME/$(PYTHON_MODULE)/' $^ > $@

$(PYTHON_MODULE)_wrap.c:$(PYTHON_MODULE).I
	swig -python $<


$(PYTHON_MODULE).py:$(PYTHON_MODULE).I
	swig -python $<

CPPFLAGS+=-I$(PYTHON_INCLUDE)

%.o:%.c
	$(COMPILE) $< -o $@

clean:
	rm -f *.o *.tmp *.I *.H $(PYTHON_MODULE)_wrap.c *.pyc $(PYTHON_MODULE).h

distclean:clean
	rm -f $(PYTHON_MODULE).py _$(PYTHON_MODULE).so pyps.py pipscc.py

pipscc.py:pipscc
	ln -s $< $@


check:sample.py all pipscc.py pii.py
	PYTHONPATH=$(PY.d) python -c "import pypips"
	PYTHONPATH=$(PY.d) python -c "import pyps"
	PYTHONPATH=$(PY.d) python -c "import pipscc"
	PYTHONPATH=$(PY.d) python -c "import pii"
	PYTHONPATH=$(PY.d) PAGER=cat ./$<
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
