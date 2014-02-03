# $Id$
#
# Copyright 1989-2014 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#

ifndef has_swig_done

SWIG=swig
has_swig := $(shell type $(SWIG) > /dev/null 2>&1 && echo ok)
ifeq ($(has_swig),ok)

PYTHON_CONFIG=python-config
has_python_dev := $(shell type $(PYTHON_CONFIG) > /dev/null 2>&1 && echo ok)
ifeq ($(has_python_dev),ok)
else
$(warning pyps compilation requested but $(PYTHON_CONFIG) not found in PATH. You can force the value of the PYTHON_CONFIG variable or disable pyps by setting PIPS_NO_PYPS=1)
PIPS_NO_PYPS=1
endif

else
$(warning pyps compilation requested but $(SWIG) not found in PATH. You can force the value of the SWIG makefile variable or disable  pyps by setting PIPS_NO_PYPS=1)
PIPS_NO_PYPS=1
endif

has_swig_done=1

endif

