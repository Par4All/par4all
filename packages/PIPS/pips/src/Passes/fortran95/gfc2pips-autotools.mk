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



# PIPS Libraries
PIPSLIBS_LIBS = ../../../../../Libs/
LIBS_PIPS = \
	$(PIPSLIBS_LIBS)/ri-util/.libs/libri-util.a \
	$(PIPSLIBS_LIBS)/text-util/.libs/libtext-util.a \
	$(PIPSLIBS_LIBS)/syntax/.libs/libsyntax.a \
	$(PIPSLIBS_LIBS)/alias-classes/.libs/libalias-classes.a \
	$(PIPSLIBS_LIBS)/effects-util/.libs/libeffects-util.a \
	$(PIPSLIBS_LIBS)/ri-util/.libs/libri-util.a \
	$(PIPSLIBS_LIBS)/misc/.libs/libmisc.a \
	$(PIPSLIBS_LIBS)/newgen/.libs/libnewgen.a

PIPS_INC_PRE   = -I../../../../../Documentation/newgen/ 
PIPS_INC_POST  = -I../../../../../Libs/preprocessor/
PIPS_INC_POST += -I../../../../../Libs/ri-util/
PIPS_INC_POST += -I../../../../../Libs/syntax/
PIPS_INC_POST += -I../../../../../Libs/misc/
PIPS_INC_POST += -I../../../../../Libs/newgen/
PIPS_INC_POST += -I../../../../../Libs/text-util/
PIPS_INC_POST += -I$(pipssrcdir)/../../Documentation/newgen/
PIPS_INC_POST += -I$(pipssrcdir)/../../Documentation/constants/
PIPS_INC_POST += $(LINEARLIBS_CFLAGS) $(NEWGENLIBS_CFLAGS) 


