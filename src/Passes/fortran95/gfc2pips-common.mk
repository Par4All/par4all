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




GFC2PIPS_OBJS = fortran/gfc2pips.o fortran/gfc2pips-util.o fortran/gfc2pips-stubs.o fortran/gfc2pips-comments.o

# The compiler itself is called f951.
f951$(exeext): $(F95_OBJS) $(LIBS_PIPS) $(GFC2PIPS_OBJS) \
		$(BACKEND) $(LIBDEPS) attribs.o
	$(CC) $(ALL_CFLAGS) $(LDFLAGS) -o $@ \
		$(F95_OBJS) $(BACKEND) $(LIBS) $(GFC2PIPS_OBJS) $(LIBS_PIPS) $(LINEARLIBS_LIBS) $(NEWGENLIBS_LIBS)  attribs.o $(BACKENDLIBS) -lgmp -lmpfr



fortran/gfc2pips-comments.o: fortran/gfc2pips-comments.c fortran/gfc2pips.h fortran/gfc2pips-private.h
	$(CC) $(PIPS_INC_PRE) -std=c99 -g -c $(ALL_CPPFLAGS) $(PIPS_INC_POST) -DBASEVER=$(BASEVER_s)  \
		$< $(OUTPUT_OPTION)

fortran/gfc2pips-stubs.o: fortran/gfc2pips-stubs.c fortran/gfc2pips.h fortran/gfc2pips-private.h
	$(CC) $(PIPS_INC_PRE) -std=c99 -g -c $(ALL_CPPFLAGS) $(PIPS_INC_POST) -DBASEVER=$(BASEVER_s)  \
		$< $(OUTPUT_OPTION)

fortran/gfc2pips-util.o: fortran/gfc2pips-util.c fortran/gfc2pips.h fortran/gfc2pips-private.h
	$(CC) $(PIPS_INC_PRE) -std=c99 -g -c $(ALL_CPPFLAGS) $(PIPS_INC_POST) -DBASEVER=$(BASEVER_s)  \
		$< $(OUTPUT_OPTION)

fortran/gfc2pips.o: fortran/gfc2pips.c fortran/gfc2pips.h fortran/gfc2pips-private.h
	$(CC) $(PIPS_INC_PRE) -std=c99 -g -c $(ALL_CPPFLAGS) $(PIPS_INC_POST) -DBASEVER=$(BASEVER_s)  \
		$< $(OUTPUT_OPTION)

