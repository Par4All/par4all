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

use_pdflatex = 1
INSTALL_DOC = pyps.pdf
#INSTALL_HTM = pyps.htdoc
#DOC.subd = pyps.htdoc

# common stuff
ROOT    = ../../..
PROJECT	= pips
include $(ROOT)/makes/main.mk

# local stuff
clean: local-clean

local-clean:
	$(RM) -rf $(INSTALL_DOC) $(INSTALL_HTM) \
		*.aux *.bbl *.blg *.brf *.rel *.toc *.idx *.ilg *.ind *.log *.dvi *.ps *.out
