#
# $Id$
#
# Copyright 1989-2014 MINES ParisTech
#
# This file is part of NewGen.
#
# NewGen is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or any later version.
#
# NewGen is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License along with
# NewGen.  If not, see <http://www.gnu.org/licenses/>.
#
# Newgen documention

use_pdflatex	= 1

F.tex =	newgen_manual.tex \
	newgen_paper.tex \
	newgen_recursion_paper.tex \
	tutoriel_newgen.tex

SOURCES = $(F.tex) obtention.txt

F.pdf 	= $(F.tex:%.tex=%.pdf)

INSTALL_DOC = $(F.pdf)

ngF.tex	= $(wildcard newgen_*.tex)
INSTALL_HTM = $(ngF.tex:%.tex=%.htdoc) index.html

ROOT	= ../..
include $(ROOT)/makes/main.mk

all: $(INSTALL_DOC)

pdf: $(F.pdf)

clean: local-clean dox-clean

dox-clean:
	$(MAKE) -C doxygen clean

local-clean:
	$(RM) -r $(INSTALL_DOC) $(PDF) *.htdoc \
		*.aux *.log *.ps *.dvi *.out *~
