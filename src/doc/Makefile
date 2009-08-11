#
# $Id$
#
# Copyright 1989-2009 MINES ParisTech
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

TEX = 	newgen_manual.tex \
	newgen_paper.tex \
	tutoriel_newgen.tex

SOURCES = $(TEX) obtention.txt

PDF 	= $(TEX:%.tex=%.pdf)

INSTALL_DOC =	$(PDF)

ROOT	= ../..
include $(ROOT)/makes/main.mk

all: $(INSTALL_DOC)

pdf: $(PDF)
ps: $(PS)
dvi: $(DVI)

clean: local-clean

local-clean:
	$(RM) -r $(INSTALL_DOC) $(INSTALL_HTM) $(PDF) \
		*.html *.aux *.log *.ps *.dvi *.out *~
