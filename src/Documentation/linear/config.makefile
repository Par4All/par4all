#
# $Id$
#

SOURCES =	linear.tex

all: linear.ps

linear.dvi: linear.tex;	frlatex $< ; frlatex $<;
linear.ps: linear.dvi; dvips $< -o

clean: lclean
lclean:; $(RM) linear.dvi linear.ps
