# $Id$

use_pdflatex	= 1

ROOT	= ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk

# local stuff

# Figures are missing
A-471.pdf: A-471.tex \
	reportCh1/constraints.tex \
	reportCh1/goals.tex \
	reportCh1/paws.tex \
	reportCh1/process.tex \
	reportCh1/reportCh1.tex \
	reportCh1/requirements.tex \
	reportCh2/architecture.tex \
	reportCh2/optimal_solution.tex \
	reportCh2/problems.tex \
	reportCh2/reportCh2.tex \
	reportCh2/solutions.tex \
	reportCh2/technologies.tex \
	reportCh3/current_configuration.tex \
	reportCh3/extension.tex \
	reportCh3/installation.tex \
	reportCh3/launching.tex \
	reportCh3/reportCh3.tex \
	reportCh4/basic_tools.tex \
	reportCh4/full_control.tex \
	reportCh4/paws.tex \
	reportCh4/reportCh4.tex \
	reportCh4/tutorial.tex \
	reportDef.tex \
	reportEnd.tex \
	reportIntro.tex \
	reportLit.tex

clean: local-clean

local-clean:
	$(RM) -rf $(INSTALL_DOC) $(INSTALL_HTM) \
		*.aux *.bbl *.blg *.brf *.rel *.toc *.idx *.ilg *.ind *.log *.dvi \
		*.ps *.out *.lof *.pdf
