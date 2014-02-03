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

# This makefile forwards make targets to sub-directories...
# It must be used with GNU make (or some make that supports -C)
#
# If you do 'gmake FOO' in a directory with this Makefile
# then 'gmake FOO' is done downwards in the sub-directories.
# Gmake automatically forward make macros of the command line,
# thus 'gmake CC=mycc all' will also forward the CC macro.
#
# 5 macros are of special interet for the make target forwarding process:
#
# FWD_DIRS:
#   the list of sub directories for forwarding commands.
#   if speficied, the target are only forwarded to these directories.
#   the default value is all sub-directories.
# FWD_MKFLAGS:
#   additional flags for make
# FWD_MSG:
#   message to print when reporting failure or success of a sub-make.
#   the default is no message.
# FWD_OUT:
#   where to redirect the output. default is nothing (stderr/stdout).
#   one can suggest "2>&1 > /dev/null" for keeking stderr on stdout.
# FWD_REPORT:
#   where to report failures or successes.
#   may be a pipe, a redirection as "> /dev/null", "" or "> file"...
#   the default is to report to stderr.
# FWD_ROOT:
#   relative directory from the start of the forward process.
#   the default value is "." (current directory).
# FWD_STOP_ON_ERROR:
#   if set, stop on the first failure met.
# FWD_PARALLEL:
#   if set, run in parallel
#
# These variables may be overwritten on the command line as
# gmake FWD_DIRS="foo" FWD_MSG="argh!" FWD_REPORT="> /dev/null" FWD_ROOT=$PWD
# Their value is printed with special target "debug_forward_makefile"
# (which is also forwarded:-)
# the default target is all.
#

# doxygen is a target to do, even if there is such a directory/file with
# this name:
.PHONY: doxygen

# Just to avoid troubles...
SHELL		= /bin/sh

# default values
FWD_DIRS	= *
FWD_MSG		=
FWD_REPORT	= >&2
FWD_ROOT	= .
FWD_OUT		=
FWD_STOP_ON_ERROR = 1
FWD_MKFLAGS	=
FWD_PARALLEL	=

la_cible_par_defaut_si_aucune_n_est_precisee_sur_la_ligne_de_commande: all

# get local stuff if any.
-include *.mk

# Forward any command to the specified directories (if any).
ifdef FWD_PARALLEL
.DEFAULT:
	$(MAKE) $(FWD_DIRS:%=fwd-%) FWD_TARGET=$@

fwd-%: %
	$(MAKE) -C $< $(FWD_MKFLAGS) \
		FWD_ROOT="$(FWD_ROOT)/$$d" \
		FWD_STOP_ON_ERROR="$(FWD_STOP_ON_ERROR)" $(FWD_TARGET)
else
# Report the result of the forward.
.DEFAULT doxygen:
	@echo Making $@ in $(FWD_ROOT) >&2;\
	globally_failed=;\
	if test "$@" = debug_forward_makefile ; \
	then \
	  echo "FWD_DIRS=$(FWD_DIRS)"; \
	  echo "FWD_MSG=$(FWD_MSG)"; \
	  echo "FWD_ROOT=$(FWD_ROOT)"; \
	  echo "FWD_REPORT=$(FWD_REPORT)"; \
	  echo "FWD_OUT=$(FWD_OUT)"; \
	  echo "FWD_STOP_ON_ERROR=$(FWD_STOP_ON_ERROR)"; \
          echo "FWD_PARALLEL=$(FWD_PARALLEL)" ; \
	fi;\
	for d in $(FWD_DIRS) ; do \
	  if test -d $$d && test -f $$d/Makefile ; \
	  then \
	    echo "Forwarding $@ to $(FWD_ROOT)/$$d" >&2 ;\
	    if $(MAKE) -C $$d $(FWD_MKFLAGS) \
		FWD_ROOT="$(FWD_ROOT)/$$d" \
		FWD_STOP_ON_ERROR="$(FWD_STOP_ON_ERROR)" $@ ;\
	    then report=succeeded ;\
	    else report=failed ;\
		 globally_failed=1 ;\
            fi ;\
	    echo "$(FWD_MSG) $(FWD_ROOT)/$$d: $@ $$report" $(FWD_REPORT);\
	    [ "$(FWD_STOP_ON_ERROR)" -a "$$globally_failed" ] && break;\
	  else \
	    echo "Ignoring directory $$d" >&2 ;\
	  fi ; \
	 done $(FWD_OUT) ; \
	 [ "$$globally_failed" ] && \
		echo "$(FWD_MSG) $(FWD_ROOT):" \
			"some making in `pwd` failed" $(FWD_REPORT);\
	 [ "$(FWD_STOP_ON_ERROR)" ] && exit $$globally_failed;\
	 exit 0
endif
