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

#   	COMPLEXITY EVALUATION
#   	---------------------
#
# Pierre Berthomier  07-09-90
# Lei Zhou	     22-02-91
#
####### The source files directly involved in complexity are:
#	
# in $INCLUDEDIR:
#	complexity_ri.f.tex	describing `complexity' data structures
#	complexity_ri.newgen	  |
#	complexity_ri.spec	  | three files generated with NewGen
#	complexity_ri.h		  |
#
# in ~pips/Pips/Development/Lib/complexity:
#      (complexity-local.h	my local header)
#	complexity.h		automatically generated header
#				  the local one + subroutines decl.
#	comp_scan.c		subroutines that scan the RI to
#                                 count operations
#	expr_to_pnome.c		subroutines that walk the RI expressions
#				  to try to give them a polynomial form
#	comp_unstr.c		subroutines that cope with unstructured
#				  graphs of statements
#	comp_util.c		useful subroutines for evaluation
#				  of complexity
#	comp_math.c		"mathematical" operations on complexities:
#				  addition, integration, ...
#	comp_matrice.c	        matrice inversion for floating point
#
#	comp_prettyprint.c	routines for prettyprinting complexities
#				  with Fortran source code
#	polynome_ri.c		interface polynomial library / RI
#
#	main.c		        main(), to test complexity routines.
#
#
####### The usable files created are:
#
#	complexity		contains all routines and the main
#				  to become a pass of PIPS
#	libcomplexity.a		contains all but main.c
#
#######
#

TARGET	= complexity

INSTALL_ETC =	complexity_cost_tables

LIB_CFILES = \
	comp_scan.c comp_expr_to_pnome.c comp_unstr.c \
	comp_util.c comp_math.c comp_prettyprint.c polynome_ri.c \
	comp_matrice.c

INC_TARGET =    $(TARGET).h

LIB_TARGET      = lib$(TARGET).a

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk
