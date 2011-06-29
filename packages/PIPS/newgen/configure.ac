#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
#
# Copyright 1989-2010 MINES ParisTech
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

AC_PREREQ(2.61)
AC_INIT([newgen], [0.1], [pipsdev@cri.ensmp.fr])
AC_CONFIG_SRCDIR([src/genC/genC.c])
AC_CONFIG_HEADER([config.h])
AC_CONFIG_MACRO_DIR([makes/m4])

AM_INIT_AUTOMAKE([gnu 1.0])

# Checks for programs.
AX_PROG_YACC
AC_PROG_CC_C99
gl_EARLY
AC_PROG_CPP
AX_PROG_LEX
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_LIBTOOL
AX_CHECK_CPROTO([4.7])

# doc
AX_ARG_ENABLE([doc],
			  [enable documentation build],
			  [no],
			  [
			   AX_CHECK_PROG([pdflatex])
			   AX_CHECK_PROG([kpsepath])
			   AX_CHECK_PROG([epstopdf])
			   AX_LATEX_PACKAGE([svninfo])
			   AX_LATEX_PACKAGE([hyperref])
			   AX_DEPENDS([doc],[pdflatex kpsepath epstopdf svninfo hyperref])
			  ]
)
AM_CONDITIONAL([WITH_DOC],[AX_HAS([doc])])

# Checks for libraries.

# Checks for header files.
AC_HEADER_STDC
AX_CHECK_HEADERS([inttypes.h memory.h stdint.h stdlib.h string.h strings.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_TYPE_SIZE_T
AX_LD_NO_UNDEFINED

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_MEMCMP
AC_FUNC_REALLOC
AC_FUNC_VPRINTF

# this takes care of the gnulib stuff
gl_INIT
CPPFLAGS="${CPPFLAGS} -I\$(top_builddir)/src/gnulib"

# devel mode, sets various compiler flags and so on
# this must be the last check because it modifies CFLAGS
AX_ARG_ENABLE([devel-mode],[enable developer mode],[no],
			  [
			   	# set CFLAGS for more warnings
			   	CFLAGS="-g -O0 -Wall -Werror"
			  ]
)

AC_CONFIG_FILES([Makefile
				 newgenlibs.pc
                 src/Makefile
				 src/gnulib/Makefile
                 src/genC/Makefile
                 src/genC/genread/Makefile
                 src/genC/genspec/Makefile
                 src/scripts/Makefile
				 src/doc/Makefile
                 ])

AX_DEPENDS([minimum],[lex yacc cproto std_headers])
AX_OUTPUT([minimum],[doc])
