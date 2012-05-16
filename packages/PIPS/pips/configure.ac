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
AC_INIT([pips],[0.1],[pipsdev@cri.ensmp.fr])
AC_CONFIG_SRCDIR([src/Passes/pips/pips.c])
AC_CONFIG_HEADER([config.h])
AC_CONFIG_MACRO_DIR([makes/m4])
AM_INIT_AUTOMAKE([foreign 1.0]) # foreign means we drop bsd support
AM_SILENT_RULES([yes])

AM_PROG_CC_C_O

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

# Checks for libraries.
AX_CHECK_PKG([polylib64],[Polyhedron_Enumerate],[polylib.h])

AX_CHECK_PROG([perl])
AX_CHECK_CPROTO([4.7])
AX_CHECK_PROG([newgen])
AX_CHECK_PROG([make_all_specs])

# Checks for required libraries.
AX_CHECK_PKG([newgenlibs], [gen_free],[newgen_array.h])
AX_CHECK_PKG([linearlibs], [norm_syst],[sc.h])




# Checks for header files.
AC_FUNC_ALLOCA
AC_HEADER_DIRENT
AC_HEADER_STDC
AC_HEADER_SYS_WAIT
AX_CHECK_HEADERS([fcntl.h limits.h stdint.h stdlib.h string.h strings.h sys/ioctl.h sys/param.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_C_INLINE
AC_TYPE_INT32_T
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_HEADER_TIME
AC_STRUCT_TM
AC_TYPE_UINT32_T
AC_C_VOLATILE
AX_LD_NO_UNDEFINED

# Checks for library functions.
AC_FUNC_CLOSEDIR_VOID
AC_FUNC_ERROR_AT_LINE
AC_FUNC_FORK
AC_PROG_GCC_TRADITIONAL
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_TYPE_SIGNAL
AC_FUNC_STAT
AC_FUNC_VPRINTF
AX_CHECK_FUNCS([dup2 getcwd gethostname gettimeofday memset mkdir putenv regcomp setenv strcasecmp strchr strcspn strerror strrchr strspn strstr strtol strtoul strtoull])

# doc
AX_ARG_ENABLE([doc],
			  [enable documentation build],
			  [no],
			  [
			   AX_CHECK_PROG([pdflatex])
			   AX_CHECK_PROG([kpsepath])
			   AX_CHECK_PROG([epstopdf])
			   AX_LATEX_PACKAGE([listings])
			   AX_LATEX_PACKAGE([tikz])
			   AX_LATEX_PACKAGE([hyperref])
			   AX_LATEX_PACKAGE([breakurl])
			   AX_LATEX_PACKAGE([epsfig])
			   AX_DEPENDS([doc],[pdflatex kpsepath epstopdf listings tikz breakurl hyperref epsfig])
			  ]
)
AM_CONDITIONAL([WITH_DOC],[AX_HAS([doc])])

# gpips
AX_ARG_ENABLE([gpips],
			[enable gtk interface],
			[no],
			[
 			 AX_CHECK_PKG([gtk+-2.0], [gtk_drag_update],[gtkbox.h])
 			 AX_DEPENDS([gpips],[gtk+-2.0])
			]
)
# tpips
AX_ARG_ENABLE([tpips],
			[enable shell interface],
			[yes],
			[
			 AX_CHECK_PKG([readline], [readline], [readline/readline.h])
			 # These tests discover differences between readline 4.1 and 4.3, with modified cppflags
			 ax_cppflags="$CPPFLAGS"
			 CPPFLAGS="$CPPFLAGS $READLINE_CPPFLAGS"
			 AC_CHECK_DECLS([rl_completion_matches], , ,[#include <stdio.h>
							 #include <readline/readline.h>])
			 AC_CHECK_DECLS([rl_filename_completion_function], , ,[#include <stdio.h>
							 #include<readline/readline.h>])
			 AC_CHECK_TYPES([rl_compentry_func_t], , , [#include <stdio.h>
							 #include <readline/readline.h>])
			 #reset CPPFLAGS
			 CPPFLAGS="$ax_cppflags"
			 AX_CHECK_PKG([ncurses],[tgetent],[term.h])
			 AX_DEPENDS([tpips],[readline ncurses])
			]
)

# phpips
AX_ARG_ENABLE([phpips],
			[enable PHP interface],
			[no],
			[
			 AC_PROG_SWIG
			 AX_CHECK_PROG([php])
			]
)

# pyps
AX_ARG_ENABLE([pyps],
			[enable python interface],
			[no],
			[
			 AC_PROG_SWIG
			 AC_PYTHON_DEVEL([2.5])
			 AX_CHECK_PROG([ipython])
			 AX_DEPENDS([pyps],[swig python_devel ipython])
			 AX_DEPENDS([tpips2pyps],[python_devel])
			 # for ax_extra_scripts
			 AS_IF([AX_HAS([tpips2pyps])],[ax_enable_tpips2pyps=tpips2pyps])
			]
)
AC_SUBST(pypsruntimedir,[$\{datarootdir\}/runtime])
AX_ARG_ENABLE([pyps-extra],
			[enable extra python tools],
			[no],
			[
			 AX_PYTHON_MODULE([Pyro]) 
			 AX_DEPENDS([pyps-extra],[pyps Pyro])
			]
)

AX_ARG_ENABLE([paws],
			[enable pips as web service],
			[no],
			[
			 AX_PYTHON_MODULE([Pyro]) 
			 AX_PYTHON_MODULE([pygments])		 
			 AX_PYTHON_MODULE([pylons], [if pylons.__version__ < '1.0': raise ImportError('The minimum required version of pylons is 1.0.')])		 
			 AX_PYTHON_MODULE([PIL])		 
			 AX_DEPENDS([paws],[pyps Pyro pygments pylons PIL])
			]
)
# hpfc
AX_ARG_ENABLE([hpfc],
			  [enable high performance fortran compiler],
			  [no],
			  [
			   AX_CHECK_PKG([readline],[readline],[readline/readline.h])
			   AX_CHECK_PKG([ncurses],[tgetent],[term.h])
			   AX_DEPENDS([hpfc],[readline ncurses])
			  ]
)

#gfc2pips
AX_ARG_ENABLE([fortran95],
                          [enable Fortran95 support via GCC parser (GFC)],
                          [no],
                          [
                           AX_CHECK_PROG([wget])
                           AX_CHECK_PROG([rsync])
                           AX_CHECK_PROG([md5sum],[MD5SUM=:])
                           AX_DEPENDS([fortran95],[wget rsync])
                          ]
)

# set the optionnal passes dirs there
AC_SUBST([PIPS_EXTRA_PASSES],["$ax_enable_gpips $ax_enable_tpips $ax_enable_pyps $ax_enable_phpips $ax_enable_fortran95 $ax_enable_paws"])



# this takes care of the gnulib stuff
gl_INIT

# devel mode, sets various compiler flags and so on
# this must be the last check because it modifies CFLAGS
AX_ARG_ENABLE([devel-mode],[enable developer mode],[no],
			  [
			   	# set CFLAGS for more warnings
			   	CFLAGS="-g -O0 -Wall -Werror"
				AX_DEPENDS([devel-mode],[])
			  ],[validation]
)

# release mode, sets various compiler flags and so on
AX_ARG_ENABLE([release-mode],[enable release mode],[no],
			  [
			   	# set CFLAGS for more warnings
			   	CFLAGS="-g -O2 -Wall -DNDEBUG"
			  ]
)

# set the optionnal scripts dirs there
AC_SUBST([PIPS_EXTRA_SCRIPTS],["$ax_enable_hpfc $ax_enable_devel_mode $ax_enable_tpips2pyps"])


# register autoconf translated files here
AC_CONFIG_FILES([pips_config.h
				 Makefile
				 pips.pc

				 src/Makefile

				 src/Libs/Makefile
				 src/Libs/alias-classes/Makefile
				 src/Libs/atomizer/Makefile
				 src/Libs/bootstrap/Makefile
				 src/Libs/c_syntax/Makefile
				 src/Libs/callgraph/Makefile
				 src/Libs/chains/Makefile
				 src/Libs/comp_sections/Makefile
				 src/Libs/complexity/Makefile
				 src/Libs/continuation/Makefile
				 src/Libs/control/Makefile
				 src/Libs/conversion/Makefile
				 src/Libs/effects-util/Makefile
				 src/Libs/effects-convex/Makefile
				 src/Libs/effects-generic/Makefile
				 src/Libs/effects-simple/Makefile
				 src/Libs/expressions/Makefile
				 src/Libs/flint/Makefile
				 src/Libs/gpu/Makefile
				 src/Libs/hpfc/Makefile
				 src/Libs/hwac/Makefile
				 src/Libs/hyperplane/Makefile
				 src/Libs/icfg/Makefile
				 src/Libs/instrumentation/Makefile
				 src/Libs/locality/Makefile
				 src/Libs/misc/Makefile
				 src/Libs/movements/Makefile
				 src/Libs/newgen/Makefile
				 src/Libs/phrase/Makefile
				 src/Libs/pipsdbm/Makefile
				 src/Libs/pipsmake/Makefile
				 src/Libs/pointer_values/Makefile
				 src/Libs/preprocessor/Makefile
				 src/Libs/prettyprint/Makefile
				 src/Libs/properties/Makefile
				 src/Libs/reductions/Makefile
				 src/Libs/ri-util/Makefile
				 src/Libs/rice/Makefile
				 src/Libs/ricedg/Makefile
				 src/Libs/sac/Makefile
				 src/Libs/safescale/Makefile
				 src/Libs/scalopes/Makefile
				 src/Libs/semantics/Makefile
				 src/Libs/step/Makefile
				 src/Libs/syntax/Makefile
				 src/Libs/text-util/Makefile
				 src/Libs/to_begin_with/Makefile
				 src/Libs/top-level/Makefile
				 src/Libs/transformations/Makefile
				 src/Libs/accel-util/Makefile
				 src/Libs/pocc-interface/Makefile
				 src/Libs/transformer/Makefile
				 src/Libs/static_controlize/Makefile
				 src/Libs/paf-util/Makefile
				 src/Libs/wp65/Makefile
				 src/Libs/gnulib/Makefile

				 src/Passes/Makefile
				 src/Passes/fortran95/Makefile
				 src/Passes/gpips/Makefile
				 src/Passes/paws/Makefile
				 src/Passes/paws/paws
				 src/Passes/pips/Makefile
				 src/Passes/pyps/Makefile
				 src/Passes/pyps/ipyps
				 src/Passes/pyps/impl/Makefile
				 src/Passes/pyps/doc/Makefile
				 src/Passes/pyps/drivers/Makefile
				 src/Passes/pyps/drivers/binary_size/Makefile
				 src/Passes/pyps/drivers/sac/Makefile
				 src/Passes/pyps/drivers/sac/impl/Makefile
				 src/Passes/pyps/drivers/workspace_gettime/Makefile
				 src/Passes/pyps/drivers/workspace_check/Makefile
				 src/Passes/pyps/drivers/workspace_remote/Makefile
				 src/Passes/pyps/drivers/workspace_git/Makefile
				 src/Passes/pyps/drivers/pypsearch/Makefile
				 src/Passes/pyps/drivers/memalign/Makefile
				 src/Passes/pyps/drivers/sacexplore/Makefile
				 src/Passes/pyps/drivers/p3/Makefile
				 src/Passes/pyps/drivers/openmp/Makefile
				 src/Passes/pyps/drivers/terapyps/Makefile
				 src/Passes/pyps/drivers/broker/Makefile
				 src/Passes/pyps/drivers/pocc/Makefile
         src/Passes/pyps/drivers/validation/Makefile
         src/Passes/pyps/drivers/ir_navigator/Makefile
				 src/Passes/tpips/Makefile

				 src/Passes/phpips/Makefile
				 src/Runtimes/Makefile

				 src/Scripts/Makefile
				 src/Scripts/drivers/Makefile
				 src/Scripts/hpfc/Makefile
                 src/Scripts/step/Makefile
                 src/Scripts/step/step_install
				 src/Scripts/simple_tools/Makefile
				 src/Scripts/simple_tools/pips_c2openmp
				 src/Scripts/simple_tools/pips_f2openmp
				 src/Scripts/validation/Makefile
				 src/Scripts/misc/Makefile
				 src/Scripts/tpips2pyps/Makefile

				 src/Documentation/Makefile
				 src/Documentation/constants/Makefile
				 src/Documentation/help/Makefile
				 src/Documentation/newgen/Makefile
				 src/Documentation/pipsmake/Makefile
				 src/Documentation/dev_guide/Makefile
				 src/Documentation/auto_pips/Makefile
				 src/Documentation/tpips-user-manual/Makefile
				 src/Documentation/linear/Makefile
				 src/Documentation/bugs_and_wishes/Makefile
				 src/Documentation/faq/Makefile
				 src/Documentation/xpomp/Makefile
				 src/Documentation/manl/Makefile
				 src/Documentation/pyps/Makefile
				 ])
				
				 #src/Documentation/doxygen/Makefile
				 #src/Documentation/logo/Makefile
                 #src/Documentation/web/Makefile

                 #src/Libs/pip/Makefile
                 #src/Libs/prgm_mapping/Makefile
                 #src/Libs/reindexing/Makefile
                 #src/Libs/scheduling/Makefile
                 #src/Libs/array_dfg/Makefile
#                 src/Runtimes/hpfc/Makefile
                 #makes/Makefile
#                 src/Passes/fpips/Makefile
#                 src/Passes/pypips/Makefile
#                 src/Passes/tpips/Makefile
#                 src/Passes/wpips/Makefile
#                 src/Scripts/build_release/Makefile
#                 src/Scripts/dev/Makefile
#                 src/Scripts/env/Makefile
#                 src/Scripts/epips/Makefile
#                 src/Scripts/graph/Makefile
#                 src/Scripts/hpfc/Makefile
#                 src/Scripts/jpips/Makefile
#                 src/Scripts/make/Makefile
#                 src/Scripts/misc/Makefile
#                 src/Scripts/stats/Makefile
#                 src/Scripts/step/Makefile
#                 src/Scripts/stf/Makefile
                 

AX_DEPENDS([minimum],[lex yacc polylib64 newgen perl cproto make_all_specs newgenlibs linearlibs std_headers dup2_getcwd_gethostname_gettimeofday_memset_mkdir_putenv_regcomp_setenv_strcasecmp_strchr_strcspn_strerror_strrchr_strspn_strstr_strtol_strtoul_strtoull])

AX_OUTPUT([minimum],[tpips pyps pyps-extra gpips hpfc doc fortran95 paws])
