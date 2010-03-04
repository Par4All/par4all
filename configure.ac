#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.61)
AC_INIT([pips],[0.1], [pipsdev@cri.ensmp.fr])
AC_CONFIG_SRCDIR([src/Passes/pips/pips.c])
AC_CONFIG_HEADER([pips_config.h])
AC_CONFIG_MACRO_DIR([makes/m4])
AM_INIT_AUTOMAKE([gnu 1.0])
AM_PROG_CC_C_O
AM_PATH_PYTHON(2.5)

# Checks for programs.
AC_PROG_YACC
AC_PROG_CC_C99
gl_EARLY
AC_PROG_CPP
AC_PROG_LEX
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_LIBTOOL
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
			 AX_CHECK_PKG([termcap],[tgetent],[term.h])
			 AX_DEPENDS([tpips],[readline termcap])
			]
)

# pyps
AX_ARG_ENABLE([pyps],
			[enable python interface],
			[no],
			[
			 AC_PROG_SWIG
			 AC_PYTHON_DEVEL([2.5])
			 AX_DEPENDS([pyps],[swig python_devel])
			]
)

# hpfc
AX_ARG_ENABLE([hpfc],
			  [enable high performance fortran compiler],
			  [no],
			  [
			   AX_CHECK_PKG([readline],[readline],[readline/readline.h])
			   AX_CHECK_PKG([termcap],[tgetent],[term.h])
			   AX_DEPENDS([hpfc],[readline termcap])
			  ]
)

# set the optionnal passes dirs there
AC_SUBST([PIPS_EXTRA_PASSES],["$ax_enable_gpips $ax_enable_tpips $ax_enable_pyps"])

# set the optionnal scripts dirs there
AC_SUBST([PIPS_EXTRA_SCRIPTS],["$ax_enable_hpfc"])

# this takes care of the gnulib stuff, the CPPFLAGS change is mandatory
gl_INIT
CPPFLAGS="${CPPFLAGS} -I\$(top_builddir)/src/Libs/gnulib"

AC_CONFIG_FILES([Makefile
                 src/Documentation/Makefile
                 src/Documentation/constants/Makefile
                 src/Documentation/help/Makefile
                 src/Documentation/newgen/Makefile
                 src/Documentation/pipsmake/Makefile
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
                 src/Libs/transformer/Makefile
                 src/Libs/static_controlize/Makefile
                 src/Libs/paf-util/Makefile
                 src/Libs/wp65/Makefile
				 src/Libs/gnulib/Makefile
                 src/Makefile
                 src/Passes/Makefile
                 src/Passes/gpips/Makefile
                 src/Passes/pips/Makefile
                 src/Passes/pyps/Makefile
                 src/Passes/tpips/Makefile
                 src/Runtimes/Makefile
                 src/Scripts/Makefile
                 src/Scripts/drivers/Makefile
                 src/Scripts/hpfc/Makefile
                 src/Scripts/simple_tools/Makefile
                 src/Scripts/simple_tools/pips_c2openmp
                 src/Scripts/simple_tools/pips_f2openmp
])
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
#                 src/Scripts/validation/Makefile
                 #src/Documentation/dev_guide/Makefile
                 #src/Documentation/doxygen/Makefile
                 #src/Documentation/logo/Makefile
                 #src/Documentation/manl/Makefile
                 #src/Documentation/tpips-user-manual/Makefile
                 #src/Documentation/web/Makefile
AC_OUTPUT


AX_DEPENDS([minimum],[newgen perl cproto make_all_specs newgenlibs linearlibs std_headers dup2_getcwd_gethostname_gettimeofday_memset_mkdir_putenv_regcomp_setenv_strcasecmp_strchr_strcspn_strerror_strrchr_strspn_strstr_strtol_strtoul_strtoull])

AX_EXIT([minimum],[tpips pyps gpips hpfc])
