#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.61)
AC_INIT([linear], [0.1], [pipsdev@cri.ensmp.fr])
AC_CONFIG_SRCDIR([src/polyedre/chernikova.c])
AC_CONFIG_HEADER([config.h])
AC_CONFIG_MACRO_DIR([makes/m4])

AM_INIT_AUTOMAKE([gnu 1.0])

# Checks for programs.
AX_PROG_YACC
AC_PROG_CC_C99
gl_EARLY

# must be right there
AC_PROG_CPP
AX_PROG_LEX
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_LIBTOOL

# Checks for libraries.
AX_CHECK_PKG([polylib64],[Polyhedron_Enumerate],[polylib.h])

# Checks for header files.
AC_FUNC_ALLOCA
AC_HEADER_DIRENT
AC_HEADER_STDC
AX_CHECK_HEADERS([inttypes.h libintl.h limits.h stddef.h stdint.h stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_HEADER_STDBOOL
AC_TYPE_INT16_T
AC_TYPE_INT32_T
AC_TYPE_INT8_T
AC_TYPE_SIZE_T
AC_HEADER_TIME
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT8_T
AC_C_VOLATILE
AX_LD_NO_UNDEFINED

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_TYPE_SIGNAL
AC_FUNC_VPRINTF
AX_CHECK_FUNCS([alarm gettimeofday mkdir])

# Check for cproto
AX_CHECK_CPROTO([4.7])

# set linear cppflags
CPPFLAGS="-DLINEAR_VALUE_IS_LONGLONG -DLINEAR_VALUE_PROTECT_MULTIPLY -DLINEAR_VALUE_ASSUME_SOFTWARE_IDIV"

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
				 linearlibs.pc
                 src/Makefile
				 src/gnulib/Makefile
                 src/arithmetique/Makefile
                 src/contrainte/Makefile
                 src/matrice/Makefile
                 src/matrix/Makefile
                 src/polyedre/Makefile
                 src/polynome/Makefile
                 src/ray_dte/Makefile
                 src/sc/Makefile
                 src/sg/Makefile
                 src/union/Makefile
                 src/sommet/Makefile
                 src/sparse_sc/Makefile
                 src/vecteur/Makefile])

AX_DEPENDS([minimum],[lex yacc polylib64 cproto std_headers alarm_gettimeofday_mkdir])
AX_OUTPUT([minimum])

