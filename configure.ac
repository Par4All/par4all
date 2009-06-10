#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
#
AC_PREREQ(2.63)
AC_INIT([newgen],[0.1],[pipsdev@cri.ensmp.fr])
AM_INIT_AUTOMAKE([gnu 1.0])
AC_CONFIG_SRCDIR([src/genC/build.c])
AC_CONFIG_HEADER([newgen_config.h])

# Checks for programs.
AC_PROG_CPP
AC_PROG_CC
AC_PROG_RANLIB
AC_PROG_YACC
AC_PROG_SED
AM_PROG_LEX

AC_PATH_PROG([CPROTO],[cproto],[not found])
AC_ARG_VAR([CPROTO],[cproto program])

# Checks for libraries.

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([malloc.h memory.h stdint.h stdlib.h string.h strings.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_VPRINTF
AC_CHECK_FUNCS([strdup])

AC_CONFIG_FILES([Makefile
				 src/Makefile
                 src/genC/Makefile
                 src/scripts/Makefile
                 ])
AC_OUTPUT
