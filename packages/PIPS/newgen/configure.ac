#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.61)
AC_INIT([newgen], [0.1], [pipsdev@cri.ensmp.fr])
AC_CONFIG_SRCDIR([src/genC/genC.c])
AC_CONFIG_HEADER([config.h])
AC_CONFIG_MACRO_DIR([makes/m4])

AM_INIT_AUTOMAKE([gnu 1.0])
AC_LANG([C])

# Checks for programs.
AX_PROG_YACC
AC_PROG_CC_C99
AC_PROG_CPP
AX_PROG_LEX
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_LIBTOOL
AX_CHECK_CPROTO([4.7])

# Checks for libraries.

# Checks for header files.
AC_HEADER_STDC
AX_CHECK_HEADERS([inttypes.h memory.h stdint.h stdlib.h string.h strings.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_MEMCMP
AC_FUNC_REALLOC
AC_FUNC_VPRINTF
AX_CHECK_FUNCS([strdup])

AC_CONFIG_FILES([Makefile
				 newgenlibs.pc
                 src/Makefile
                 src/genC/Makefile
                 src/genC/genread/Makefile
                 src/genC/genspec/Makefile
                 src/scripts/Makefile
                 ])
AX_DEPENDS([minimum],[lex yacc cproto std_headers strdup])
AX_OUTPUT([minimum])
