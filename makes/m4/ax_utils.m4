dnl 
dnl This file is part of hyantes.
dnl 
dnl hypersmooth is free software; you can redistribute it and/or modify
dnl it under the terms of the CeCILL-C License
dnl 
dnl You should have received a copy of the CeCILL-C License
dnl along with this program.  If not, see <http://www.cecill.info/licences>.
dnl 

m4_pattern_forbid([^_?AX_])
m4_pattern_allow([^_PKG_ERRORS$])

dnl return the string in uppercase suitable for shell variable names
dnl usage AX_TR_UP(name)
dnl expands in a reworked name
AC_DEFUN([AX_TR_UP],[m4_translit([AS_TR_SH([$1])],[a-z],[A-Z])])

AC_DEFUN([AX_MSG],[msg_[]AS_TR_SH([$1])])
AC_DEFUN([AX_WITH],[with_[]AS_TR_SH([$1])])

dnl test for feature result
dnl usage : AX_HAS(progname-or-libname)
dnl expands in a test that checks if this prog or library was found ?
AC_DEFUN([AX_HAS],[AC_REQUIRE([AX_WITH]) test "x${AX_WITH([$1])}" = xyes ])


dnl check for a function
dnl usage AX_CHECK_FUNC(funcname,[action-if-found],[action-if-not-found])
dnl defines with_funcname=(yes|no) and msg_funcname
AC_DEFUN([AX_CHECK_FUNCS],
    [
        AC_REQUIRE([AX_MSG])dnl
        AC_REQUIRE([AX_WITH])dnl
        AC_CHECK_FUNCS([$1],
            [
                AX_WITH([$1])=yes
                m4_ifval([$2],[$2])dnl
            ],
            [
                AX_WITH([$1])=no
                AX_MSG([$1])="function $ac_func not found
${AX_MSG([$1])}"
                m4_ifval([$3],[$3])dnl
            ]
        )
    ]
)
        
dnl check for a libray and its headers
dnl usage : AX_CHECK_PKG(libname,[function],[headers],[action-if-found],[action-if-not-found])
dnl defines LIBNAME_CFLAGS and LIBNAME_LIBS for further use in makefiles
dnl defines with_libname=(yes|no) and msg_libname
AC_DEFUN([AX_CHECK_PKG],
    [
dnl        AC_REQUIRE([PKG_CHECK_MODULES])dnl
dnl        AC_REQUIRE([AX_HAS])dnl
dnl        AC_REQUIRE([AX_TR_UP])dnl
dnl        AC_REQUIRE([AX_MSG])dnl
dnl        AC_REQUIRE([AX_WITH])dnl

        PKG_CHECK_MODULES(AX_TR_UP([$1]),[$1],
            [
				AX_WITH([$1])=yes
			],
            [
                m4_ifval([$2],
                    [
                        m4_ifval([$3],
                            [
                                AC_CHECK_HEADERS(
                                    [$3],[],
                                    [
                                        AX_WITH([$1])=no
                                        AX_MSG([$1])="header $3 not found"
                                    ]
                                )
                            ]
                        )
                        AC_CHECK_LIB(
                            [$1],
                            [$2],
                            [
                                AX_TR_UP([$1])_LIBS="-l$1"
                                AS_IF([test x${AX_WITH([$1])} = xno],[],[AX_WITH([$1])=yes])
                            ],
                            [
                                AX_WITH([$1])=no
                                AX_MSG([$1])="library $1 not found
${AX_MSG([$1])}"
                            ]
                        )
                    ],
                    [
                        AX_WITH([$1])=no
                        AX_MSG([$1])="pkg-config info for $1 not found"
                    ]
                )
            ]
        )
        AS_IF([AX_HAS([$1])],[$4],[$5])
    ]
)

dnl check fo a program
dnl usage : AX_CHECK_PROG(progname,[action-if-found],[action-if-not-found])
dnl defines PROGNAME for further use in the makefiles
dnl defines with_progname=(yes|no) and msg_progname
AC_DEFUN([AX_CHECK_PROG],
    [
        AC_REQUIRE([AX_TR_UP])dnl
        AC_REQUIRE([AX_MSG])dnl
        AC_REQUIRE([AX_WITH])dnl
        AC_MSG_CHECKING([user given AX_TR_UP([$1])])
        AS_IF(
            [test "x${AX_TR_UP([$1])}" = x],
            [
                AC_MSG_RESULT([none found])
                AC_PATH_PROG(AX_TR_UP([$1]),[$1],[])
                AS_IF([test "x${AX_TR_UP([$1])}" = x],
                    [
                        AX_WITH([$1])=no
                        AX_MSG([$1])="program $1 not found"
                        m4_ifval([$2],[$2])
                    ],
                    [
                        AX_WITH([$1])=yes
                        m4_ifval([$3],[$3])
                    ]
                )
            ],
            [
                AC_MSG_RESULT([found, using it])
                AX_WITH([$1])=yes
                m4_ifval([$2],[$2])
            ]
        )
        AC_ARG_VAR(AX_TR_UP([$1]),[path to the $1 command, overriding tests])
        AC_SUBST(AX_TR_UP([$1]))
    ]
)


dnl check for headers
dnl usage : AX_CHECK_HEADERS(header ...,[action-if-found],[action-if-not-found])
dnl defines with_std_headers=(yes|no) and msg_system_headers
AC_DEFUN([AX_CHECK_HEADERS],
    [
        AC_REQUIRE([AX_MSG])dnl
        AC_REQUIRE([AX_WITH])dnl
        AC_CHECK_HEADERS([$1],
            [
                AS_IF([test x${AX_WITH([std headers])} = xno ],
                    [
                        AX_WITH([std headers])=no
						AX_MSG([std headers])="${AX_MSG([std headers])}
$ac_header not found"
                        m4_ifval([$3],[$3])dnl
                    ],
                    [
                        AX_WITH([std headers])=yes
                        m4_ifval([$2],[$2])dnl
                    ]
                )
            ],
            [
                AX_WITH([std headers])=no
                AX_MSG([std headers])="${AX_MSG([std headers])}
$ac_header not found"
                m4_ifval([$3],[$3])
            ]
        )
    ]
)

dnl create dependecy info
dnl usage : AX_DEPENDS(dependency-var-name, dependencies ...)
dnl defines with_dependency-var-name and msg_dependency-var-name
AC_DEFUN([AX_DEPENDS],
    [
        AC_REQUIRE([AX_MSG])dnl
        AC_REQUIRE([AX_WITH])dnl
        pushdef([_TEST_],[true ])
        pushdef([_MSG_],[])
        m4_foreach_w([_DEP_],[$2],
            [
                m4_append([_TEST_],&& AX_HAS(AX_TR_UP(_DEP_)))
                m4_append([_MSG_],${AX_MSG(AX_TR_UP([_DEP_]))} )
            ]
        )
		AX_MSG([$1])="_MSG_"
        AS_IF(_TEST_,[AX_WITH([$1])=yes],[AX_WITH([$1])=no])
        m4_popdef([_MSG_])
        m4_popdef([_TEST_])
    ]
)

dnl exit with relevant exit status and a small message
dnl dnl usage: AX_EXIT(dependency-var-name)
AC_DEFUN([AX_EXIT],[AS_IF([AX_HAS([$1])],[AS_MESSAGE([Configure suceeded])],[AS_EXIT([Configure failed])])])

