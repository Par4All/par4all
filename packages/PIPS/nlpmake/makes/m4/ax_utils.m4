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

AC_DEFUN([AX_MSG],[ax_msg_[]AS_TR_SH([$1])])
AC_DEFUN([AX_WITH],[ax_with_[]AS_TR_SH([$1])])

dnl test for feature result
dnl usage : AX_HAS_OR_DISABLED(progname-or-libname)
dnl expands in a test that checks if this prog or library was found or disabled
AC_DEFUN([AX_HAS_OR_DISABLED],[test "x${AX_WITH([$1])}" != xno ])

dnl test for feature result
dnl usage : AX_HAS(progname-or-libname)
dnl expands in a test that checks if this prog or library was found ?
AC_DEFUN([AX_HAS],[test "x${AX_WITH([$1])}" = xyes ])


dnl check for a function
dnl usage AX_CHECK_FUNC(funcname,[action-if-found],[action-if-not-found])
dnl defines with_funcname=(yes|no) and msg_funcname
AC_DEFUN([AX_CHECK_FUNCS],
    [
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
dnl check for programs
dnl usage : AX_CHECK_PROGS(progname,alternatives,[action-if-found],[action-if-not-found])
dnl defines PROGNAME for further use in the makefiles
dnl checks among progname and alternatives for a valid program
dnl defines with_progname=(yes|no) and msg_progname
AC_DEFUN([AX_CHECK_PROGS],
    [
        AC_MSG_CHECKING([user given $1])
        AS_IF(
            [test "x${AX_TR_UP([$1])}" = x],
            [
                AC_MSG_RESULT([none found])
                AC_PATH_PROG(AX_TR_UP([$1]),[$1],[])
                AS_IF([test "x${AX_TR_UP([$1])}" = x],
                    [
                		AC_PATH_PROGS(AX_TR_UP([$1]),[$2],[])
                		AS_IF([test "x${AX_TR_UP([$1])}" = x],[
                        AX_WITH([$1])=no
                        AX_MSG([$1])="programs $2 not found"
                        m4_ifval([$3],[$3])
						],
						[
							AX_WITH([$1])=yes
							m4_ifval([$4],[$4])
						]
						)

                    ],
                    [
                        AX_WITH([$1])=yes
                        m4_ifval([$4],[$4])
                    ]
                )
            ],
            [
                AC_MSG_RESULT([found, using it])
                AX_WITH([$1])=yes
                m4_ifval([$3],[$3])
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
		m4_foreach_w([_header_],[$1],[
			AC_CHECK_HEADER(_header_,
				[
				AS_IF([test x${AX_WITH([std headers])} = xno ],
					[
					AX_WITH([std headers])=no
					AX_MSG([std headers])="${AX_MSG([std headers])}
    _header_ not found"
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
    _header_ not found"
				m4_ifval([$3],[$3])
				]
			)
			])
    	]
)

dnl create dependecy info
dnl usage : AX_DEPENDS(dependency-var-name, dependencies ...)
dnl defines with_dependency-var-name and msg_dependency-var-name
AC_DEFUN([AX_DEPENDS],
    [
        pushdef([_TEST_],[true ])
        pushdef([_MSG_],[])
        m4_foreach_w([_DEP_],[$2],
            [
                m4_append([_TEST_],&& AX_HAS(AS_TR_SH(_DEP_)))
                m4_append([_MSG_],${AX_MSG(AS_TR_SH([_DEP_]))} )
            ]
        )
		AX_MSG([$1])="_MSG_"
        AS_IF(_TEST_,[AX_WITH([$1])=yes],[AX_WITH([$1])=no])
        m4_popdef([_MSG_])
        m4_popdef([_TEST_])
    ]
)

dnl exit with relevant exit status and an extended configuration message
dnl usage: AX_OUTPUT(required-dependency-var-name,optional-dependency-var-names)
AC_DEFUN([AX_OUTPUT],[
# test if configure is a succes
	pushdef([_TEST_],[AX_HAS($1)])
	m4_foreach_w([_i_],[$2],[m4_append([_TEST_],&& AX_HAS_OR_DISABLED(_i_) )])

# if it's a success, perform substitution
	AS_IF(_TEST_,[AC_OUTPUT])

# in both case print a summary
	eval my_bindir="`eval echo $[]{bindir}`"
	eval my_libdir="`eval echo $[]{libdir}`"
	eval my_docdir="`eval echo $[]{docdir}`"

	cat << EOF

$PACKAGE_NAME Configuration: 
 
  Version:     $VERSION$VERSIONINFO 
 

  Executables: $my_bindir
  Libraries:   $my_libdir
  Docs:        $my_docdir

  Minimal deps ok ? $[]AX_WITH([$1])
$[]AX_MSG([$1])
m4_foreach_w([_i_],[$2],[dnl
  Build _i_ ? $[]AX_WITH(_i_)
  `AS_IF([test x"$[]AX_WITH(_i_)" = "xdisabled"],,echo $[]AX_MSG(_i_))`
])
EOF

# and a conclusion message
	AS_IF(_TEST_,[AS_MESSAGE([Configure succeded])],[
		AS_MESSAGE([Configure failed])
		AS_EXIT([1])
	])
	m4_popdef([_TEST_])
])

dnl configure an optionnal feature
dnl usage AX_ARG_ENABLE(feature-name,help-message,default=[yes|no],configuration-action,[new-variable-value])
dnl sets the shell_variable ax_enable_feature-name to FEATURE-NAME or, if given new-variable-value , if test succeeded
dnl sets the conditionnal WITH_FEATURE-NAME
dnl and set the disable message for AX_HAS(FEATURE-NAME)
AC_DEFUN([AX_ARG_ENABLE],[
		
		AC_ARG_ENABLE([$1],
			[AS_HELP_STRING([--enable-$1],[$2 (defaut is $3)])],
			[AS_IF([test x"$enableval" = "xyes"],[$4],
				[
					AX_WITH([$1])=disabled
					AX_MSG([$1])="$1 disabled"	
				]
			)],
			[
				m4_if([$3],[yes],[$4],
					[
						AX_WITH([$1])=disabled
						AX_MSG([$1])="$1 disabled"
					])
			]
		)
		AS_IF([AX_HAS([$1])],[ax_enable_[]AS_TR_SH($1)=m4_if($5,,[AS_TR_SH($1)],[$5])])
		AM_CONDITIONAL(WITH_[]AX_TR_UP([$1]),[AX_HAS([$1])])
	]
)

dnl checks presence of cproto
dnl usage AX_CHECK_CPROTO(minimum-version-number)
dnl sets AX_WITH(cproto) and AX_MSG(cproto)
AC_DEFUN([AX_CHECK_CPROTO],[
	AX_CHECK_PROG([cproto])
	AS_IF([AX_HAS([cproto])],[
		AC_MSG_CHECKING([wether cproto version is >= $1])
		ax_cproto_version="`$CPROTO -V 2>&1 | sed -e 's/[[a-zA-Z]]//g'`"
		AX_COMPARE_VERSION([$ax_cproto_version],[ge],[$1],[AC_MSG_RESULT([yes ($ax_cproto_version)])],[
			AC_MSG_RESULT([no ($ax_cproto_version)])
			AX_WITH([cproto])="no"
			AX_MSG([cproto])="cproto version too low, $1 minimum required"
		])
	])
])

dnl checks for yacc and sets ax_ variables accordingly
AC_DEFUN([AX_PROG_YACC],[
	AC_PROG_YACC
	AS_IF([test "${YACC}" = yacc],[
		unset YACC
		AX_CHECK_PROG([yacc])
	],[
		AX_WITH([yacc])="yes"
		AX_MSG([yacc])=""
	])
])

dnl checks for lex in an ax_ compatible way
AC_DEFUN([AX_PROG_LEX],[
	AC_PROG_LEX
	AS_IF([test "${LEX}" = ":"],[
		AX_WITH([lex])="no"
		AX_MSG([lex])="[f]lex not found"
	],[
		AX_WITH([lex])="yes"
		AX_MSG([lex])=""
	])
	AS_IF([test x"$LEXLIB" != x],[
		AC_DEFINE([HAVE_LEXLIB],[1],[Defined if lexlib is used])
		])
])

dnl checks how to checks for undefined symbols
dnl in the linker.
dnl valid option is made available in LDFLAGS_NO_UNDEFINED
AC_DEFUN([AX_LD_NO_UNDEFINED],[
	AC_MSG_CHECKING([how to forbid unresolved references in object files])
	ax_saved_ldflags="$LDFLAGS"
	LDFLAGS="$LDFLAGS -Wl,--no-undefined"
	AC_TRY_LINK([],[void foo(){}],[
			AC_SUBST([LDFLAGS_NO_UNDEFINED],[-Wl,--no-undefined])
			AC_MSG_RESULT([found])],
		[AC_MSG_RESULT([not found])]
	)
	LDFLAGS="$ax_saved_ldflags"
])

