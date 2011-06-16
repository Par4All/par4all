# ===========================================================================
#     http://www.gnu.org/software/autoconf-archive/ax_python_module.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_PYTHON_MODULE(modname)
#
# DESCRIPTION
#
#   Checks for Python module.
#
#   If fatal is non-empty then absence of a module will trigger an error.
#
# LICENSE
#
#   Copyright (c) 2008 Andrew Collier <colliera@ukzn.ac.za>
#
#   Copying and distribution of this file, with or without modification, are
#   permitted in any medium without royalty provided the copyright notice
#   and this notice are preserved. This file is offered as-is, without any
#   warranty.

#serial 5

AC_DEFUN([AX_PYTHON_MODULE],[
    if test -z "$PYTHON";
    then
        PYTHON="python"
    fi
    PYTHON_NAME=`basename $PYTHON`
    AC_MSG_CHECKING($PYTHON_NAME module: $1)
	$PYTHON  2>/dev/null << EOF
import $1
$2
EOF
	if test $? -eq 0;
	then
		AC_MSG_RESULT(yes)
		AX_WITH([$1])=yes
		eval AS_TR_CPP(HAVE_PYMOD_$1)=yes
	else
		AC_MSG_RESULT(no)
		AX_WITH([$1])=no
        AX_MSG([$1])="failed to find required module $1"
		eval AS_TR_CPP(HAVE_PYMOD_$1)=no
	fi
])
