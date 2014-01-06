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

# Reference file for PIPS environment variables.
# A restricted version of this file should be defined for pips users.
# The syntax is compatible with both sh and csh... edit with care!
#
# This files sets as expected: {NEWGEN,LINEAR,PIPS,EXTERN}_{ROOT,ARCH,DEVEDIR}
# and also {,MAN}PATH.

set -a

#
#           ##    #####    ####   #    #
#          #  #   #    #  #    #  #    #
#         #    #  #    #  #       ######
#         ######  #####   #       #    #
#         #    #  #   #   #    #  #    #
#         #    #  #    #   ####   #    #
#

# hum.... uname output vary quite a lot... :-(
# on sunos, they are not orderer as expected...
# on sol2, the version is "generic"...
# on aix, the v and r seems inverted wrt sun...
uname_s=`uname -s`
# uname_v=`uname -v`
uname_r=`uname -r`

# default architecture for building pips
export PIPS_ARCH="GNU"

# the makefile_macros.$(PIPS_ARCH) file is used for compilers and options

test ${uname_s} = SunOS && expr ${uname_r} : '^5' > /dev/null && \
	export PIPS_ARCH="GNUSOL2LL"

test ${uname_s} = SunOS && expr ${uname_r} : '^4' > /dev/null && \
	export PIPS_ARCH="GNUSOL1"

test ${uname_s} = Linux && \
	export PIPS_ARCH="LINUXI86LL"

test ${uname_s} = FreeBSD && \
	export PIPS_ARCH="FREEBSDLL"

# 
#         #    #  ######  #    #   ####   ######  #    #
#         ##   #  #       #    #  #    #  #       ##   #
#         # #  #  #####   #    #  #       #####   # #  #
#         #  # #  #       # ## #  #  ###  #       #  # #
#         #   ##  #       ##  ##  #    #  #       #   ##
#         #    #  ######  #    #   ####   ######  #    #
#

# the localtion of newgen-related files
newgen_dir=/projects/Newgen

# the root location of where to find the newgen to use 
export NEWGEN_ROOT=${newgen_dir}/prod

# PATH update
export PATH="${PATH}:${NEWGEN_ROOT}/bin:${NEWGEN_ROOT}/bin/${PIPS_ARCH}"

#
#         #          #    #    #  ######    ##    #####
#         #          #    ##   #  #        #  #   #    #
#         #          #    # #  #  #####   #    #  #    #
#         #          #    #  # #  #       ######  #####
#         #          #    #   ##  #       #    #  #   #
#         ######     #    #    #  ######  #    #  #    #
#

# the location of the C3/Linear files
linear_dir=/projects/C3/Linear

# the version to use
export LINEAR_ROOT=${linear_dir}/prod

#
#         ######   ####   #       ######
#         #       #    #  #       #
#         #####   #    #  #       #####
#         #       #    #  #       #
#         #       #    #  #       #
#         ######   ####   ######  ######
#
# Evaluation Optimization for Loops and Expressions
#
# Julien ZORY Stuff based on CAVEAT and STORM.
#

eole_dir=/projects/Eole

export EOLE_ROOT=${eole_dir}/prod

PATH="${PATH}:${EOLE_ROOT}/bin/${PIPS_ARCH}"

# 
#         #####      #    #####    ####
#         #    #     #    #    #  #
#         #    #     #    #    #   ####
#         #####      #    #####        #
#         #          #    #       #    #
#         #          #    #        ####
# 


############################################################### PIPS LOCATION

# Pips project sources and binaries
pips_dir=/projects/Pips

export PIPS_ROOT="${pips_dir}/prod"

# man pages?
MANPATH="${MANPATH}:${PIPS_ROOT}/man"

PATH="${PATH}:${PIPS_ROOT}/bin:${PIPS_ROOT}/bin/${PIPS_ARCH}"

# to get some determinism...

export PIPS_CPP='gcc -E -C'
export PIPS_FPP='gcc -E -C'
test "${PIPS_ARCH}" = GNUSOL2LL && \
	PIPS_FPP="fpp"

# Additional options to pass to cpp for .F files:
# default: PIPS_CPP_FLAGS=""


############################################################## COMPILING PIPS

# make MUST be gmake: thus this is the default. 
# redefine this variable if you wish to change the default path or the name...
# PIPS_MAKE='gmake'

################################################################ TESTING PIPS

# Validation directory
# PIPS_VALIDDIR="${PIPS_ROOT}/Validation"

# hosts for Validate:
test "${PIPS_ARCH}" = GNUSOL2LL && \
	export PIPS_HOSTS="champeaux"

test "${PIPS_ARCH}" = . -o "${PIPS_ARCH}" = GNU && \
	export PIPS_HOSTS=""

test "${PIPS_ARCH}" = LINUXI86 -o "${PIPS_ARCH}" = LINUXI86LL && \
	export PIPS_HOSTS=""

test "${PIPS_ARCH}" = FREEBSD -o "${PIPS_ARCH}" = FREEBSDLL && \
	export PIPS_HOSTS=""

#
# others... they are mainly documented in install_pips...
#

test ${uname_s} = Linux && \
	export PIPS_PING="ping -c 1"

# under solaris basename is refexpr based.
test ${uname_s} = SunOS && \
	export PIPS_BASENAME="/usr/ucb/basename"

test ${uname_s} = Linux && \
	export PIPS_FLINT="g77 -c -Wall -Wimplicit"

test ${uname_s} = Linux && \
	export PIPS_MAKE="make"

test ${uname_s} = FreeBSD && \
	export PIPS_MAKE="gmake"


#
#         #    #  #####      #    #####    ####
#         #    #  #    #     #    #    #  #
#         #    #  #    #     #    #    #   ####
#         # ## #  #####      #    #####        #
#         ##  ##  #          #    #       #    #
#         #    #  #          #    #        ####
# 
# Settings added for WPIPS (used for wpips and wtest)
# home directory for openwin (only used by the XView interface: wpips)
#

export OPENWINHOME="/usr/openwin"

export X11_ROOT="$OPENWINHOME"

test ${uname_s} = Linux && \
	export X11_ROOT="/usr/X11"



# 
#        ######  #####      #    #####    ####
#        #       #    #     #    #    #  #
#        #####   #    #     #    #    #   ####
#        #       #####      #    #####        #
#        #       #          #    #       #    #
#        ######  #          #    #        ####
#
# Settings added for EPIPS (Emacs PIPS).
# started with the epips command.
#
# emacs to be used by epips. should be gnu emacs.
# you may change the default by defining this variable
# default: EPIPS_EMACS='emacs'

#test ${uname_s} = SunOS && EPIPS_EMACS='Emacs'


#
# emacs-lisp code for epips
# you may change this default by defining this variable
# default: EPIPS_LISP="${PIPS_ROOT}/Share/epips.el" 



#
# 
#         #    #  #####   ######   ####
#         #    #  #    #  #       #    #
#         ######  #    #  #####   #
#         #    #  #####   #       #
#         #    #  #       #       #    #
#         #    #  #       #        ####
# 
# HPFC environment variables 
#
# where HPFC runtime can be found
# default: HPFC_RUNTIME=$PIPS_ROOT/Runtime/hpfc
#
# some commands
# default: HPFC_SED='gsed'
# default: HPFC_TR='tr'
# default: HPFC_MAKE='gmake'
# default: HPFC_M4='gm4'

test ${uname_s} = Linux && \
	export HPFC_SED=sed
test ${uname_s} = Linux && \
	export HPFC_M4=m4


#
#                         #####  #######
#         #    #  #####  #     # #
#         #    #  #    # #       #
#         #    #  #    # ######  ######
#         # ## #  #####  #     #       #
#         ##  ##  #      #     # #     #
#         #    #  #       #####   #####
#
#
# where the WP65 runtime is expected
# (can be changed to $PIPS_DEVEDIR/Runtime/wp65 to link with dev)
# default: WP65_LIBRARY=$PIPS_ROOT/Runtime/wp65



# 
#         #    #     #     ####    ####
#         ##  ##     #    #       #    #
#         # ## #     #     ####   #
#         #    #     #         #  #
#         #    #     #    #    #  #    #
#         #    #     #     ####    ####
# 
#
# EXTERNAL softwares

# An absolute file name is necessary for LISP
#LISP=/usr/local/bin/cl

# PVM directory
#PVMDIR=${pips_home}/pvm3.3

#
# For the daVinci Graph Viewer/Editor :

#DAVINCIHOME="${pips_home}/daVinci/daVinci_V2.0"
#GRAPHEDITOR_DIRECTORY="${DAVINCIHOME}/grapheditor"
#DAVINCI_ICONDIR="${DAVINCIHOME}/images"

#MANPATH="${MANPATH}:${DAVINCIHOME}/man"
#PATH="${PATH}:${DAVINCIHOME}"

# Toolpack is used for STF
#TOOLPACK_DIR="${pips_home}/Toolpack/toolpack1.2"



#
#         ######  #    #   #####  ######  #####   #    #
#         #        #  #      #    #       #    #  ##   #
#         #####     ##       #    #####   #    #  # #  #
#         #         ##       #    #       #####   #  # #
#         #        #  #      #    #       #   #   #   ##
#         ######  #    #     #    ######  #    #  #    #
#


# libraries which are not part of Pips project
# the readline or rx libraries may be expected here for instance, 
# if not available in a standard location.

export EXTERN_ROOT="${pips_dir}/Externals"

PATH="${PATH}:${EXTERN_ROOT}/Share:${EXTERN_ROOT}/Utils"
PATH="${PATH}:${EXTERN_ROOT}/Bin/${PIPS_ARCH}"

#LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${EXTERN_ROOT}/Lib/${PIPS_ARCH}"



#
#          ####   #       #####
#         #    #  #       #    #
#         #    #  #       #    #
#         #    #  #       #    #
#         #    #  #       #    #
#          ####   ######  #####
#
# OBSOLETE variables

# that is all for the pips reference environment file.
#
