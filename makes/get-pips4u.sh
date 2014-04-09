#! /usr/bin/env bash
################################################################################
# install-pips4u - Install PIPS (see http://pips4u.org/)
# Creation : 04 Mar 2010
# Time-stamp: <Ven 2010-03-05 09:13 svarrette>
#
# Copyright (c) 2010 Sebastien Varrette <Sebastien.Varrette@uni.lu>
#               http://varrette.gforge.uni.lu
# $Id$
#
# Description : see the print_help function or launch 'install-pips4u --help'
# Based on the get-pips4u designed by Serge Guelton <Serge.Guelton@telecom-bretagne.eu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
################################################################################

### Global variables
VERSION=0.1
COMMAND=`basename $0`
VERBOSE=""
DEBUG=""
SIMULATION=""
FORCE_MODE=""
DEVEL_MODE=""

error=0 ; trap "error=$((error|1))" ERR

### displayed colors
# ??? is that portable?
COLOR_GREEN="\033[0;32m"
COLOR_RED="\033[0;31m"
COLOR_YELLOW="\033[0;33m"
COLOR_VIOLET="\033[0;35m"
COLOR_CYAN="\033[0;36m"
COLOR_BOLD="\033[1m"
COLOR_BACK="\033[0m"

### Local variables
# PIPS version (at least for the autotools version ;)
# ??? why this suffix? it does not make sense with --devel at least.
# ??? should it be named PIPS_VERSION instead?
# ??? where does this version number come from?
# ??? the real 0.9.0 tag created in 2005 is anterior to this version...
PIPS_AUTOTOOLS_VERSION=0.1
# default flags for the configure script
PIPS_CONFIGURE_COMMON_FLAGS=" --disable-static"
# install prefix passed by --prefix to the configure script
INSTALL_PREFIX=$HOME/pips4u-${PIPS_AUTOTOOLS_VERSION}
# local working directory where the sources to be configured/compiled will be retrieved
SRC_DIR=$PWD/pips4u-${PIPS_AUTOTOOLS_VERSION}/src

# ---  Library to install ---
# user mode i.e retrieve tgz sources
TGZ_POLYLIB=http://icps.u-strasbg.fr/polylib/polylib_src/polylib-5.22.5.tar.gz
TGZ_NEWGEN=http://ridee.enstb.org/pips/newgen-${PIPS_AUTOTOOLS_VERSION}.tar.gz
TGZ_LINEAR=http://ridee.enstb.org/pips/linear-${PIPS_AUTOTOOLS_VERSION}.tar.gz
TGZ_PIPS=http://ridee.enstb.org/pips/pips-${PIPS_AUTOTOOLS_VERSION}.tar.gz

# devel mode (i.e. checkout from the SVN tree)
# ??? https is prefered?
SVN_NEWGEN=https://scm.cri.ensmp.fr/svn/newgen/trunk
SVN_LINEAR=https://scm.cri.ensmp.fr/svn/linear/trunk
SVN_PIPS=https://scm.cri.ensmp.fr/svn/pips/trunk
# ??? what about the validation?
# ??? what about the user development branches?

#######################
### print functions ###
#######################

####
# print version of this program
##
print_version() {
    cat <<EOF
This is $COMMAND version "$VERSION".
Copyright (c) 2010 Sebastien Varrette <Sebastien.Varrette@uni.lu>
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
EOF
}

####
# print help
##
print_help() {
cat <<EOF
NAME
	$COMMAND -- Install PIPS (see http://pips4u.org/) using either the latest
                    svn sources tree or remote tarballs

SYNOPSIS
	$COMMAND [-V | -h]
	$COMMAND [--debug] [-v] [-n]
        $COMMAND [--prefix PREFIX] [--devel [USERNAME]]

DESCRIPTION
	$COMMAND helps the automatic installation of PIPS using the nice port to
        Autotools done by Serge Guelton.
        Two modes are available:
           * by default tarballs of the main packages are retrieved
           * in devel mode (using --devel), the package sources are retrieved from SVN.

OPTIONS
	--debug
		Debug mode. Causes $COMMAND to print debugging messages.
        --devel [USERNAME]
                install PIPS in devel mode i.e. from the latest version available on the
                SVN repository of each package.
                Eventually, precise the username to be used for the checkout operation.
        -f --force
                do no interactively ask to continue
	-h --help
		Display a help screen and quit.
	-n --dry-run
		Simulation mode.
        --prefix PREFIX
                install architecture-independent files in PREFIX
		Default: [${INSTALL_PREFIX}]
        --srcdir DIR
                local working directory where the sources to be configured/compiled will
                be retrieved
                Default: [${SRC_DIR}]
	-v --verbose
		Verbose mode.
	-V --version
		Display the version number then quit.

SIGNIFICANT ENVIRONMENT VARIABLES
        PKG_CONFIG_PATH: additionnal pkg-config dirs
        LD_LIBRARY_PATH: additionnal shared libraries dirs
        PATH: additionnal binaries dir
        PIPS_CONFIG: extra pips config flags

        For instance, to activate the pyps module of PIPS, you will run:

        PIPS_CONFIG="--enable-pyps" $COMMAND

        Note: I run Mac OS X (snow leopard) and here is the command I used to compile PIPS
        (note the usage of stow which I (as always) strongly suggest):

        PIPS_CONFIG="--enable-pyps LDFLAGS=-L/usr/lib" $COMMAND --devel --prefix /usr/local/stow/pips-0.1


AUTHOR
	Sebastien Varrette <Sebastien.Varrette@uni.lu>
	Web page: http://varrette.gforge.uni.lu

REPORTING BUGS
	Please report bugs to <Sebastien.Varrette@uni.lu>

COPYRIGHT
	This is free software; see the source for copying conditions.  There is
	NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR
	PURPOSE.

SEE ALSO
	Other scripts are available on my web site http://varrette.gforge.uni.lu
EOF
}

######
# Print information in the following form: '[$2] $1' ($2=INFO if not submitted)
# usage: info text [title]
##
info() {
    [ -z "$1" ] && print_error_and_exit "[$FUNCNAME] missing text argument"
    local text=$1
    local title=$2
    # add default title if not submitted but don't print anything
    [ -n "$text" ] && text="${title:==>} $text"
    echo -e $text
}
debug()   { [ -n "$DEBUG"   ] && info "$1" "[${COLOR_YELLOW}DEBUG${COLOR_BACK}]"; }
verbose() { [ -n "$VERBOSE" ] && info "$1"; }
error()   { info "$1" "[${COLOR_RED}ERROR${COLOR_BACK}]"; }
warning() { info "$1" "[${COLOR_VIOLET}WARNING${COLOR_BACK}]"; }
print_error_and_exit() {
    local text=$1
    [ -z "$1" ] && text=" Bad format"
    error  "$text. '$COMMAND -h' for help."
    exit 1
}
#####
# print the strings [ OK ] or [ FAILED ] or [ FAILED ]\n$1
##
print_ok()     { echo -e "[   ${COLOR_GREEN}OK${COLOR_BACK}   ]"; }
print_failed() { echo -e "[ ${COLOR_RED}FAILED${COLOR_BACK} ]"; }
print_failed_and_exit() {
    print_failed
    [ ! -z "$1" ] && echo "$1"
    exit 1
}

#########################
### toolbox functions ###
#########################

#####
# execute a local command
# usage: execute command
###
execute() {
    [ $# -eq 0 ] && print_error_and_exit "[$FUNCNAME] missing command argument"
    debug "[$FUNCNAME] $*"
    [ -n "${SIMULATION}" ] && echo "(simulation) $*" || eval $*
    local exit_status=$?
    debug "[$FUNCNAME] exit status: $exit_status"
    [ $exit_status = 0 ] || print_error_and_exit "[$FUNCNAME] failed"
    return $exit_status
}

####
# ask to continue. exit 1 if the answer is no
# usage: really_continue text
##
really_continue() {
    if [ -z "${FORCE_MODE}" ]; then
        echo -e -n "[${COLOR_VIOLET}WARNING${COLOR_BACK}] $1 Are you sure you want to continue? [Y|n] "
        read ans
        case $ans in
	    n*|N*) exit 1;;
        esac
    else
        [ -n "$1" ] && info "$1"
    fi
}

#####
# Check availability of binaries passed as arguments on the current system
# usage: check_bin prog1 prog2 ...
##
check_bin() {
    [ $# -eq 0 ] && print_error_and_exit "[$FUNCNAME] missing argument"
    for appl in $*; do
	echo -n -e "=> checking availability of the command '$appl' on your system \t"
	local tmp=`which $appl`
	[ -z "$tmp" ] && print_failed_and_exit "Please install $appl or check \$PATH." || print_ok
    done
}

#####
# Operate a 'svn checkout' or 'svn update' whether what is the most appropriate
# Usage:  svn_co_or_up repo_name svn_url
# ex:
##
svn_co_or_up() {
    [ $# -le 1 ] && print_error_and_exit "[$FUNCNAME] missing argument"
    local repo=$1
    local svn_url=$2
    # Add eventual username
    [ -n "${USERNAME}" ] && SVN="${SVN} --username ${USERNAME}"
    debug "repo name = $repo"
    debug "svn_url = $svn_url"
    if [ ! -d $repo ]; then
        info "retrieving ${COLOR_BOLD}$repo${COLOR_BACK} by svn (co)"
        execute "${SVN} co $svn_url $repo"
    else
        info "updating svn repository ${COLOR_BOLD}$repo${COLOR_BACK}"
        execute "cd $repo"
        execute "svn up"
        execute "cd -"
    fi

}

####
# wget a remote tgz specified by $1
# Usage: get_remote_tgz url
##
get_remote_tgz() {
    [ $# -eq 0 ] && print_error_and_exit "[$FUNCNAME] missing argument"
    local url=$1
    local archive=`basename $url`
    local basedir=`basename $url .tar.gz`
    debug "[$FUNCNAME] wget on $url"
    #{
    if [ ! -r $archive ]; then
        info "retrieving $archive via wget"
        execute "wget -nd --quiet $url"
    fi
    if [ ! -d $basedir ]; then
        info "uncompressing $archive "
        execute "tar xzf $archive"
    fi
    #} &> ${INSTALL_LOGGER}
}

#####
# build a given part of PIPS
# Usage: build dir
###
build() {
    [ $# -eq 0 ] && print_error_and_exit "[$FUNCNAME] missing argument"
    local dir=$1
    echo ""
    really_continue "building ${COLOR_BOLD}$dir${COLOR_BACK}."
    [ ! -d "$dir" -a -z "${SIMULATION}" ] && print_error_and_exit "directory $dir do not exists"
    execute "cd $dir"
    [ -n "${DEVEL_MODE}" ] && [ ! -f configure ] && info "running ${AUTORECONF}" && execute "${AUTORECONF}"
    execute "mkdir -p _build"
    execute "cd _build"
    info "configuring with options:  ${COLOR_BOLD}${PIPS_CONFIGURE_COMMON_FLAGS}${COLOR_BACK}"
    [ -f Makefile ] || execute "../configure ${PIPS_CONFIGURE_COMMON_FLAGS}"
    really_continue "you are about to launch the compilation (make/make install)."
    execute "make"
    info "running make install"
    execute "make install"
    execute "cd ../.."
}


################################################################################
################################################################################
#[ $UID -gt 0 ] && print_error_and_exit "You must be root to execute this script (current uid: $UID)"

# Check for required argument
#[ $# -eq 0 ] && print_error_and_exit

# Check for options
while [ $# -ge 1 ]; do
    case $1 in
	-h | --help)     print_help;        exit 0;;
	-V | --version)  print_version;     exit 0;;
	-f | --force)    FORCE_MODE="--force";;
	--debug)         DEBUG="--debug";
	                 VERBOSE="--verbose";;
	-v | --verbose)  VERBOSE="--verbose";;
 	-n | --dry-run)  SIMULATION="--dry-run";;
        --devel)
	    # ??? should include the validation!
	    # ??? version should be "dev" instead of "0.1"
            DEVEL_MODE=1;
            # check first character of $2: should differ from '-'
            if [ $# -ne 1 ] && [ ${2:0:1} != "-" ]; then
                 USERNAME=$2;
                 shift;
            fi;;
        --srcdir)
            # check first character of $2: should differ from '-'
            if [ $# -ne 1 ] && [ ${2:0:1} != "-" ]; then
                SRC_DIR=$2;
                shift;
            else
                warning "src dir not given correctly and ignored"
            fi;;
         --prefix)
            # check first character of $2: should differ from '-'
            if [ $# -ne 1 ] && [ ${2:0:1} != "-" ]; then
                INSTALL_PREFIX=$2;
                shift;
            else
                warning "prefix not given correctly and ignored"
            fi;;
    esac
    shift
done

if [ -z "${INSTALL_PREFIX}" ] || [ "${INSTALL_PREFIX:0:1}" == "-" ]; then
    print_error_and_exit "prefix not given";
fi
[ -z "${SRC_DIR}"  ] && print_error_and_exit "source dir not given";

cat << EOF
================================================================================
           PIPS4U install script

Prefix: ${INSTALL_PREFIX}
Source dir: ${SRC_DIR}
================================================================================
EOF
really_continue
PIPS_CONFIGURE_COMMON_FLAGS="${PIPS_CONFIGURE_COMMON_FLAGS} --prefix=${INSTALL_PREFIX}"

info "checking basic requirements"
check_bin "wget svn pkg-config"
WGET=`which wget`
SVN=`which svn`
if [ -n "${DEVEL_MODE}" ]; then
    check_bin "autoreconf automake libtool"
    AUTORECONF="`which autoreconf` --install"
    [ -n "${USERNAME}" ] && SVN="$SVN --username=$USERNAME"
fi

# INSTALL_LOGGER=`mktemp -t ${COMMAND}.logXXXXXXX`
# info "logging everything into ${INSTALL_LOGGER}"

if [ ! -d "${SRC_DIR}" ]; then
    info "${SRC_DIR} do not exist: creating it"
    execute "mkdir -p ${SRC_DIR}"
else
    warning "${SRC_DIR} already exist"
    really_continue "you are about to use directory ${SRC_DIR} as the local working directory where the sources to be configured/compiled will be retrieved"
fi
execute "cd ${SRC_DIR}"

# start with polylib
get_remote_tgz ${TGZ_POLYLIB}
build `basename ${TGZ_POLYLIB} .tar.gz`

for package in newgen linear pips; do
    upper_d=`echo $package | tr "[:lower:]" "[:upper:]"`
    svn_url_varname=SVN_$upper_d
    wget_url_varname=TGZ_$upper_d
    if [ -n "${DEVEL_MODE}" ]; then
        svn_co_or_up $package-${PIPS_AUTOTOOLS_VERSION} ${!svn_url_varname}
    else
        get_remote_tgz ${!wget_url_varname}
    fi
done
build newgen-${PIPS_AUTOTOOLS_VERSION}

PIPS_CONFIGURE_COMMON_FLAGS="${PIPS_CONFIGURE_COMMON_FLAGS} PKG_CONFIG_PATH=${INSTALL_PREFIX}/lib/pkgconfig:$PKG_CONFIG_PATH"
build linear-${PIPS_AUTOTOOLS_VERSION}

PIPS_CONFIGURE_COMMON_FLAGS="${PIPS_CONFIGURE_COMMON_FLAGS} ${PIPS_CONFIG} PATH=${INSTALL_PREFIX}/bin:$PATH "
build $package-${PIPS_AUTOTOOLS_VERSION}

if [ $error -gt 1 ]; then
    warning "Errors occurs"
else
# that's all folks

    if [ -z "${SIMULATION}" ]; then
        cat << EOF
================================================================================
PIPS4U is ready
everything got installed in:    ${INSTALL_PREFIX}
all sources are available from: ${SRC_DIR}

you should consider add the following lines
to your .bashrc or whatever:

  export PATH=${INSTALL_PREFIX}/bin:$PATH
  export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib
  export PKG_CONFIG_PATH=${INSTALL_PREFIX}/lib/pkgconfig
  export PYTHONPATH=${INSTALL_PREFIX}/lib/python2.7/site-packages/pips

================================================================================
EOF
    fi
fi
