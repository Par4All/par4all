#!/usr/bin/python
#
# $Id$
#
# Copyright 1989-2009 MINES ParisTech
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

import subprocess
from sys import argv
import string

# globals
__tpips__="tpips"
__project__="PROJECT"
__process__=subprocess.Popen(args=__tpips__, stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
__sources__=[]


# helpers
def run(*commands):
    __process__.stdin.write(string.join(commands)+"\n")
def activate(*command):
    run("activate" , string.join(command) )
def property(prop,val):
    run("setproperty",prop,val)

def delete():
    run("delete", __project__ )

def create( *files):
    def check_file(file,ext): return file[-1] != ext
    args=""
    for file in files:
        args+= " " + args
    run("create", __project__, args )
    if check_file(files[0],"f") and check_file(files[0],"F") :
        activate("C_PARSER")
        property("PRETTYPRINT_C_CODE", "TRUE")
        property("PRETTYPRINT_STATEMENT_NUMBER", "FALSE")

# common init
delete()
create(argv)
