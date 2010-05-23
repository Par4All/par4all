#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#

'''
Guess Par4All Version
'''

import sys, os, string
from p4a_util import *
from git import *

actual_script = os.path.abspath(os.path.realpath(os.path.expanduser(__file__)))
script_dir = os.path.split(actual_script)[0]

def guess_file_revision(file):
    '''Make up a revision string for the given file. If it is in a Git repository, use the Git revision, otherwise try with svnversion.
    If file is not versioned, fall back on last modification date.'''

    revision = git(file).current_revision(file)
    
    if not revision:
        revision = run2([ "svnversion", file ], can_fail = True)[0].strip()
        
    if not revision:
        version_file = os.path.join(script_dir, "p4a_version")
        if os.path.exists(version_file):
            revision = re.sub("\s+", "", slurp(version_file))
    
    if not revision:
        try:
            revision = file_lastmod(os.path.abspath(os.path.realpath(os.path.expanduser(file)))).strftime("%Y%m%dT%H%M%S") + "~unknown"
        except:
            revision = "unknown"
    return revision

if __name__ == "__main__":
    print(__doc__)
    print("This module is not directly executable")

# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
