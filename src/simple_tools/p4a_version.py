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
from p4a_git import *

actual_script = os.path.abspath(os.path.realpath(os.path.expanduser(__file__)))
script_dir = os.path.split(actual_script)[0]

def get_version_file_path(dist_dir = None):
    '''Returns the Par4All version file path, if any.'''
    if dist_dir and os.path.isdir(dist_dir):
        return os.path.join(dist_dir, "lib", "p4a_version")
    elif "P4A_DIST" in os.environ and os.path.isdir(os.environ["P4A_DIST"]):
        return os.path.join(os.environ["P4A_DIST"], "lib", "p4a_version")
    else:
        global script_dir
        return os.path.join(script_dir, "p4a_version")

def guess_file_revision(file):
    '''Try to guess a revision/version string for a given file.
    Try locating the Par4All version file first, then try the Git revision,
    then svnversion, and finally fall back on the last modification date
    of the given file.'''
    
    revision = ""
    
    version_file = get_version_file_path()
    if os.path.exists(version_file):
        revision = re.sub("\s+", "", slurp(version_file))

    if not revision:
        try:
            revision = p4a_git(file).current_revision(file)
        except:
            pass

    if not revision:
        revision = run2([ "svnversion", file ], can_fail = True)[0].strip()

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
