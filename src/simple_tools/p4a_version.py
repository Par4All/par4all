#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#

'''
This module guesses the version of a given file or the current Par4All version.
'''

import sys, os, string
from p4a_util import *
from p4a_git import *

actual_script = change_file_ext(os.path.abspath(os.path.expanduser(__file__)), ".py", if_ext = ".pyc")
script_dir = os.path.split(actual_script)[0]
program_dir = os.path.split(os.path.realpath(os.path.abspath(sys.argv[0])))[0]


def get_version_file_path(dist_dir = None):
    '''Returns the Par4All version file path.'''
    global program_dir
    version_file_name = "VERSION"
    if dist_dir and os.path.isdir(dist_dir):
        version_file_path = os.path.join(dist_dir, version_file_name)
    else:
        version_file_path = os.path.normpath(os.path.join(program_dir, "..", version_file_name))
    debug("p4a_version is " + version_file_path)
    return version_file_path


def guess_file_revision(file_dir = None):
    '''Try to guess a revision/version string for a given file or directory.'''
    
    if file_dir:
        file_dir = os.path.realpath(os.path.abspath(os.path.expanduser(file_dir)))
    else:
        # Default to version for this p4a_version.py script.
        global actual_script
        file_dir = actual_script

    # File is invalid, return gracefully.
    if not os.path.exists(file_dir):
        return "unknown"
    
    revision = ""
    
    # First, attempt to get the Git revision for the whole repos where the file lies, if any.
    try:
        revision = p4a_git(file_dir).current_revision()
    except:
        pass
    if revision:
        return revision
    
    # Try to locate a version file.
    # If file_dir is a directory, look it up underneath this directory.
    # Else, locate the "regular" version file path which should be in the
    # same directory as the p4a_version.py file you are currently reading.
    version_file = ""
    if os.path.isdir(file_dir):
        version_file = get_version_file_path(file_dir)
    else:
        version_file = get_version_file_path()
    if version_file and os.path.exists(version_file):
        # Read the version_file and replace any blanks in it.
        revision = re.sub("\s+", "", slurp(version_file))
    if revision:
        return revision

    # Next, try svnversion.
    #revision = run2([ "svnversion", file ], can_fail = True)[0].strip()
    #if revision:
    #    return revision

    # Finally, make up a version string based on last file modification date/time.
    #~ try:
        #~ revision = file_lastmod(file_dir).strftime("%Y%m%dT%H%M%S") + "~unknown"
    #~ except:
        #~ pass
    
    # Fail gracefully.
    if not revision:
        revision = "unknown"

    return revision


if __name__ == "__main__":
    print(__doc__)
    if len(sys.argv) > 1:
        print("Version file path for " + sys.argv[1] + " is " + get_version_file_path(sys.argv[1]))
    else:
        print("Version file path is " + get_version_file_path())
    print("Default global version is " + guess_file_revision())


# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
