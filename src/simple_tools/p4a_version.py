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

actual_script = change_file_ext(os.path.realpath(os.path.abspath(__file__)), ".py", if_ext = ".pyc")
script_dir = os.path.split(actual_script)[0]
program_dir = os.path.split(os.path.realpath(os.path.abspath(sys.argv[0])))[0]


def VERSION_file_path(dist_dir = None):
    '''Returns the Par4All VERSION file path.'''
    global program_dir
    version_file_name = "VERSION"
    if dist_dir and os.path.isdir(dist_dir):
        # Assume there is a VERSION file at root of dist_dir:
        version_file_path = os.path.join(dist_dir, version_file_name)
    else:
        d = ""
        try:
            # Assume first that the VERSION file is located at root of Git repos
            # in which the calling program lies.
            d = p4a_git(program_dir).dir()
        except:
            # Assume the VERSION file is one level above the program directory
            # (if program is /usr/local/par4all/bin/p4a, the VERSION file will be
            # /usr/local/par4all/VERSION).
            d = os.path.join(program_dir, "..")
        version_file_path = os.path.normpath(os.path.join(d, version_file_name))
    debug(version_file_name + " file path is " + version_file_path)
    return version_file_path

def GITREV_file_path(dist_dir = None):
    '''Returns the Par4All GITREV file path.'''
    global program_dir
    version_file_name = "GITREV"
    if dist_dir and os.path.isdir(dist_dir):
        version_file_path = os.path.join(dist_dir, version_file_name)
    else:
        d = ""
        try:
            d = p4a_git(program_dir).dir()
        except:
            d = os.path.join(program_dir, "..")
        version_file_path = os.path.normpath(os.path.join(d, version_file_name))
    debug(version_file_name + " file path is " + version_file_path)
    return version_file_path


def VERSION(file_dir = None):
    version = None
    version_file = VERSION_file_path(file_dir)
    if os.path.exists(version_file):
        version = read_file(version_file)
        version = re.sub("\s+", "", version)
        debug("Contents of " + version_file + ": " + version)
    else:
        version = "devel"
    debug("VERSION(" + repr(file_dir) + ") = " + version)
    return version

def GITREV(file_dir = None, test_dirty = True, include_tag = False):
    gitrev = ""
    gitrev_file = GITREV_file_path(file_dir)
    if os.path.exists(gitrev_file):
        gitrev = read_file(gitrev_file)
        gitrev = re.sub("\s+", "", gitrev)
        debug("Contents of " + gitrev_file + ": " + gitrev)
    else:
        if not file_dir:
            file_dir = program_dir
        try:
            gitrev = p4a_git(file_dir).current_revision(
                test_dirty = test_dirty, include_tag = include_tag)
        except:
            pass
    debug("GITREV(" + repr(file_dir) + ") = " + gitrev)
    return gitrev


def write_VERSION(dir, version):
    write_file(os.path.join(dir, "VERSION"), version)

def write_GITREV(dir, gitrev):
    write_file(os.path.join(dir, "GITREV"), gitrev)


def make_full_revision(file_dir = None, custom_version = "", custom_gitrev = ""):

    '''Make up a precise revision/version string for a given file or directory,
    or from passed version and git revision strings.'''

    version = custom_version
    if not version:
        version = VERSION(file_dir)

    gitrev = custom_gitrev
    if not gitrev:
        gitrev = GITREV(file_dir)

    if gitrev:
        version += "~" + gitrev
    version = version.replace("-", "~")
    max_len = 128
    if len(version) > max_len:
        warn("Version is too long in VERSION file")
        version = version[0:max_len]
    versionm = []
    versiond = ""
    for v in version.split("~"):
        v = re.sub("[^\w_\.]", "", v)
        if not v:
            continue
        if re.match(r"^\d\.\d(\.\d)?$", v):
            versiond = v
            debug("Numeric version: " + v)
        else:
            versionm.append(v)
    if versiond:
        version = versiond + "-" + "~".join(versionm)
    else:
        warn("Numeric version not found, please add/fix VERSION file, "
            + "custom version specification, or Git tag")
        version = "0.0-" + version

    #if append_date:
    #    version += "~" + utc_datetime()

    debug("Version string for " + repr(file_dir) + ": " + version)

    return version


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
