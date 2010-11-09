#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#

'''
Clone the Par4All public Git repository, build it, package it, and optionally publish it.
'''

import string, sys, os, re, optparse, tempfile
from p4a_util import *
from p4a_builder import *
from p4a_git import *
from p4a_version import *

import p4a_setup
import p4a_pack


def add_module_options(parser):
    '''Add options specific to this module to an existing optparse options parser.'''

    group = optparse.OptionGroup(parser, "Coffee Options")

    #~ group.add_option("--work-dir", metavar = "DIR", default = None,
        #~ help = "Directory where the Git repository will be cloned and where the build will happen. "
        #~ + "By default, it will pick a temporary directory and remove it afterwards unless an error occurred.")

    group.add_option("--here", action = "store_true", default = False,
        help = "Do not clone the repository, assume we are building from the Git tree where the script " + sys.argv[0] + " lies.")

    group.add_option("--git-revision", metavar = "VERSION",
                     action = "store", default = None,
                     help = "By default Par4All is built from the 'p4a' branch when cloning or the current one if the option --here is used. With this option you can precise something else, such as 'p4a-1.0.3' or 'p4a@{yesterday}'")

    parser.add_option_group(group)

    p4a_setup.add_module_options(parser)
    p4a_pack.add_module_options(parser)


def main(options, args = []):

    # To be able to execute the very same version of this command later:
    exec_path_name = os.path.abspath(sys.argv[0])

    #~ work_dir = ""
    #~ if options.work_dir:
        #~ work_dir = os.path.abspath(os.path.expanduser(options.work_dir))
    #~ else:
    work_dir = ""

    #~ if not os.path.isdir(work_dir):
    #~ os.makedirs(work_dir)

    try:
        if options.here:
            #os.system("env")

            setup_options = options
            #~ options.packages_dir = os.path.join(work_dir_p4a_version, "packages")
            #~ warn("Forcing --packages-dir=" + options.packages_dir)
            p4a_setup.main(setup_options)

            pack_options = options

            if options.git_revision:
                run([ "git", "checkout", options.git_revision ])

            #~ options.pack_dir = work_dir_p4a_version
            #~ warn("Forcing --pack-dir=" + options.pack_dir)

            # Unset P4A_ROOT environment variable if set, to avoid using
            # the packages from somewhere else:
            os.environ.pop("P4A_ROOT", None)
            p4a_pack.main(pack_options)

        else:
            work_dir = tempfile.mkdtemp(prefix = "p4a_coffee_")

            prev_cwd = os.getcwd()
            os.chdir(work_dir)

            work_dir_p4a = os.path.join(work_dir, "p4a")
            #~ if os.path.isdir(work_dir_p4a):
                #~ warn("p4a directory already exists (" + work_dir_p4a + "), will not clone the repository again")
                #~ os.chdir(work_dir_p4a)
                #~ run([ "git", "checkout", "-b", "p4a", "remotes/origin/p4a" ])
                #~ run([ "git", "pull" ])
            #~ else:
            run([ "git", "clone", "git://git.hpc-project.com/par4all", "p4a" ])
            os.chdir(work_dir_p4a)

            if not options.git_revision:
                # The revision by default:
                options.git_revision = 'p4a'
            run([ "git", "checkout", options.git_revision ])

            suffix = utc_datetime()
            revision = p4a_git(work_dir_p4a).current_revision()
            if revision:
                suffix += "~" + revision

            # Include the revision in the path so that it appears in debug messages
            # of PIPS and we can trace back a faulty revision.
            work_dir_p4a_version = os.path.join(work_dir, "p4a_" + suffix)
            #~ run([ "rm", "-fv", work_dir_p4a_version ])
            #~ run([ "ln", "-sv", work_dir_p4a, work_dir_p4a_version ])
            run([ "mv", "-v", work_dir_p4a, work_dir_p4a_version ])

            os.chdir(prev_cwd)
            ## Make sure child coffee maker will be using the python modules
            ## which come with the git repos which was just cloned:
            #os.environ["PYTHONPATH"] = os.path.join(work_dir_p4a_version, "src/simple_tools")
            # To be able to build old version, use current p4_coffee instead.
            os.environ["PYTHONPATH"] = os.path.join(os.path.dirname(exec_path_name))
            ret = os.system(exec_path_name + " --here --root " + work_dir_p4a_version + " " + " ".join(sys.argv[1:]))
            if ret:
                raise p4a_error("Child p4a_coffee failed")

            done('''All done. Here is your cup of coffee:

         (    (     (     (
          )    )     )     )
        _(____(_____(_____(___
        |                     |
        |                     |______
        |                      ___   |
        |                     |   |  |
        |    P a r 4 A l l    |___|  |
        |                     ______/
        |                     |
         \___________________/

''')

    except:
        #~ if not options.work_dir:
        if work_dir:
            warn("Work directory was " + work_dir + " (not removed)")
        raise

    #~ if not options.work_dir:
    if work_dir:
        rmtree(work_dir)


# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
