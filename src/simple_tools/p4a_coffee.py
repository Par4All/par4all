#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#
import p4a_git
import p4a_opts
import p4a_util
from urlparse import urlparse
import p4a_pack
import p4a_setup
import sys
import os
import optparse
import tempfile

'''
Clone the Par4All public Git repository, build it, package it, and optionally publish it.
'''

#from urllib.parse import urlparse



def add_module_options(parser):
    '''Add options specific to this module to an existing optparse options parser.'''

    group = optparse.OptionGroup(parser, "Coffee Options")

    group.add_option("--here", action = "store_true", default = False,
        help = "Do not clone the repository, assume we are building from the Git tree where the script " + sys.argv[0] + " lies.")

    group.add_option("--git-revision", metavar = "VERSION",
                     action = "store", default = None,
                     help = "By default Par4All is built from the 'p4a' branch when cloning or the current one if the option --here is used. With this option you can precise something else, such as 'p4a-1.0.3' or 'p4a@{yesterday}'")

    group.add_option("--git-repository", metavar = "URL",
                     action = "store", default = None,
                     help = "By default Par4All is cloned from the public repository. But if you have already a local clone, it may be useful to use it instead of transferring on the network or if you want to try with your own developments. For example you can use the directory name of your clone, such as .../par4all. For example, the advantage of this instead using --here is to avoid embedding parasitic files in the package.")

    parser.add_option_group(group)

    p4a_setup.add_module_options(parser)
    p4a_pack.add_module_options(parser)


def main():
    '''The function called when this program is executed by its own'''

    parser = optparse.OptionParser(description = __doc__, usage = "%prog [options]; run %prog --help for options")

    add_module_options(parser)

    p4a_opts.add_common_options(parser)

    (options, args) = parser.parse_args()

    if p4a_opts.process_common_options(options, args):
        # To be able to execute the very same version of this command later:
        exec_path_name = os.path.abspath(sys.argv[0])

        work_dir = ""

        try:
            if options.here:
                #os.system("env")
                p4a_util.info("Using the local working copy", level = 0)
                setup_options = options
                p4a_setup.work(setup_options)

                pack_options = options

                if options.git_revision:
                    p4a_util.run([ "git", "checkout", options.git_revision ])

                # Unset P4A_ROOT environment variable if set, to avoid using
                # the packages from somewhere else:
                os.environ.pop("P4A_ROOT", None)
                p4a_pack.work(pack_options)

            else:
                prev_cwd = os.getcwd()

                if options.git_repository:
                    url = options.git_repository
                    o = urlparse(url)
                    if o.scheme == '' and o.netloc == '':
                        # It does not like as a remote git, it may be a
                        # local directory. Convert it into an absolute
                        # directory so that we can clone it later from any
                        # directory:
                        url = os.path.abspath(url)
                else:
                    # Use the default location:
                    url = "git://git.par4all.org/par4all"

                # Create and jump into a temporary directory:
                work_dir = tempfile.mkdtemp(prefix = "p4a_coffee_")
                os.chdir(work_dir)

                if options.git_revision:
                    git_branch = options.git_revision
                else:
                    # Use the default branch:
                    git_branch = "p4a"

                # Clone the git repository into p4a directory:
                p4a_util.run([ "git", "clone", "-b", git_branch, url, "p4a" ])

                # Jump into the new working copy:
                work_dir_p4a = os.path.join(work_dir, "p4a")
                os.chdir(work_dir_p4a)

                # Compute a revision name that includes the current date:
                suffix = p4a_util.utc_datetime()
                revision = p4a_git.p4a_git(work_dir_p4a).current_revision()
                if revision:
                    suffix += "~" + revision

                # Include the revision in the path so that it appears in
                # debug messages of PIPS and we can trace back a faulty
                # revision.
                work_dir_p4a_version = os.path.join(work_dir, "p4a_" + suffix)
                p4a_util.run([ "mv", "-v", work_dir_p4a, work_dir_p4a_version ])

                os.chdir(prev_cwd)
                ## Make sure child coffee maker will be using the python modules
                ## which come with the git repos which was just cloned:
                #os.environ["PYTHONPATH"] = os.path.join(work_dir_p4a_version, "src/simple_tools")
                # To be able to build old version, use current p4_coffee instead.
                os.environ["PYTHONPATH"] = os.path.join(os.path.dirname(exec_path_name))
                ret = os.system(exec_path_name + " --here --root " + work_dir_p4a_version + " " + " ".join(sys.argv[1:]))
                if ret:
                    raise p4a_util.p4a_error("Child p4a_coffee failed")

                p4a_util.done('''All done. Here is your cup of coffee:

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
                p4a_util.warn("Work directory was " + work_dir + " (not removed)")
            raise

        #~ if not options.work_dir:
        if work_dir:
            p4a_util.rmtree(work_dir)


# If this file is called as a script, execute the main:
if __name__ == "__main__":
    p4a_opts.exec_and_deal_with_errors(main)


# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
