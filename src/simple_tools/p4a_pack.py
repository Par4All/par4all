#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#
import p4a_git
import p4a_opts
import p4a_rc
import p4a_util
import p4a_version
import string
import sys
import os
import re
import optparse
import platform
import tempfile
import shutil
import glob
import time

'''
Par4All packing routines: allows you to create .deb or .tar.gz packages of Par4all.
'''



# Default directory to take current binary installation from.
default_pack_dir                    = "/usr/local/par4all"

# Default installation prefix when installing a .deb or .rpm package on the client machine.
default_install_prefix              = "/usr/local/par4all"

# Package name. Used for naming the packages.
package_name                        = "par4all"

# Settings for --publish. There is currently no command line option to override these defaults.
# Use the $DISTRO and $ARCH placeholders if you want the current distribution and architecture
# to appear in the paths. Use the $DATE placeholder if you wish to have the date in the path.

# The publishing server:
default_publish_host                = "download.par4all.org"
# To debug gently the publishing process instead of defacing the real server:
#default_publish_host                = "127.0.0.1"

default_publish_dir                 = "/srv/www-par4all/download/releases/$DISTRO/$ARCH/$VERSION"

# Use a more hierarchical directory to clean things up:
default_development_publish_dir     = "/srv/www-par4all/download/development/$DISTRO/$ARCH/$YEAR/$MONTH/$DATE"
default_deb_publish_dir             = "/srv/www-par4all/download/apt/$DISTRO/dists/releases/main"
default_deb_development_publish_dir = "/srv/www-par4all/download/apt/$DISTRO/dists/development/main"

# Some useful variables:
actual_script = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(actual_script)

# Where are the .deb settings files? (i.e. the control, postinst, etc. files).
debian_dir = os.path.join(script_dir, "DEBIAN")
# Put temp directories in this array to make sure no temp dir remains after script exits.
temp_dirs = []


def add_module_options(parser):
    '''Add options specific to this module to an existing optparse options parser.'''

    group = optparse.OptionGroup(parser, "Packing Options")

    group.add_option("--pack-dir", metavar = "DIR", default = None,
        help = "Directory where the distribution to package is currently installed. "
        + "Default is to take the root of the Git repository in which this script lies.")

    group.add_option("--deb", action = "store_true", default = False,
        help = "Build a .deb package.")

    group.add_option("--tgz", "-T", action = "store_true", default = False,
        help = "Create a .tar.gz archive.")

    group.add_option("--stgz", action = "store_true", default = False,
        help = "Create a source .tar.gz archive.")

    group.add_option("--arch", metavar = "ARCH", default = None,
        help = "Specify the package architecture manually. By default, the current machine architecture is used.")

    group.add_option("--distro", metavar = "DISTRO", default = None,
        help = "Specify the target distribution manually. By default, the current running Linux distribution is used.")

    group.add_option("--package-version", dest = "version", metavar = "VERSION",
        help = "Specify version for packages.")

    group.add_option("--append-date", "--date", action = "store_true", default = False,
        help = "Automatically append current date & time to version string.")

    group.add_option("--publish", action = "store_true", default = False,
        help = "Publish the produced packages on the server.")

    group.add_option("--publish-only", action = "append", metavar = "FILE", default = [],
        help = "Publish only a given file (.deb, tgz, stgz or even whatever for testing) without rebuilding it. Several files are allowed by using this option several times.")

    group.add_option("--retry-publish", action = "store_true",  default = False,
        help = "Retry to publish only files (.deb and/or tgz and/or stgz) from the local directory without rebuilding them. Be sure from what is in your local directory! To be used with p4a_pack and not with p4a_coffee")

    group.add_option("--release", dest = "development", action = "store_false", default = True,
        help = "When publishing, put the packages in release directories instead of development ones.")

    group.add_option("--install-prefix", metavar = "DIR", default = None,
        help = "Specify the installation prefix. Default is /usr/local/par4all.")

    group.add_option("--keep-temp", "-k", action = "store_true", default = False,
        help = "Do not remove temporary directories after script execution.")

    group.add_option("--pack-output-dir", metavar = "DIR", default = None,
        help = "Directory where package files will be put (locally). Any existing package with the same name (exact same revision) "
        + "will be overwritten without prompt. Defaults to current working directory: " + os.getcwd())

    parser.add_option_group(group)


def create_dist(pack_dir, install_prefix, version, gitrev):
    '''Creates a temporary directory and copy the whole installation directory (pack_dir), under the prefix designated by install_prefix.
    Also writes updated rc files (shell scripts for setting the environment), and a version file.
    Returns a list with the temporary directory created and the full path of the temporary directory and the appended install_prefix.'''
    global temp_dirs
    temp_dir = tempfile.mkdtemp(prefix = "p4a_pack_")
    temp_dirs.append(temp_dir)
    p4a_util.debug("Temp dir is " + temp_dir)
    temp_dir_with_prefix = os.path.join(temp_dir, install_prefix)
    os.makedirs(os.path.split(temp_dir_with_prefix)[0])
    p4a_util.info("Copying " + pack_dir + " to " + temp_dir_with_prefix)
    p4a_util.run([ "cp", "-av", pack_dir + "/", temp_dir_with_prefix ])
    abs_prefix = "/" + install_prefix

    p4a_rc.p4a_write_rc(os.path.join(temp_dir_with_prefix, "etc"),
        dict(
            root = abs_prefix,
            dist = abs_prefix,
            accel = "share/p4a_accel",
            scmp = "share/p4a_scmp",
            fortran = "gfortran" # ???
        )
    )

    p4a_version.write_VERSION(temp_dir_with_prefix, version)
    p4a_version.write_GITREV(temp_dir_with_prefix, gitrev)

    return [ temp_dir, temp_dir_with_prefix ]


def create_deb(pack_dir, install_prefix, version, gitrev, distro, arch, keep_temp = False):
    '''Creates a .deb package. Simply adds the necessary DEBIAN directory in the temporary directory
    and substitute some values in files in this DEBIAN directory. No modification of the
    distribution is made.'''
    global debian_dir, package_name
    (temp_dir, temp_dir_with_prefix) = create_dist(pack_dir, install_prefix, version, gitrev)
    temp_debian_dir = os.path.join(temp_dir, "DEBIAN")
    p4a_util.info("Copying " + debian_dir + " to " + temp_debian_dir)
    shutil.copytree(debian_dir, temp_debian_dir)
    control_file = os.path.join(temp_debian_dir, "control.tpl")
    (revision, versiond) = p4a_version.make_full_revision(custom_version = version, custom_gitrev = gitrev)
    subs_map = dict(NAME = package_name, VERSION = revision, ARCH = arch, DIST = "/" + install_prefix)
    p4a_util.info("Adjusting values in " + control_file)
    p4a_util.subs_template_file(control_file, subs_map)
    postinst_file = os.path.join(temp_debian_dir, "postinst.tpl")
    p4a_util.info("Adjusting values in " + postinst_file)
    p4a_util.subs_template_file(postinst_file, subs_map)
    package_file_name = package_name + "-" + revision + "_" + arch + ".deb"
    package_file = os.path.abspath(package_file_name)
    if os.path.exists(package_file):
        p4a_util.warn("Removing existing " + package_file)
        os.remove(package_file)
    p4a_util.run([ "fakeroot", "dpkg-deb", "--build", temp_dir, package_file_name ])
    if os.path.exists(package_file_name):
        p4a_util.done("Created " + package_file_name)
    else:
        p4a_util.warn(package_file_name + " file not created!?")
    if keep_temp:
        p4a_util.warn("Temporary directory was " + temp_dir)
    else:
        p4a_util.rmtree(temp_dir, can_fail = 1)
    return package_file_name


def create_tgz(pack_dir, install_prefix, version, gitrev, distro, arch, keep_temp = False):
    '''Creates a simple .tar.gz package.'''
    global package_name
    (temp_dir, temp_dir_with_prefix) = create_dist(pack_dir, install_prefix, version, gitrev)
    if not arch:
        arch = p4a_util.get_machine_arch()
    (revision, versiond) = p4a_version.make_full_revision(custom_version = version, custom_gitrev = gitrev)
    package_file_name = package_name + "-" + revision + "_" + arch + ".tar.gz"
    package_file = os.path.abspath(package_file_name)
    if os.path.exists(package_file):
        p4a_util.warn("Removing existing " + package_file)
        os.remove(package_file)
    tar_dir = os.path.split(temp_dir_with_prefix)[0]
    new_temp_dir_with_prefix = os.path.join(tar_dir, package_name + "-" + versiond)
    shutil.move(temp_dir_with_prefix, new_temp_dir_with_prefix)
    tar_cmd = " ".join([ "tar", "czvf", package_file_name, "-C", tar_dir, "." ])
    sh_cmd = '"chown -R root:root ' + tar_dir + " && find " + tar_dir + " -type d -exec chmod 755 '{}' \\; && " + tar_cmd + '"'
    p4a_util.run([ "fakeroot", "sh", "-c", sh_cmd])
    if os.path.exists(package_file_name):
        p4a_util.done("Created " + package_file_name)
    else:
        p4a_util.warn(package_file_name + " file not created!?")
    if keep_temp:
        p4a_util.warn("Temporary directory was " + temp_dir)
    else:
        p4a_util.rmtree(temp_dir, can_fail = True)
    return package_file

def create_stgz(pack_dir, install_prefix, version, gitrev, keep_temp = False):
    global package_name, temp_dirs
    (revision, versiond) = p4a_version.make_full_revision(custom_version = version, custom_gitrev = gitrev)
    package_full_name = package_name + "-" + revision + "_src"
    package_file_name = package_full_name + ".tar.gz"
    package_file = os.path.abspath(package_file_name)
    package_file_tar = p4a_util.change_file_ext(package_file, "")
    if os.path.exists(package_file):
        p4a_util.warn("Removing existing " + package_file)
        os.remove(package_file)
    git = p4a_git.p4a_git(script_dir)
    current_branch = git.current_branch()
    #if current_branch != "p4a":
    #    p4a_util.die("Not on branch p4a (" + current_branch + "), cannot create a source package")
    prefix = package_name + "-" + versiond + "_src"
    git.archive(package_file_tar, prefix = prefix + "/")
    temp_dir = tempfile.mkdtemp(prefix = "p4a_pack_version_")
    p4a_util.debug("Temp dir is " + temp_dir)
    temp_dirs.append(temp_dir)
    prev_cwd = os.getcwd()
    os.chdir(temp_dir)
    os.makedirs(os.path.join(temp_dir, prefix))
    relative_version_file = p4a_version.write_VERSION(prefix, version)
    relative_gitrev_file = p4a_version.write_GITREV(prefix, gitrev)
    tar_cmd = " ".join([ "tar", "uvf", package_file_tar, relative_version_file ])
    sh_cmd = '"chown root:root ' + relative_version_file + " && " + tar_cmd + '"'
    p4a_util.run([ "fakeroot", "sh", "-c", sh_cmd])
    tar_cmd = " ".join([ "tar", "uvf", package_file_tar, relative_gitrev_file ])
    sh_cmd = '"chown root:root ' + relative_gitrev_file + " && " + tar_cmd + '"'
    p4a_util.run([ "fakeroot", "sh", "-c", sh_cmd])
    os.chdir(prev_cwd)
    p4a_util.rmtree(temp_dir)
    p4a_util.run([ "gzip", "-9", package_file_tar ])
    if os.path.exists(package_file_name):
        p4a_util.done("Created " + package_file_name)
    return package_file


def publish_deb(file, host, repos_dir, orig_dir, arch):
    """Publish a Debian package on a serve by linking it in repos_dir to
    orig_dir where file is supposed to already be here"""
    #~ arch = p4a_util.change_file_ext(file, "").split("_")[-1]
    p4a_util.info("\n\nPublishing " + file + " in the deb repository (" + arch + ") " + repos_dir )
    local_file_name = os.path.basename(file)
    repos_arch_dir = os.path.join(repos_dir, "binary-" + arch)
    # Create directory hierarchy:
    p4a_util.run([ "ssh", host, "mkdir -p " + repos_arch_dir ])
    p4a_util.warn("Removing existing .deb file for arch " + arch + " (" + repos_arch_dir + ")")
    p4a_util.run([ "ssh", host, "rm -fv " + os.path.join(repos_arch_dir, "*.deb") ])
    p4a_util.warn("Link " + local_file_name + " on " + host + " (in " + repos_arch_dir + " to " + orig_dir + ")")
    # Point to a relative name to help moving the repository on the server
    # later. Absolute link names would be a nightmare else...
    relative_dest = os.path.relpath(os.path.join(orig_dir, local_file_name),
                                    start = repos_arch_dir)
    p4a_util.run([ "ssh", host, "ln -s " + relative_dest +
                   " " + os.path.join(repos_arch_dir, local_file_name) ])
    p4a_util.warn("Please allow max. 5 minutes for deb repository indexes to get updated by cron")
    p4a_util.warn("Alternatively, you can run /srv/update-par4all.sh on " + host + " as root")


def make_publish_dirs_from_template(publish_dir, distro, arch, version,
                                    deb_publish_dir, deb_distro, deb_arch):
    "Build the publish dir with its package dir at the same time for atomicity"
    # We have to compute the time only once, because time is evolving! :-/
    # Just imagine this program run at midnight for example and the debug
    # joy we would have...
    time_once = time.gmtime()
    date = time.strftime("%Y-%m-%d", time_once)
    year = time.strftime("%Y", time_once)
    month = time.strftime("%m", time_once)
    # Substitute placeholders such as $DISTRO, $DATE, etc.
    return (string.Template(publish_dir).substitute(DISTRO = distro,
                                                    ARCH = arch,
                                                    YEAR = year,
                                                    MONTH = month,
                                                    DATE = date,
                                                    VERSION = version),
            string.Template(deb_publish_dir).substitute(DISTRO = distro,
                                                        ARCH = arch,
                                                        YEAR = year,
                                                        MONTH = month,
                                                        DATE = date,
                                                        VERSION = version))


def publish_files(files, distro, deb_distro, arch, deb_arch, version,
                  development = False):
    "Publish a list of files on the server"
    global default_publish_host
    global default_publish_dir, default_development_publish_dir
    global default_deb_publish_dir, default_deb_development_publish_dir

    publish_dir = None
    deb_publish_dir = None
    if development:
        publish_dir = default_development_publish_dir
        deb_publish_dir = default_deb_development_publish_dir
    else:
        publish_dir = default_publish_dir
        deb_publish_dir = default_deb_publish_dir

    (publish_dir, deb_publish_dir) = make_publish_dirs_from_template(publish_dir, distro, arch, version, deb_publish_dir, deb_distro, deb_arch)

    for file in files:
        file = os.path.abspath(os.path.expanduser(file))
        if not os.path.exists(file):
            p4a_util.die("Invalid file: " + file)
        p4a_util.warn("\n\nCopying " + file + " in " + publish_dir + " on " + default_publish_host)
        p4a_util.warn("If something goes wrong, try running /srv/fixacl-par4all.sh to fix permissions")
        p4a_util.warn("If something goes wrong, try creating directories or publishing the file manually")
        # Use -p to create any intermediate dir hierarchy without failing
        # on already existence:
        p4a_util.run([ "ssh", default_publish_host, "mkdir -p " + publish_dir ])
        # Survive to rsync error #30 "Timeout in data send/receive", by
        # retrying 10 times after rsync has waited for 60 seconds:
        p4a_util.run([ "rsync --partial --sparse --timeout=60", file, default_publish_host + ":" + publish_dir ], error_code = 30, retry = 10,
            msg= " (timeout).\n" + "Retry to publish using p4a_pack.py with option --retry-publish")
        ext = p4a_util.get_file_ext(file)
        if ext == ".deb":
            publish_deb(file, default_publish_host, deb_publish_dir, publish_dir, deb_arch)


def work(options, args = []):
    '''Do the real work. The main goal of this function is to be able to
    call p4a_setup from another tool (p4a_coffee.py) that has already
    parsed the arguments and options of the command.
    '''

    # Determine architecture for binary packages (and special arch name for debs).
    deb_arch = arch = options.arch
    if not arch:
        arch = platform.machine()
    if arch == "x86_64":
        deb_arch = "amd64"
    elif re.match("i\d86", arch):
        deb_arch = "i386"
    p4a_util.debug("arch=" + arch)
    p4a_util.debug("deb_arch=" + deb_arch)

    # Determine current running distro unless provided.
    distro = options.distro
    if not distro:
        distro = p4a_util.get_distro()
    if options.deb:
        if distro not in [ "ubuntu", "debian" ]:
            p4a_util.die("Invalid target distro for building .debs: " + distro)
    p4a_util.debug("distro=" + distro)

    pack_dir = options.pack_dir
    if pack_dir:
        pack_dir = os.path.realpath(os.path.abspath(os.path.expanduser(pack_dir)))
        p4a_util.warn("Par4All installation is to be found in " + pack_dir + " (--pack-dir)")
    else:
        pack_dir = default_pack_dir
        p4a_util.warn("Assuming Par4All is currently installed in " + pack_dir + " (default; use --pack-dir to override)")
    if not os.path.isdir(pack_dir):
        p4a_util.die("Directory does not exist: " + pack_dir)

    # By default use the version computed by p4a_version.VERSION:
    version = None
    if options.version:
        version = options.version
        p4a_util.info("Version: " + version + " (--package-version)")
    else:
        version = p4a_version.VERSION(pack_dir)
        p4a_util.info("Version: " + version + " (override with --package-version)")

    if options.append_date:
        dt = p4a_util.utc_datetime()
        version += "~" + dt

    gitrev = p4a_version.GITREV(pack_dir)
    p4a_util.info("Git revision: " + gitrev)

    if len(args):
        if not options.publish:
            p4a_util.die("You specified files on command line but did not specify --publish")
        publish_files(args, distro, distro, arch, deb_arch,
                      version, options.development)
        return

    if options.development and not options.append_date:
        options.append_date = True

    if len(options.publish_only):
        (full_version, version) = p4a_version.make_full_revision(custom_version = version)
        publish_files(options.publish_only, distro, distro, arch, deb_arch,
                      version, options.development)
        return

    if (not options.deb #and not options.sdeb
        and not options.tgz and not options.stgz):
        p4a_util.warn("--deb and/or --tgz and/or --stgz not specified, assuming --deb --tgz --stgz")
        options.deb = True
        options.tgz = True
        options.stgz = True

	if options.retry_publish:
		file_to_publish=[]
		if options.deb:
			if options.pack_output_dir:
				file_to_publish+=glob.glob(options.pack_output_dir+"/*.deb")
			else:
				file_to_publish+=glob.glob("*.deb")
		if options.tgz or options.stgz:
			if options.pack_output_dir:
				file_to_publish+=glob.glob(options.pack_output_dir+"/*.gz")
			else:
				file_to_publish+=glob.glob("*.gz")
		publish_files(file_to_publish, distro, distro, arch, deb_arch,
                      version, options.development)
		return

    prefix = options.install_prefix
    if prefix:
        p4a_util.warn("Installation prefix: " + prefix + " (--install-prefix)")
    else:
        prefix = default_install_prefix
        p4a_util.warn("Installation prefix: " + prefix + " (default; override with --install-prefix)")

    # Strip forward slash:
    if prefix[0] == "/" and len(prefix):
        prefix = prefix[1:]

    output_dir = options.pack_output_dir
    if output_dir:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        p4a_util.warn("Packages will be put in " + output_dir + " (--pack-output-dir)")
        if output_dir == os.getcwd():
            output_dir = None
    else:
        p4a_util.warn("Packages will be put in " + os.getcwd() + " (default; override with --pack-output-dir)")

    # Check that fakeroot is available.
    if not p4a_util.which("fakeroot"):
        p4a_util.die("fakeroot not found, please install it (sudo aptitude install fakeroot)")


    global temp_dirs
    output_files = []
    try:
        if options.deb:
			#remove previous *.deb
			p4a_util.run(["rm -fv *.deb"])
			output_files.append(create_deb(pack_dir = pack_dir, install_prefix = prefix,
				version = version, gitrev = gitrev, distro = distro, arch = deb_arch,
                keep_temp = options.keep_temp))

        if options.tgz:
			#remove previous *.gz
			p4a_util.run(["rm -fv *.gz"])
			output_files.append(create_tgz(pack_dir = pack_dir, install_prefix = prefix,
                version = version, gitrev = gitrev, distro = distro, arch = arch,
                keep_temp = options.keep_temp))

        if options.stgz:
			#remove previous *.gz
			p4a_util.run(["rm -fv *src.tar.gz"])
			output_files.append(create_stgz(pack_dir = pack_dir, install_prefix = prefix,
                version = version, gitrev = gitrev,
                keep_temp = options.keep_temp))

        if options.publish:
            publish_files(output_files, distro, distro, arch, deb_arch,
                          version, options.development)

        if output_dir:
            for file in output_files:
                dest_file = os.path.join(output_dir, os.path.split(file)[1])
                if dest_file == file:
                    continue
                if os.path.exists(dest_file):
                    p4a_util.warn("Removing existing " + dest_file)
                    os.remove(dest_file)
                p4a_util.run([ "mv", "-v", file, dest_file ])
    except:
        for dir in temp_dirs:
            if os.path.isdir(dir):
                p4a_util.warn("NOT Removing " + dir)
        raise

    for dir in temp_dirs:
        if os.path.isdir(dir):
            if options.keep_temp:
                p4a_util.warn("NOT Removing " + dir + " (--keep-temp)")
            else:
                p4a_util.rmtree(dir, can_fail = True)


def main():
    '''The function called when this program is executed by its own'''

    parser = optparse.OptionParser(description = __doc__,
        usage = "%prog --deb|--tgz|-stgz [other options] [optional additional files to publish]")

    add_module_options(parser)

    p4a_opts.add_common_options(parser)

    (options, args) = parser.parse_args()

    if p4a_opts.process_common_options(options, args):
        work(options, args)


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
