#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#

'''
Par4All packing routines: allows you to create .deb or .tar.gz packages of Par4all.
'''

import string, sys, os, re, optparse, tempfile, shutil, time
from p4a_util import *
from p4a_rc import *
from p4a_git import *
from p4a_version import *


# Default directory to take current binary installation from.
default_pack_dir                = "/usr/local/par4all"

# Default installation prefix when installing a .deb or .rpm package on the client machine.
default_install_prefix          = "/usr/local/par4all"

# Package name. Used for naming the packages.
package_name                    = "par4all"

# If no version is specified, assume this is the version:
default_version                 = "1.0"

# Settings for --publish. There is currently no option to override these defaults.
default_publish_host            = "download.par4all.org"
default_publish_dir             = "/srv/www-par4all/download/releases"
default_nightly_publish_dir     = "/srv/www-par4all/download/nightly"
default_deb_publish_dir         = "/srv/www-par4all/download/ubuntu/dists/releases/main"
default_deb_nightly_publish_dir = "/srv/www-par4all/download/ubuntu/dists/nightly/main"

# Where are the .deb settings files? (i.e. the control, postinst, etc. files).
debian_dir = os.path.join(script_dir, "DEBIAN")

# Some useful variables:
actual_script = change_file_ext(os.path.realpath(os.path.abspath(__file__)), ".py", if_ext = ".pyc")
script_dir = os.path.split(actual_script)[0]

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

    #~ group.add_option("--sdeb", action = "store_true", default = False,
        #~ help = "Create a source .deb package.")

    #~ group.add_option("--rpm", "-R", action = "store_true", default = False,
        #~ help = "Build a .rpm package.")

    #~ group.add_option("--srpm", action = "store_true", default = False,
        #~ help = "Build a source .rpm package.")

    group.add_option("--tgz", "-T", action = "store_true", default = False,
        help = "Create a .tar.gz archive.")

    group.add_option("--stgz", action = "store_true", default = False,
        help = "Create a source .tar.gz archive.")

    group.add_option("--arch", metavar = "ARCH", default = None,
        help = "Specify the package architecture manually. By default, the current machine architecture is used.")

    group.add_option("--version", "--revision", metavar = "REVISION",
        help = "Specify package version. Current Git revision will be automatically appended.")

    group.add_option("--append-date", action = "store_true", default = False,
        help = "Automatically append date to version string.")

    group.add_option("--publish", action = "store_true", default = False,
        help = "Create a .tar.gz archive.")

    group.add_option("--nightly", action = "store_true", default = False,
        help = "When publishing, store files in the 'nightly' directory (i.e. for nightly builds). Implies --append-date.")

    group.add_option("--install-prefix", metavar = "DIR", default = None,
        help = "Specify the installation prefix. Default is /usr/local/par4all.")

    group.add_option("--keep-temp", "-k", action = "store_true", default = False,
        help = "Do not remove temporary directories after script execution.")

    group.add_option("--pack-output-dir", metavar = "DIR", default = None,
        help = "Directory where package files will be put (locally). Any existing package with the same name (exact same revision) "
        + "will be overwritten without prompt. Defaults to current working directory: " + os.getcwd())

    parser.add_option_group(group)


def create_dist(pack_dir, install_prefix, revision):
    '''Creates a temporary directory and copy the whole installation directory (pack_dir), under the prefix designated by install_prefix.
    Also writes updated rc files (shell scripts for setting the environment), and a version file.
    Returns a list with the temporary directory created and the full path of the temporary directory and the appended install_prefix.'''
    global temp_dirs
    temp_dir = tempfile.mkdtemp(prefix = "p4a_pack_")
    temp_dirs.append(temp_dir)
    debug("Temp dir is " + temp_dir)
    temp_dir_with_prefix = os.path.join(temp_dir, install_prefix)
    os.makedirs(os.path.split(temp_dir_with_prefix)[0])
    info("Copying " + pack_dir + " to " + temp_dir_with_prefix)
    #~ shutil.copytree(pack_dir, temp_dir_with_prefix)
    run([ "cp", "-av", pack_dir + "/", temp_dir_with_prefix ])
    abs_prefix = "/" + install_prefix
    # XXX: gfortran or not??
    p4a_write_rc(os.path.join(temp_dir_with_prefix, "etc"), 
        dict(root = abs_prefix, dist = abs_prefix, accel = os.path.join(abs_prefix, "share/p4a_accel"), fortran = "gfortran"))
    write_file(get_version_file_path(temp_dir_with_prefix), revision)
    return [ temp_dir, temp_dir_with_prefix ]


def create_deb(pack_dir, install_prefix, revision, keep_temp = False, arch = None):
    '''Creates a .deb package. Simply adds the necessary DEBIAN directory in the temporary directory
    and substitute some values in files in this DEBIAN directory. No modification of the
    distribution is made.'''
    global debian_dir, package_name
    (temp_dir, temp_dir_with_prefix) = create_dist(pack_dir, install_prefix, revision)
    temp_debian_dir = os.path.join(temp_dir, "DEBIAN")
    info("Copying " + debian_dir + " to " + temp_debian_dir)
    shutil.copytree(debian_dir, temp_debian_dir)
    control_file = os.path.join(temp_debian_dir, "control.tpl")
    if not arch:
        arch = get_machine_arch()
        if arch == "x86_64":
            arch = "amd64"
        elif re.match("i\d86", arch):
            arch = "i386"
        else:
            die("Unknown architecture " + arch + ", use --arch")
    subs_map = dict(NAME = package_name, VERSION = revision, ARCH = arch, DIST = "/" + install_prefix)
    info("Adjusting values in " + control_file)
    subs_template_file(control_file, subs_map)
    postinst_file = os.path.join(temp_debian_dir, "postinst.tpl")
    info("Adjusting values in " + postinst_file)
    subs_template_file(postinst_file, subs_map)
    package_file_name = "_".join([ package_name, revision, arch ]) + ".deb"
    package_file = os.path.abspath(package_file_name)
    if os.path.exists(package_file):
        warn("Removing existing " + package_file)
        os.remove(package_file)
    run([ "fakeroot", "dpkg-deb", "--build", temp_dir, package_file_name ])
    if os.path.exists(package_file_name):
        done("Created " + package_file_name)
    else:
        warn(package_file_name + " file not created!?")
    if keep_temp:
        warn("Temporary directory was " + temp_dir)
    else:
        rmtree(temp_dir, can_fail = 1)
    return package_file_name


def publish_deb(file, host, remote_dir):
    arch = change_file_ext(file, "").split("_")[-1]
    info("Publishing " + file + " (" + arch + ")")
    warn("Removing existing .deb file for arch " + arch)
    run([ "ssh", host, "rm -fv " + remote_dir + "/binary-" + arch + "/*.deb" ])
    warn("Copying " + file + " on " + host)
    run([ "scp", file, host + ":" + remote_dir + "/binary-" + arch ])
    warn("Please wait 5 minute for repository indexes to get updated by cron")
    warn("Alternatively, you can run /srv/update-par4all.sh on " + host + " as root")


def publish_tgz(file, host, remote_dir):
    warn("Copying " + file + " on " + host)
    run([ "scp", file, host + ":" + remote_dir ])


def publish_files(files, nightly = False):
    global default_publish_host
    global default_publish_dir, default_nightly_publish_dir
    global default_deb_publish_dir, default_deb_nightly_publish_dir

    publish_dir = ""
    deb_publish_dir = ""
    if nightly:
        publish_dir = default_nightly_publish_dir
        deb_publish_dir = default_deb_nightly_publish_dir
    else:
        publish_dir = default_publish_dir
        deb_publish_dir = default_deb_publish_dir

    for file in files:
        file = os.path.abspath(os.path.expanduser(file))
        if not os.path.exists(file):
            die("Invalid file: " + file)
        ext = get_file_ext(file)
        if ext == ".deb":
            publish_deb(file, default_publish_host, deb_publish_dir)
        warn("Copying " + file + " in " + publish_dir + " on " + default_publish_host)
        run([ "scp", file, default_publish_host + ":" + publish_dir ])


def create_sdeb(pack_dir, install_prefix, revision, keep_temp = False, arch = None):
    die("create_sdeb is TODO")


def create_tgz(pack_dir, install_prefix, revision, keep_temp = False, arch = None):
    '''Creates a simple .tar.gz package.'''
    global package_name
    (temp_dir, temp_dir_with_prefix) = create_dist(pack_dir, install_prefix, revision)
    if not arch:
        arch = get_machine_arch()
    package_file_name = "_".join([ package_name, revision, arch ]) + ".tar.gz"
    package_file = os.path.abspath(package_file_name)
    if os.path.exists(package_file):
        warn("Removing existing " + package_file)
        os.remove(package_file)
    #~ new_temp_dir_with_prefix = os.path.join(os.path.split(temp_dir_with_prefix)[0], package_name + "_" + revision)
    #~ shutil.move(temp_dir_with_prefix, new_temp_dir_with_prefix)
    new_temp_dir_with_prefix = temp_dir_with_prefix
    tar_dir = os.path.split(new_temp_dir_with_prefix)[0] # one level up
    tar_cmd = " ".join([ "tar", "czvf", package_file_name, "-C", tar_dir, "." ])
    sh_cmd = '"chown -R root:root ' + tar_dir + " && find " + tar_dir + " -type d -exec chmod 755 '{}' \\; && " + tar_cmd + '"'
    run([ "fakeroot", "sh", "-c", sh_cmd])
    if os.path.exists(package_file_name):
        done("Created " + package_file_name)
    else:
        warn(package_file_name + " file not created!?")
    if keep_temp:
        warn("Temporary directory was " + temp_dir)
    else:
        rmtree(temp_dir, can_fail = 1)
    return package_file


def create_stgz(pack_dir, install_prefix, revision, keep_temp = False, arch = None):
    global package_name, temp_dirs
    package_full_name = "_".join([ package_name, revision, "src" ])
    package_file_name = package_full_name + ".tar.gz"
    package_file = os.path.abspath(package_file_name)
    package_file_tar = change_file_ext(package_file, "")
    if os.path.exists(package_file):
        warn("Removing existing " + package_file)
        os.remove(package_file)
    git = p4a_git(script_dir)
    current_branch = git.current_branch()
    if current_branch != "p4a":
        die("Not on branch p4a (" + current_branch + "), cannot create a source package")
    prefix = package_name + "_src"
    #~ git.archive(change_file_ext(package_file, ""), prefix = package_full_name + "/")
    git.archive(package_file_tar, prefix = prefix + "/")
    temp_dir = tempfile.mkdtemp(prefix = "p4a_pack_version_")
    debug("Temp dir is " + temp_dir)
    temp_dirs.append(temp_dir)
    prev_cwd = os.getcwd()
    os.chdir(temp_dir)
    os.makedirs(os.path.join(temp_dir, prefix))
    relative_version_file = os.path.join(prefix, "VERSION")
    write_file(relative_version_file, revision)
    tar_cmd = " ".join([ "tar", "uvf", package_file_tar, relative_version_file ])
    sh_cmd = '"chown root:root ' + relative_version_file + " && " + tar_cmd + '"'
    run([ "fakeroot", "sh", "-c", sh_cmd])
    os.chdir(prev_cwd)
    rmtree(temp_dir)
    run([ "gzip", "-9", package_file_tar ])
    if os.path.exists(package_file_name):
        done("Created " + package_file_name)
    return package_file


def main(options, args = []):

    if len(args):
        if not options.publish:
            die("You specified files on command line but did not specify --publish")
        publish_files(args, options.nightly)
        return

    if options.nightly and not options.append_date:
        options.append_date = True

    if (not options.deb #and not options.sdeb 
        and not options.tgz and not options.stgz):
        warn("--deb and/or --tgz and/or --stgz not specified, assuming --deb --tgz --stgz")
        options.deb = True
        options.tgz = True
        options.stgz = True

    prefix = options.install_prefix
    if prefix:
        warn("Installation prefix: " + prefix + " (--install-prefix)")
    else:
        prefix = default_install_prefix
        warn("Installation prefix: " + prefix + " (default; override with --install-prefix)")

    # Strip forward slash:
    if prefix[0] == "/" and len(prefix):
        prefix = prefix[1:]

    pack_dir = options.pack_dir
    if options.deb or options.tgz:
        if pack_dir:
            pack_dir = os.path.realpath(os.path.abspath(os.path.expanduser(pack_dir)))
            warn("Par4All installation is to be found in " + pack_dir + " (--pack-dir)")
        else:
            pack_dir = default_pack_dir
            warn("Assuming Par4All is currently installed in " + pack_dir + " (default; use --pack-dir to override)")
        if not os.path.isdir(pack_dir):
            die("Directory does not exist: " + pack_dir)

    output_dir = options.pack_output_dir
    if output_dir:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        warn("Packages will be put in " + output_dir + " (--pack-output-dir)")
        if output_dir == os.getcwd():
            output_dir = None
    else:
        warn("Packages will be put in " + os.getcwd() + " (default; override with --pack-output-dir)")

    version = ""
    if options.version:
        version = options.version
        warn("Version: " + version + " (--version)")
    else:
        global default_version
        version = default_version
        warn("Version: " + version + " (default; override with --version)")

    tmp = version.split("-")
    if len(tmp) > 2:
        die("Invalid revision format: it can have only one '-', e.g. 0.2-beta~foo")
    if options.append_date and len(tmp) > 1:
        die("Invalid revision format: no '-' permitted with --append-date")
    if not re.match("\d\.\d(\.\d)?", tmp[0]):
        die("Revision must begin with a version string, e.g. 1.2.3 or 0.2")
    for i in range(len(tmp)):
        tmp[i] = re.sub("[^\w~\.]", "", tmp[i])
    if options.append_date:
        tmp.append(time.strftime("%Y%m%dT%H%M%S"))
    revision = "-".join(tmp)

    if not revision:
        die("Invalid characters in revision")

    append_revision_bin = ""
    append_revision_src = ""

    if options.deb or options.tgz:
        append_revision_bin = guess_file_revision(pack_dir)
        if not append_revision_bin:
            die("Unable to determine appended revision for binary packages")
        append_revision_bin = append_revision_bin.replace("~exported", "")
        debug("Appended revision for binary packages: " + append_revision_bin)

    if options.stgz:
        append_revision_src = p4a_git(script_dir).current_revision()
        if not append_revision_src:
            die("Unable to determine appended revision for source packages")
        debug("Appended revision for source packages: " + append_revision_bin)

    # Check that fakeroot is available.
    if not which("fakeroot"):
        die("fakeroot not found, please install it (sudo aptitude install fakeroot)")

    global temp_dirs
    output_files = []
    try:
        if options.deb:
            output_files.append(create_deb(pack_dir = pack_dir, install_prefix = prefix, 
                revision = revision + "~" + append_revision_bin,
                keep_temp = options.keep_temp, arch = options.arch))
        #~ if options.sdeb:
            #~ output_files.append(create_sdeb(pack_dir = pack_dir, install_prefix = options.install_prefix, revision = revision,
                #~ keep_temp = options.keep_temp, arch = options.arch))
        if options.tgz:
            output_files.append(create_tgz(pack_dir = pack_dir, install_prefix = prefix,
                revision = revision + "~" + append_revision_bin,
                keep_temp = options.keep_temp, arch = options.arch))
        if options.stgz:
            output_files.append(create_stgz(pack_dir = pack_dir, install_prefix = prefix,
                revision = revision + "~" + append_revision_src,
                keep_temp = options.keep_temp, arch = options.arch))

        if options.publish:
            publish_files(output_files, options.nightly)

        if output_dir:
            for file in output_files:
                dest_file = os.path.join(output_dir, os.path.split(file)[1])
                if dest_file == file:
                    continue
                if os.path.exists(dest_file):
                    warn("Removing existing " + dest_file)
                    os.remove(dest_file)
                run([ "mv", "-v", file, dest_file ])
    except:
        for dir in temp_dirs:
            if os.path.isdir(dir):
                warn("NOT Removing " + dir)
        raise

    for dir in temp_dirs:
        if os.path.isdir(dir):
            if options.keep_temp:
                warn("NOT Removing " + dir + " (--keep-temp)")
            else:
                rmtree(dir, can_fail = True)


# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
