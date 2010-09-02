#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Ronan Keryell <ronan.keryell@hpc-project.com>
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#

'''
Par4All setup script implementation module.
'''

import string, sys, os, re, optparse, tempfile, shutil
from p4a_util import *
from p4a_rc import *
from p4a_git import *
from p4a_version import *


default_configure_opts = [ "--disable-static", "CFLAGS='-O2 -std=c99'" ]
default_debug_configure_opts = [ "--disable-static", "CFLAGS='-ggdb -g3 -O0 -Wall -std=c99'" ]
default_pips_conf_opts = [ "--enable-tpips", "--enable-pyps", "--enable-hpfc" ]


def add_module_options(parser):
    '''Add options specific to this module to an existing optparse options parser.'''

    group = optparse.OptionGroup(parser, "Setup Options")

    group.add_option("--rebuild", "-R", action = "store_true", default = False,
        help = "Rebuild the packages completely.")

    group.add_option("--clean", "-C", action = "store_true", default = False,
        help = "Wipe out the installation directory before proceeding. Implies -R and not skipping any package.")

    group.add_option("--skip-polylib", "--sp", action = "store_true", default = False,
        help = "Skip building and installing of the polylib library.")

    group.add_option("--skip-newgen", "--sn", action = "store_true", default = False,
        help = "Skip building and installing of the newgen library.")

    group.add_option("--skip-linear", "--sl", action = "store_true", default = False,
        help = "Skip building and installing of the linear library.")

    group.add_option("--skip-pips", "--sP", action = "store_true", default = False,
        help = "Skip building and installing of PIPS.")

    group.add_option("--skip", "-s", metavar = "PACKAGE", action = "append", default = [],
        help = "Alias for being able to say -s pips (besides --skip pips), for example. -sall or --skip all are also available and means 'skip all', in which case only final installation stages will be performed.")

    group.add_option("--only", "-o", metavar = "PACKAGE", action = "append", default = [],
        help = "Build only the selected package. Overrides any other option.")

    group.add_option("--reconf", "-r", metavar = "PACKAGE", action = "append", default = [],
        help = "Always run autoreconf and configure selected packages. By default, only packages which lack a Makefile will be reconfigured. If --rebuild is specified, all packages will be reconfigured.")

    group.add_option("--root", metavar = "DIR", default = None,
        help = "Specify the directory for the Par4All source tree. The default is to use the source tree from which this script comes.")

    group.add_option("--packages-dir", "--package-dir", "-P", metavar = "DIR", default = None,
        help = "Specify the packages location. By default it is <root>/packages.")

    # XXX: DISABLED because PIPS does not play well with "make install"'s DESTDIR parameter.
    #~ group.add_option("--dest-dir", "--dest", "-d", metavar = "DIR", default = None,
           #~ help = "[IGNORED, BLANK] Specify the staging installation directory.")

    group.add_option("--prefix", "-p", metavar = "DIR", default = None,
        help = "Specify the prefix used to configure the packages. Default is /usr/local/par4all.")

    group.add_option("--polylib-src", metavar = "DIR", default = None,
        help = "Specify polylib source directory.")

    group.add_option("--newgen-src", metavar = "DIR", default = None,
        help = "Specify newgen source directory.")

    group.add_option("--linear-src", metavar = "DIR", default = None,
        help = "Specify linear source directory.")

    group.add_option("--pips-src", metavar = "DIR", default = None,
        help = "Specify PIPS source directory. When changing the directory, do not forget to reconfigure since this is when the source location is taken into account.")

    group.add_option("--nlpmake-src", metavar = "DIR", default = None,
        help = "Specify nlpmake source directory.")

    global default_configure_opts, default_debug_configure_opts
    group.add_option("--configure-opts", "--configure-flags", "-c", metavar = "OPTS", action = "append", default = [],
        help = "Specify global configure opts. Default is '" + " ".join(default_configure_opts)
        + "' OR '" + " ".join(default_debug_configure_opts) + "' if --debug is specified.")

    group.add_option("--debug", "-g", action = "store_true", default = False,
        help = "Set debug CFLAGS in configure opts (see --configure-opts). Please note that this option has NO EFFECT if --configure-opts is manually set.")

    group.add_option("--polylib-conf-opts", "--polylib-conf-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify polylib configure opts (appended to --configure-opts).")

    group.add_option("--newgen-conf-opts", "--newgen-conf-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify newgen configure opts (appended to --configure-opts).")

    group.add_option("--linear-conf-opts", "--linear-conf-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify linear configure opts (appended to --configure-opts).")

    global default_pips_conf_opts
    group.add_option("--pips-conf-opts", "--pips-conf-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify PIPS configure opts (appended to --configure-opts). Defaults to " + " ".join(default_pips_conf_opts))

    group.add_option("--make-opts", "--make-flags", "-m", metavar = "OPTS", action = "append", default = [],
        help = "Specify global make opts.")

    group.add_option("--polylib-make-opts", "--polylib-make-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify polylib make opts (appended to --make-opts).")

    group.add_option("--newgen-make-opts", "--newgen-make-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify newgen make opts (appended to --make-opts).")

    group.add_option("--linear-make-opts", "--linear-make-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify linear make opts (appended to --make-opts).")

    group.add_option("--pips-make-opts", "--pips-make-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify PIPS make opts (appended to --make-opts).")

    group.add_option("--jobs", "-j", metavar = "COUNT", default = None,
        help = "Make packages concurrently using COUNT jobs.")

    group.add_option("--no-install", "-I", action = "store_true", default = False,
        help = "Do not install any package (do not run make install for any package). NB: this might break the compilation of packages depending on the binaries of uninstalled previous packages.")

    group.add_option("--no-final", "-F", action = "store_true", default = False,
        help = "Skip final installations steps in install directory (installation of various files). NB: never running the final installation step will not give you a functional Par4All build.")

    parser.add_option_group(group)


def build_package(package_dir, build_dir, dest_dir, configure_opts = [], make_opts = [], install = True, reconf = False):
    '''Builds the given package in package_dir using autotools.'''

    # Normalize paths because a relative path makes the configure to fail
    package_dir = os.path.abspath(package_dir)
    build_dir = os.path.abspath(build_dir)
    dest_dir = os.path.abspath(dest_dir)

    configure_script = os.path.join(package_dir, "configure")
    makefile = os.path.join(build_dir, "Makefile")

    if (not os.path.exists(configure_script)
        or not os.path.isdir(build_dir)
        or not os.path.exists(makefile)):
        reconf = True

    if reconf:
        # Call autoconf to generate the configure utility.
        run([ "autoreconf", "--install" ], working_dir = package_dir)
        print package_dir
        #~ if dest_dir:
            #~ configure_opts += [ "DESTDIR=" + dest_dir ]
        # Call configure to generate the Makefiles.
        print build_dir
        run([ configure_script ] + configure_opts, working_dir = build_dir)

    # Call make all to compile.
    info("Building " + package_dir + " in " + build_dir)
    run([ "make" ] + make_opts, working_dir = build_dir)

    if install:
        # Call make install to install in DESTDIR if requested.
        install_make_opts = []
        #~ if dest_dir:
            #~ install_make_opts.append("DESTDIR=" + dest_dir)
            #~ info("Installing " + package_dir + " in " + dest_dir)
        run([ "make", "install" ] + install_make_opts, working_dir = build_dir)


def main(options, args = []):

    if args:
        die("No arguments are accepted by this script, only options")

    actual_script = change_file_ext(os.path.realpath(os.path.abspath(__file__)), ".py", if_ext = ".pyc")
    script_dir = os.path.split(actual_script)[0]
    default_root = os.path.normpath(os.path.join(script_dir, "..", ".."))
    default_prefix = "/usr/local/par4all"

    # Force options.dest_dir to None for now.
    options.dest_dir = None

    # Check options and exclude packages from
    # processing as requested.
    for s in options.skip:
        if s == "polylib":
            options.skip_polylib = True
        elif s == "newgen":
            options.skip_newgen = True
        elif s == "linear":
            options.skip_linear = True
        elif s == "pips":
            options.skip_pips = True
        elif s == "all":
            options.skip_polylib = True
            options.skip_newgen = True
            options.skip_linear = True
            options.skip_pips = True
        else:
            die("Invalid option: --skip=" + s)
        if options.clean:
            die("--skip is not compatible with --clean")
    for s in options.only:
        # Skip everything...
        options.skip_polylib = True
        options.skip_newgen = True
        options.skip_linear = True
        options.skip_pips = True
        # ... but
        if s == "polylib":
            options.skip_polylib = False
        elif s == "newgen":
            options.skip_newgen = False
        elif s == "linear":
            options.skip_linear = False
        elif s == "pips":
            options.skip_pips = False
        else:
            die("Invalid option: --only=" + s)
        if options.clean:
            die("--only is not compatible with --clean")
    if options.clean:
        options.skip_polylib = False
        options.skip_newgen = False
        options.skip_linear = False
        options.skip_pips = False
        options.rebuild = True

    options.reconf_polylib = False
    options.reconf_newgen = False
    options.reconf_linear = False
    options.reconf_pips = False
    for s in options.reconf:
        if s == "polylib":
            options.reconf_polylib = True
        elif s == "newgen":
            options.reconf_newgen = True
        elif s == "linear":
            options.reconf_linear = True
        elif s == "pips":
            options.reconf_pips = True
        elif s == "all":
            options.reconf_polylib = True
            options.reconf_newgen = True
            options.reconf_linear = True
            options.reconf_pips = True
        else:
            die("Invalid option: --reconf=" + s)

    # Initialize main variables and set defaults.
    # "root" is the Par4All source root directory.
    root = ""
    if options.root:
        root = os.path.abspath(os.path.expanduser(options.root))
        warn("Par4All source tree root is " + root + " (--root)")
    # If environment variable P4A_ROOT is defined and
    # --root was not specified, pick it up from there.
    #~ elif "P4A_ROOT" in os.environ and os.environ["P4A_ROOT"]:
        #~ root = os.path.abspath(os.path.expanduser(os.environ["P4A_ROOT"]))
        #~ warn("Assuming Par4All source tree root is " + root + " (P4A_ROOT environment variable)")
    else:
        root = default_root
        warn("Assuming Par4All source tree root is " + root + " (default; use --root to override)")
    if not os.path.isdir(root):
        die("Directory does not exist: " + root)
    #~ info("Par4All source tree root: " + root)

    # "packages_dir" is where the source packages lie.
    packages_dir = ""
    if options.packages_dir:
        packages_dir = os.path.abspath(os.path.expanduser(options.packages_dir))
        warn("Packages directory is " + packages_dir + " (--packages-dir)")
    else:
        packages_dir = os.path.join(root, "packages")
        warn("Assuming packages directory is " + packages_dir + " (individual packages location may be overriden with --xxx-src)")
    #~ if not os.path.isdir(packages_dir):
        #~ die("Invalid packages dir: " + packages_dir)

    # "dest_dir" is the staging installation directory.
    # XXX: DISABLED because PIPS does not play well with "make install"'s DESTDIR parameter.
    if options.dest_dir:
        die("--dest-dir is ignored for now")
    dest_dir = "" #options.dest_dir
    #if not dest_dir:
    #    dest_dir = os.path.join(root, "run") # By default.
    if dest_dir:
        debug("DESTDIR=" + dest_dir)
        dest_dir = os.path.abspath(os.path.expanduser(dest_dir)) # Make it absolute whatsoever.

    # "prefix" is the installation prefix which is passed
    # as option --prefix when configure is called for the
    # various packages.
    prefix = default_prefix
    if options.prefix:
        prefix = os.path.abspath(os.path.expanduser(options.prefix))
        warn("Prefix is " + prefix + " (--prefix)")
    else:
        warn("Assuming prefix is " + prefix + " (default; use --prefix to override)")

    # "safe_prefix" is the same as prefix except that
    # if prefix is empty or does not begin with a /,
    # we prepend /.
    safe_prefix = ""
    if not prefix or prefix[0] != "/":
        safe_prefix = "/" + prefix
    else:
        safe_prefix = prefix
    debug("Prefix: " + quote(prefix) + " (" + safe_prefix + ")")

    # "install_dir" is the most important variable here.
    # It is dest_dir + safe_prefix.
    install_dir = os.path.normpath(dest_dir + safe_prefix)
    warn("Install directory is " + install_dir)
    # Check that we are not installing in a system or an invalid directory.
    if not install_dir or is_system_dir(install_dir):
        die("Invalid installation/staging directory: " + install_dir + ". It must not be a system directory")
    # Create install_dir it if it does not already exist.
    if os.path.isdir(install_dir):
        if options.clean:
            if glob.glob(os.path.join(install_dir, "*")):
                # If we are requested to clean first, remove everything
                # under install_dir.
                warn("Removing everything in " + install_dir + " (--clean)")
                rmtree(install_dir, remove_top = False)
        else:
            info("Install directory " + install_dir + " already exists")
    else:
        os.makedirs(install_dir)

    # Build directory: where the Makefile are generated, where the make commands are issued, etc.
    build_dir = os.path.join(root, "build")
    debug("Build directory: " + build_dir)

    # Path for source packages:

    polylib_src_dir = ""
    if options.polylib_src:
        polylib_src_dir = options.polylib_src
    else:
        polylib_src_dir = os.path.join(packages_dir, "polylib")
    debug("polylib source directory: " + polylib_src_dir)
    if not os.path.isdir(polylib_src_dir) and not options.skip_polylib:
        die("polylib source directory does not exist: " + polylib_src_dir)

    newgen_src_dir = ""
    if options.newgen_src:
        newgen_src_dir = options.newgen_src
    else:
        newgen_src_dir = os.path.join(packages_dir, "PIPS/newgen")
    debug("newgen source directory: " + newgen_src_dir)
    if not os.path.isdir(newgen_src_dir) and not options.skip_newgen:
        die("newgen source directory does not exist: " + newgen_src_dir)

    linear_src_dir = ""
    if options.linear_src:
        linear_src_dir = options.linear_src
    else:
        linear_src_dir = os.path.join(packages_dir, "PIPS/linear")
    debug("linear source directory: " + linear_src_dir)
    if not os.path.isdir(linear_src_dir) and not options.skip_linear:
        die("linear source directory does not exist: " + linear_src_dir)

    pips_src_dir = ""
    if options.pips_src:
        pips_src_dir = options.pips_src
    else:
        pips_src_dir = os.path.join(packages_dir, "PIPS/pips")
    debug("PIPS source directory: " + pips_src_dir)
    if not os.path.isdir(pips_src_dir) and not options.skip_pips:
        die("PIPS source directory does not exist: " + pips_src_dir)

    nlpmake_src_dir = ""
    if options.nlpmake_src:
        nlpmake_src_dir = options.nlpmake_src
    else:
        nlpmake_src_dir = os.path.join(packages_dir, "PIPS/nlpmake")
    # Normalize the directory so that we can build symbolic links to here
    # easily later:
    nlpmake_src_dir = os.path.abspath(nlpmake_src_dir)
    debug("nlpmake source directory: " + nlpmake_src_dir)
    #~ if not os.path.isdir(nlpmake_src_dir):
        #~ die("Directory does not exist: " + nlpmake_src_dir)

    # Global configure flags:
    configure_opts = [ "--prefix=" + prefix ]
    if options.configure_opts:
        configure_opts += options.configure_opts
    else:
        if options.debug:
            configure_opts += default_debug_configure_opts
        else:
            configure_opts += default_configure_opts

    # Global make flags:
    make_opts = []
    if options.make_opts:
        make_opts.append(options.make_opts)
    if options.jobs:
        make_opts.append("-j" + options.jobs)

    #~ if get_verbosity() == 0:
        #~ warn("Building and installing", spin = True)

    ############################## polylib

    if not options.skip_polylib:

        info("Processing polylib")

        package_build_dir = os.path.join(build_dir, "polylib")

        # Rebuild requested? Delete existing build directory.
        if options.rebuild and os.path.isdir(package_build_dir):
                info("Package " + polylib_src_dir + " marked for rebuild, removing existing build dir " + package_build_dir)
                rmtree(package_build_dir)

        if not os.path.isdir(package_build_dir):
            os.makedirs(package_build_dir)

        polylib_conf_opts = configure_opts
        if options.polylib_conf_opts:
            polylib_conf_opts.append(options.polylib_conf_opts)
        polylib_make_opts = make_opts
        if options.polylib_make_opts:
            polylib_make_opts.append(options.polylib_make_opts)

        build_package(package_dir = polylib_src_dir, build_dir = package_build_dir,
            configure_opts = polylib_conf_opts, make_opts = polylib_make_opts, dest_dir = dest_dir,
            install = not options.no_install, reconf = options.reconf_polylib)

    ##############################

    # This was used for testing with DESTDIR...
    configure_opts += [
        #'POLYLIB64_CFLAGS="-I' + os.path.join(install_dir, "include") + '"',
        #'POLYLIB64_LIBS="-L' + os.path.join(install_dir, "lib") + ' -lpolylib64"',
        #'CFLAGS="-g -O2 -I' + os.path.join(install_dir, "include") + '"',
        #'CPPFLAGS="' + '-I' + os.path.join(install_dir, "include") + '"',
        #'LDFLAGS="-Wl,-z,defs -L' + os.path.join(install_dir, "lib") + '"'
        ]

    ############################## newgen

    if not options.skip_newgen:
        info("Processing newgen")

        package_build_dir = os.path.join(build_dir, "newgen")

        # Rebuild requested? Delete existing build directory.
        if options.rebuild and os.path.isdir(package_build_dir):
                info("Package " + newgen_src_dir + " marked for rebuild, removing existing build dir " + package_build_dir)
                rmtree(package_build_dir)

        if not os.path.isdir(package_build_dir):
            os.makedirs(package_build_dir)

        # Make a symlink to the old make infrastructure (and remove any existing one
        # or a symlink recursion will appear).
        run([ "rm", "-Rfv", os.path.join(newgen_src_dir, "makes") ])
        run([ "ln", "-sv", os.path.join(nlpmake_src_dir, "makes"), os.path.join(newgen_src_dir, "makes") ])

        newgen_conf_opts = configure_opts
        if options.newgen_conf_opts:
            newgen_conf_opts.append(options.newgen_conf_opts)
        newgen_make_opts = make_opts
        if options.newgen_make_opts:
            newgen_make_opts.append(options.newgen_make_opts)

        build_package(package_dir = newgen_src_dir, build_dir = package_build_dir,
            configure_opts = newgen_conf_opts, make_opts = newgen_make_opts, dest_dir = dest_dir,
            install = not options.no_install, reconf = options.reconf_newgen)

    ##############################

    # This was used for testing with DESTDIR...
    #configure_opts += [ 'NEWGENLIBS_CFLAGS="-I' + os.path.join(install_dir, "include") + '"',
    #    'NEWGENLIBS_LIBS="-L' + os.path.join(install_dir, "lib") + ' -lnewgenlibs"' ]
    configure_opts += [ "PKG_CONFIG_PATH=" + quote(os.path.join(install_dir, "lib/pkgconfig")) ]

    ############################## linear

    if not options.skip_linear:
        info("Processing linear")

        package_build_dir = os.path.join(build_dir, "linear")

        # Rebuild requested? Delete existing build directory.
        if options.rebuild and os.path.isdir(package_build_dir):
                info("Package " + linear_src_dir + " marked for rebuild, removing existing build dir " + package_build_dir)
                rmtree(package_build_dir)

        if not os.path.isdir(package_build_dir):
            os.makedirs(package_build_dir)

        # Make a symlink to the old make infrastructure (and remove any existing one
        # or a symlink recursion will appear).
        run([ "rm", "-Rfv", os.path.join(linear_src_dir, "makes") ])
        run([ "ln", "-sv", os.path.join(nlpmake_src_dir, "makes"), os.path.join(linear_src_dir, "makes") ])
        linear_conf_opts = configure_opts

        if options.linear_conf_opts:
            linear_conf_opts.append(options.linear_conf_opts)
        linear_make_opts = make_opts
        if options.linear_make_opts:
            linear_make_opts.append(options.linear_make_opts)

        build_package(package_dir = linear_src_dir, build_dir = package_build_dir,
            configure_opts = linear_conf_opts, make_opts = linear_make_opts, dest_dir = dest_dir,
            install = not options.no_install, reconf = options.reconf_linear)

    ##############################

    # This was used for testing with DESTDIR...
    #configure_opts += [ 'LINEARLIBS_CFLAGS="-I' + os.path.join(install_dir, "include") + ' -DLINEAR_VALUE_IS_LONGLONG -DLINEAR_VALUE_PROTECT_MULTIPLY -DLINEAR_VALUE_ASSUME_SOFTWARE_IDIV"',
    #    'LINEARLIBS_LIBS="-L' + os.path.join(install_dir, "lib") + ' -llinearlibs"' ]
    #~ configure_opts += [ 'PATH="' + os.path.join(install_dir, "bin") + ':' + env("PATH") + '"',
        #~ 'LD_LIBRARY_PATH="' + os.path.join(install_dir, "lib") + ':' + env("LD_LIBRARY_PATH") + '"' ]

    # Update the PATH. Needed because PIPS relies on utilities built by newgen.
    add_to_path(os.path.join(install_dir, "bin"))

    # This was used for testing with DESTDIR...
    #add_to_path(os.path.join(install_dir, "lib"), var = "LD_LIBRARY_PATH")

    ############################## pips

    if not options.skip_pips:
        info("Processing pips")

        package_build_dir = os.path.join(build_dir, "pips")

        # Rebuild requested? Delete existing build directory.
        if options.rebuild and os.path.isdir(package_build_dir):
                info("Package " + pips_src_dir + " marked for rebuild, removing existing build dir " + package_build_dir)
                rmtree(package_build_dir)

        if not os.path.isdir(package_build_dir):
            os.makedirs(package_build_dir)

        # Make a symlink to the old make infrastructure (and remove any existing one
        # or a symlink recursion will appear).
        run([ "rm", "-Rfv", os.path.join(pips_src_dir, "makes") ])
        run([ "ln", "-sv", os.path.join(nlpmake_src_dir, "makes"), os.path.join(pips_src_dir, "makes") ])

        # Fix the following error:
        # /bin/sed: can't read /lib/libpolylib64.la: No such file or directory
        # libtool: link: `/lib/libpolylib64.la' is not a valid libtool archive
        # make[5]: *** [libpipslibs.la] Error 1
        # make[5]: Leaving directory `/home/gpean/p4a-foo/build/pips/src/Libs'
        #~ run([ "sudo", "ln", "-sfv", os.path.join(install_dir, "lib/libpolylib64.la"), os.path.join(safe_prefix, "lib/libpolylib64.la") ])

        ### FIX for fortran
        fortran = os.path.join(build_dir, "pips/src/Passes/fortran95")
        if not os.path.isdir(fortran):
            os.makedirs(fortran)
        # Copy with a rsync instead of simply symlinking the
        # source directory because the Fortran95 parser build
        # patches the sources and that would mark the files as
        # touched in the git repositiry (if any). Use --delete so
        # that if this setup is run again, the .files are removed
        # to relauch the patch:
        run([ "rsync", "-rv", "--delete", os.path.join(packages_dir, "pips-gfc/."), os.path.join(fortran, "gcc-4.4.3") ])
        # To cheat the Makefile process that would like to
        # download the sources from the Internet:
        for file in [ "gcc-4.4.3.md5", "gcc-core-4.4.3.tar.bz2", "gcc-fortran-4.4.3.tar.bz2" ]:
            run([ "touch", os.path.join(fortran, file) ])
        fortran2 = os.path.join(fortran, "gcc-4.4.3")
        if not os.path.isdir(fortran2):
            os.makedirs(fortran2)
        for file in [ ".dir", ".md5-check-core", ".md5-check-fortran", ".untar-core", ".untar-fortran", ".untar" ]:
            run([ "touch", os.path.join(fortran2, file) ])
        ### End of FIX for fortran

        pips_conf_opts = configure_opts
        if options.pips_conf_opts:
            pips_conf_opts.append(options.pips_conf_opts)
        else:
            global default_pips_conf_opts
            pips_conf_opts += default_pips_conf_opts
        pips_make_opts = make_opts
        if options.pips_make_opts:
            pips_make_opts.append(options.pips_make_opts)

        build_package(package_dir = pips_src_dir, build_dir = package_build_dir,
            configure_opts = pips_conf_opts, make_opts = pips_make_opts, dest_dir = dest_dir,
            install = not options.no_install, reconf = options.reconf_pips)


    ##############################

    if options.no_final:
        warn("Skipping final installation steps (--no-final)")
        return


    # Proceed with local scripts and libraries installation.

    # Create directory tree.
    info("Creating dirs")
    install_dir_bin = os.path.join(install_dir, "bin")
    if not os.path.isdir(install_dir_bin):
        os.makedirs(install_dir_bin)
    install_dir_etc = os.path.join(install_dir, "etc")
    if not os.path.isdir(install_dir_etc):
        os.makedirs(install_dir_etc)
    install_dir_lib = os.path.join(install_dir, "lib")
    if not os.path.isdir(install_dir_lib):
        os.makedirs(install_dir_lib)
    install_dir_share = os.path.join(install_dir, "share")
    if not os.path.isdir(install_dir_share):
        os.makedirs(install_dir_share)
    install_dir_share_accel = os.path.join(install_dir_share, "p4a_accel")
    if not os.path.isdir(install_dir_share_accel):
        os.makedirs(install_dir_share_accel)
    install_dir_makes = os.path.join(install_dir, "makes")
    if not os.path.isdir(install_dir_makes):
        os.makedirs(install_dir_makes)

    # Install a few scripts.
    info("Installing scripts")

    for file in [
        "src/dev/p4a_git",
        "src/dev/p4a_valgrind",
        "src/simple_tools/p4a",
        "src/simple_tools/p4a_process",
        "src/postprocessor/p4a_recover_includes",
        "src/validation/p4a_validate",
        "src/validation/p4a_validation",
        "src/p4a_accel/p4a_post_processor.py"
        ]:
        run([ "cp", "-rv", "--remove-destination", os.path.join(root, file), install_dir_bin ])

    for file in [ "src/dev/p4a_git_lib.bash" ]:
        run([ "cp", "-rv", "--remove-destination", os.path.join(root, file), install_dir_etc ])

    # Install accelerator source.
    info("Installing accel files")
    accel_src_dir = os.path.join(root, "src/p4a_accel")
    for file in os.listdir(accel_src_dir):
        ext = os.path.splitext(file)[1]
        if ext == ".h" or ext == ".c" or ext == ".f" or ext == ".mk" or ext == ".cu":
            run([ "cp", "-rv", "--remove-destination", os.path.join(accel_src_dir, file), install_dir_share_accel ])

    # Copy python dependencies and templates.
    info("Copying python libs")
    install_python_lib_dir = get_python_lib_dir(install_dir)
    #~ for file in os.listdir(install_dir_lib):
        #~ if file.startswith("python") and os.path.isdir(os.path.join(install_dir_lib, file)):
            #~ install_python_lib_dir = os.path.join(install_dir_lib, file, "site-packages/pips")
            #~ if not os.path.isdir(install_python_lib_dir):
                #~ install_python_lib_dir = os.path.join(install_dir_lib, file, "dist-packages/pips")
            #~ break
    #~ if not install_python_lib_dir:
        #~ die("Cannot not determine python lib dir in " + install_dir_lib + ", try --rebuild")
    dir = os.path.join(root, "src/simple_tools")
    for file in os.listdir(dir):
        ext = os.path.splitext(file)[1]
        if ext == ".py" or ext == ".tpl":
            run([ "cp", "-rv", "--remove-destination", os.path.join(dir, file), install_python_lib_dir ])

    # Install stuff still lacking from PIPS install.
    info("Installing pips scripts")
    dir = os.path.join(pips_src_dir, "src/Scripts/validation")
    for file in os.listdir(dir):
        if file.startswith("pips"):
            run([ "cp", "-rv", "--remove-destination", os.path.join(dir, file), install_dir_bin ])
    run([ "cp", "-rv", "--remove-destination", os.path.join(pips_src_dir, "src/Scripts/misc/logfile_to_tpips"), install_dir_bin ])

    # Fix validation.
    info("Fixing validation")
    dir = os.path.join(nlpmake_src_dir, "makes")
    for file in os.listdir(dir):
        if file == "arch.sh" or file == "version.sh":
            run([ "cp", "-rv", "--remove-destination", os.path.join(dir, file), install_dir_makes ])

    # Install various files.
    info("Installing release notes")
    run([ "cp", "-rv", "--remove-destination", os.path.join(root, "RELEASE-NOTES.txt"), install_dir ])
    info("Installing examples")
    run([ "cp", "-rv", "--remove-destination", os.path.join(root, "examples"), install_dir ])

    # Write the environment shell scripts.
    info("Writing shell rc files")
    fortran = ""
    if which("gfortran"):
        fortran = "gfortran"
    elif which("g77"):
        fortran = "g77"
    else:
        fortran = "false"
    p4a_write_rc(install_dir_etc, dict(root = install_dir, dist = install_dir,
        accel = install_dir_share_accel, fortran = fortran))

    # Write version file.
    write_VERSION(install_dir, VERSION(root))
    write_GITREV(install_dir, GITREV(root))
    (revision, versiond) = make_full_revision(install_dir)

    done("")
    done("All done. Par4All " + revision + " is ready and has been installed in " + install_dir)
    done("To begin using it, you should source, depending on your shell religion:")
    done("")
    done("  " + os.path.join(install_dir, "etc/par4all-rc.sh") + " (for bash, dash, sh...) or")
    done("  " + os.path.join(install_dir, "etc/par4all-rc.csh") + " (tcsh, csh...)")
    done("")


# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
