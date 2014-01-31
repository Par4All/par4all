#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Ronan Keryell <ronan.keryell@hpc-project.com>
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#
import p4a_opts
import p4a_rc
import p4a_util
import p4a_version
import glob
import os
import optparse

'''
Par4All setup script implementation module.
'''



default_configure_opts = [ "--disable-static", "CFLAGS='-O2 -std=c99'" ]
default_debug_configure_opts = [ "--disable-static", "CFLAGS='-ggdb -g3 -O0 -Wall -std=c99'" ]
default_pips_conf_opts = [ "--enable-tpips", "--enable-pyps", "--enable-hpfc", "--enable-fortran95" ]


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

    group.add_option("--skip-examples", "--sE", action = "store_true", default = False,
        help = "Skip installing examples.")

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

    group.add_option("--build-dir", "-b", metavar = "DIR", default = "build",
        help = "Specify the build directory to be used relatively to the root directory as specify the --root option. Default to build")

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
    group.add_option("--configure-options", "--configure-flags", "-c", metavar = "OPTS", action = "append", default = [],
        help = "Specify global configure options. Default is '" + " ".join(default_configure_opts)
        + "' OR '" + " ".join(default_debug_configure_opts) + "' if --debug is specified.")

    group.add_option("--debug", "-g", action = "store_true", default = False,
        help = "Set debug CFLAGS in configure options (see --configure-options). Please note that this option has NO EFFECT if --configure-options is manually set.")

    group.add_option("--polylib-conf-options", "--polylib-conf-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify polylib configure opts (appended to --configure-options).")

    group.add_option("--newgen-conf-options", "--newgen-conf-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify newgen configure options (appended to --configure-options).")

    group.add_option("--linear-conf-options", "--linear-conf-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify linear configure options (appended to --configure-options).")

    global default_pips_conf_opts
    group.add_option("--pips-conf-options", "--pips-conf-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify PIPS configure options (appended to --configure-options). Defaults to " + " ".join(default_pips_conf_opts) +
                     ". Setting this option will reset the default value. Note that several flags can be set like this : " +
                     ' --pips-conf-options "--enable-tpips --enable-pyps --enable-doc"')

    group.add_option("--make-options", "--make-flags", "-m", metavar = "OPTS", action = "append", default = [],
        help = "Specify global make options.")

    group.add_option("--polylib-make-options", "--polylib-make-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify polylib make opts (appended to --make-options).")

    group.add_option("--newgen-make-options", "--newgen-make-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify newgen make options (appended to --make-options).")

    group.add_option("--linear-make-options", "--linear-make-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify linear make options (appended to --make-options).")

    group.add_option("--pips-make-options", "--pips-make-flags", metavar = "OPTS", action = "append", default = [],
        help = "Specify PIPS make options (appended to --make-options).")

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
        p4a_util.run([ "autoreconf", "--install" ], working_dir = package_dir)
        print package_dir
        #~ if dest_dir:
            #~ configure_opts.append ("DESTDIR=" + dest_dir)
        # Call configure to generate the Makefiles.
        print build_dir
        print [ configure_script ] + configure_opts
        p4a_util.run([ configure_script ] + configure_opts, working_dir = build_dir)

    # Call make all to compile.
    p4a_util.info("Building " + package_dir + " in " + build_dir)
    p4a_util.run([ "make" ] + make_opts, working_dir = build_dir)

    if install:
        # Call make install to install in DESTDIR if requested.
        install_make_opts = []
        #~ if dest_dir:
            #~ install_make_opts.append("DESTDIR=" + dest_dir)
            #~ p4a_util.info("Installing " + package_dir + " in " + dest_dir)
        p4a_util.run([ "make", "install" ] + install_make_opts, working_dir = build_dir)


def work(options, args = None):
    '''Do the real work. The main goal of this function is to be able to
    call p4a_setup from another tool (p4a_coffee.py) that has already
    parsed the arguments and options of the command.

    TODO: modularize this function...
    '''
    if args:
        p4a_util.die("No arguments are accepted by this script, only options")

    actual_script = p4a_util.change_file_ext(os.path.realpath(os.path.abspath(__file__)), ".py", if_ext = ".pyc")
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
        elif s == "example":
            options.skip_examples = True
        elif s == "all":
            options.skip_polylib = True
            options.skip_newgen = True
            options.skip_linear = True
            options.skip_pips = True
            options.skip_examples = True
        else:
            p4a_util.die("Invalid option: --skip=" + s)
        if options.clean:
            p4a_util.die("--skip is not compatible with --clean")
    for s in options.only:
        # Skip everything...
        options.skip_polylib = True
        options.skip_newgen = True
        options.skip_linear = True
        options.skip_pips = True
        options.skip_examples = True
        # ... but
        if s == "polylib":
            options.skip_polylib = False
        elif s == "newgen":
            options.skip_newgen = False
        elif s == "linear":
            options.skip_linear = False
        elif s == "pips":
            options.skip_pips = False
        elif s == "example":
            options.skip_examples = False
        else:
            p4a_util.die("Invalid option: --only=" + s)
        if options.clean:
            p4a_util.die("--only is not compatible with --clean")
    if options.clean:
        options.skip_polylib = False
        options.skip_newgen = False
        options.skip_linear = False
        options.skip_pips = False
        options.skip_examples = False
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
            p4a_util.die("Invalid option: --reconf=" + s)

    # Initialize main variables and set defaults.
    # "root" is the Par4All source root directory.
    root = ""
    if options.root:
        root = os.path.abspath(os.path.expanduser(options.root))
        p4a_util.warn("Par4All source tree root is " + root + " (--root)")
    # If environment variable P4A_ROOT is defined and
    # --root was not specified, pick it up from there.
    #~ elif "P4A_ROOT" in os.environ and os.environ["P4A_ROOT"]:
        #~ root = os.path.abspath(os.path.expanduser(os.environ["P4A_ROOT"]))
        #~ p4a_util.warn("Assuming Par4All source tree root is " + root + " (P4A_ROOT environment variable)")
    else:
        root = default_root
        p4a_util.warn("Assuming Par4All source tree root is " + root + " (default; use --root to override)")
    if not os.path.isdir(root):
        p4a_util.die("Directory does not exist: " + root)
    #~ p4a_util.info("Par4All source tree root: " + root)

    # "packages_dir" is where the source packages lie.
    packages_dir = ""
    if options.packages_dir:
        packages_dir = os.path.abspath(os.path.expanduser(options.packages_dir))
        p4a_util.warn("Packages directory is " + packages_dir + " (--packages-dir)")
    else:
        packages_dir = os.path.join(root, "packages")
        p4a_util.warn("Assuming packages directory is " + packages_dir + " (individual packages location may be overriden with --xxx-src)")
    #~ if not os.path.isdir(packages_dir):
        #~ p4a_util.die("Invalid packages dir: " + packages_dir)

    # "dest_dir" is the staging installation directory.
    # XXX: DISABLED because PIPS does not play well with "make install"'s DESTDIR parameter.
    if options.dest_dir:
        p4a_util.die("--dest-dir is ignored for now")
    dest_dir = "" #options.dest_dir
    #if not dest_dir:
    #    dest_dir = os.path.join(root, "run") # By default.
    if dest_dir:
        p4a_util.debug("DESTDIR=" + dest_dir)
        dest_dir = os.path.abspath(os.path.expanduser(dest_dir)) # Make it absolute whatsoever.

    # "prefix" is the installation prefix which is passed
    # as option --prefix when configure is called for the
    # various packages.
    prefix = default_prefix
    if options.prefix:
        prefix = os.path.abspath(os.path.expanduser(options.prefix))
        p4a_util.warn("Prefix is " + prefix + " (--prefix)")
    else:
        p4a_util.warn("Assuming prefix is " + prefix + " (default; use --prefix to override)")

    # "safe_prefix" is the same as prefix except that
    # if prefix is empty or does not begin with a /,
    # we prepend /.
    safe_prefix = ""
    if not prefix or prefix[0] != "/":
        safe_prefix = "/" + prefix
    else:
        safe_prefix = prefix
    p4a_util.debug("Prefix: " + p4a_util.quote(prefix) + " (" + safe_prefix + ")")

    # "install_dir" is the most important variable here.
    # It is dest_dir + safe_prefix.
    install_dir = os.path.normpath(dest_dir + safe_prefix)
    p4a_util.warn("Install directory is " + install_dir)
    # Check that we are not installing in a system or an invalid directory.
    if not install_dir or p4a_util.is_system_dir(install_dir):
        p4a_util.die("Invalid installation/staging directory: " + install_dir + ". It must not be a system directory")
    # Create install_dir it if it does not already exist.
    if os.path.isdir(install_dir):
        if options.clean:
            if glob.glob(os.path.join(install_dir, "*")):
                # If we are requested to clean first, remove everything
                # under install_dir.
                p4a_util.warn("Removing everything in " + install_dir + " (--clean)")
                p4a_util.rmtree(install_dir, remove_top = False)
        else:
            p4a_util.info("Install directory " + install_dir + " already exists")
    else:
        os.makedirs(install_dir)

    # Build directory: where the Makefile are generated, where the make commands are issued, etc.
    build_dir = os.path.join(root, options.build_dir)
    p4a_util.debug("Build directory: " + build_dir)

    # Path for source packages:

    polylib_src_dir = ""
    if options.polylib_src:
        polylib_src_dir = options.polylib_src
    else:
        polylib_src_dir = os.path.join(packages_dir, "polylib")
    p4a_util.debug("polylib source directory: " + polylib_src_dir)
    if not options.skip_polylib and not os.path.isdir(polylib_src_dir) and not options.skip_polylib:
        p4a_util.die("polylib source directory does not exist: " + polylib_src_dir)

    newgen_src_dir = ""
    if options.newgen_src:
        newgen_src_dir = options.newgen_src
    else:
        newgen_src_dir = os.path.join(packages_dir, "PIPS/newgen")
    p4a_util.debug("newgen source directory: " + newgen_src_dir)
    if not os.path.isdir(newgen_src_dir) and not options.skip_newgen:
        p4a_util.die("newgen source directory does not exist: " + newgen_src_dir)

    linear_src_dir = ""
    if options.linear_src:
        linear_src_dir = options.linear_src
    else:
        linear_src_dir = os.path.join(packages_dir, "PIPS/linear")
    p4a_util.debug("linear source directory: " + linear_src_dir)
    if not os.path.isdir(linear_src_dir) and not options.skip_linear:
        p4a_util.die("linear source directory does not exist: " + linear_src_dir)

    pips_src_dir = ""
    if options.pips_src:
        pips_src_dir = options.pips_src
    else:
        pips_src_dir = os.path.join(packages_dir, "PIPS/pips")
    p4a_util.debug("PIPS source directory: " + pips_src_dir)
    if not os.path.isdir(pips_src_dir) and not options.skip_pips:
        p4a_util.die("PIPS source directory does not exist: " + pips_src_dir)

    nlpmake_src_dir = ""
    if options.nlpmake_src:
        nlpmake_src_dir = options.nlpmake_src
    else:
        nlpmake_src_dir = os.path.join(packages_dir, "PIPS/nlpmake")
    # Normalize the directory so that we can build symbolic links to here
    # easily later:
    nlpmake_src_dir = os.path.abspath(nlpmake_src_dir)
    p4a_util.debug("nlpmake source directory: " + nlpmake_src_dir)
    #~ if not os.path.isdir(nlpmake_src_dir):
        #~ p4a_util.die("Directory does not exist: " + nlpmake_src_dir)

    # Global configure flags:
    configure_opts = [ "--prefix=" + prefix ]
    if options.configure_options:
        configure_opts.extend(options.configure_options)
    else:
        if options.debug:
            configure_opts.extend(default_debug_configure_opts)
        else:
            configure_opts.extend(default_configure_opts)

    # Global make flags:
    make_opts = []
    if options.make_options:
        make_opts.extend(options.make_options)
    if options.jobs:
        make_opts.append("-j" + options.jobs)

    #~ if get_verbosity() == 0:
        #~ p4a_util.warn("Building and installing", spin = True)

    ############################## polylib

    if not options.skip_polylib:

        p4a_util.info("Processing polylib")

        package_build_dir = os.path.join(build_dir, "polylib")

        # Rebuild requested? Delete existing build directory.
        if options.rebuild and os.path.isdir(package_build_dir):
                p4a_util.info("Package " + polylib_src_dir + " marked for rebuild, removing existing build dir " + package_build_dir)
                p4a_util.rmtree(package_build_dir)

        if not os.path.isdir(package_build_dir):
            os.makedirs(package_build_dir)

        polylib_conf_opts = configure_opts
        if options.polylib_conf_options:
            polylib_conf_opts.extend(options.polylib_conf_options)
        polylib_make_opts = make_opts
        if options.polylib_make_options:
            polylib_make_opts.extend(options.polylib_make_options)

        build_package(package_dir = polylib_src_dir, build_dir = package_build_dir,
            configure_opts = polylib_conf_opts, make_opts = polylib_make_opts, dest_dir = dest_dir,
            install = not options.no_install, reconf = options.reconf_polylib)

    ##############################

    # This was used for testing with DESTDIR...
    #configure_opts.extend ([
    #'POLYLIB64_CFLAGS="-I' + os.path.join(install_dir, "include") + '"',
    #'POLYLIB64_LIBS="-L' + os.path.join(install_dir, "lib") + ' -lpolylib64"',
    #'CFLAGS="-g -O2 -I' + os.path.join(install_dir, "include") + '"',
    #'CPPFLAGS="' + '-I' + os.path.join(install_dir, "include") + '"',
    #'LDFLAGS="-Wl,-z,defs -L' + os.path.join(install_dir, "lib") + '"'
    #])

    ############################## newgen

    if not options.skip_newgen:
        p4a_util.info("Processing newgen")

        package_build_dir = os.path.join(build_dir, "newgen")

        # Rebuild requested? Delete existing build directory.
        if options.rebuild and os.path.isdir(package_build_dir):
                p4a_util.info("Package " + newgen_src_dir + " marked for rebuild, removing existing build dir " + package_build_dir)
                p4a_util.rmtree(package_build_dir)

        if not os.path.isdir(package_build_dir):
            os.makedirs(package_build_dir)

        # Make a symlink to the old make infrastructure (and remove any existing one
        # or a symlink recursion will appear).
        p4a_util.run([ "rm", "-Rfv", os.path.join(newgen_src_dir, "makes") ])
        p4a_util.run([ "ln", "-sv", os.path.join(nlpmake_src_dir, "makes"), os.path.join(newgen_src_dir, "makes") ])

        newgen_conf_opts = configure_opts
        if options.newgen_conf_options:
            newgen_conf_opts.extend(options.newgen_conf_options)
        newgen_make_opts = make_opts
        if options.newgen_make_options:
            newgen_make_opts.extend(options.newgen_make_options)

        build_package(package_dir = newgen_src_dir, build_dir = package_build_dir,
            configure_opts = newgen_conf_opts, make_opts = newgen_make_opts, dest_dir = dest_dir,
            install = not options.no_install, reconf = options.reconf_newgen)

    ##############################

    # This was used for testing with DESTDIR...
    #configure_opts.extend ([ 'NEWGENLIBS_CFLAGS="-I' + os.path.join(install_dir, "include") + '"',
    #    'NEWGENLIBS_LIBS="-L' + os.path.join(install_dir, "lib") + ' -lnewgenlibs"' ])
    configure_opts.append ("PKG_CONFIG_PATH=" + p4a_util.quote(os.path.join(install_dir, "lib/pkgconfig")))

    ############################## linear

    if not options.skip_linear:
        p4a_util.info("Processing linear")

        package_build_dir = os.path.join(build_dir, "linear")

        # Rebuild requested? Delete existing build directory.
        if options.rebuild and os.path.isdir(package_build_dir):
                p4a_util.info("Package " + linear_src_dir + " marked for rebuild, removing existing build dir " + package_build_dir)
                p4a_util.rmtree(package_build_dir)

        if not os.path.isdir(package_build_dir):
            os.makedirs(package_build_dir)

        # Make a symlink to the old make infrastructure (and remove any existing one
        # or a symlink recursion will appear).
        p4a_util.run([ "rm", "-Rfv", os.path.join(linear_src_dir, "makes") ])
        p4a_util.run([ "ln", "-sv", os.path.join(nlpmake_src_dir, "makes"), os.path.join(linear_src_dir, "makes") ])
        linear_conf_opts = configure_opts

        if options.linear_conf_options:
            linear_conf_opts.extend(options.linear_conf_options)
        linear_make_opts = make_opts
        if options.linear_make_options:
            linear_make_opts.extend(options.linear_make_options)

        build_package(package_dir = linear_src_dir, build_dir = package_build_dir,
            configure_opts = linear_conf_opts, make_opts = linear_make_opts, dest_dir = dest_dir,
            install = not options.no_install, reconf = options.reconf_linear)

    ##############################

    # This was used for testing with DESTDIR...
    #configure_opts.extend([ 'LINEARLIBS_CFLAGS="-I' + os.path.join(install_dir, "include") + ' -DLINEAR_VALUE_IS_LONGLONG -DLINEAR_VALUE_PROTECT_MULTIPLY -DLINEAR_VALUE_ASSUME_SOFTWARE_IDIV"',
    #    'LINEARLIBS_LIBS="-L' + os.path.join(install_dir, "lib") + ' -llinearlibs"' ])
    #~ configure_opts.extend([ 'PATH="' + os.path.join(install_dir, "bin") + ':' + env("PATH") + '"',
        #~ 'LD_LIBRARY_PATH="' + os.path.join(install_dir, "lib") + ':' + env("LD_LIBRARY_PATH") + '"' ])

    # Update the PATH. Needed because PIPS relies on utilities built by newgen.
    p4a_util.add_to_path(os.path.join(install_dir, "bin"))

    # This was used for testing with DESTDIR...
    #p4a_util.add_to_path(os.path.join(install_dir, "lib"), var = "LD_LIBRARY_PATH")

    ############################## pips

    if not options.skip_pips:
        p4a_util.info("Processing pips")

        package_build_dir = os.path.join(build_dir, "pips")

        # Rebuild requested? Delete existing build directory.
        if options.rebuild and os.path.isdir(package_build_dir):
                p4a_util.info("Package " + pips_src_dir + " marked for rebuild, removing existing build dir " + package_build_dir)
                p4a_util.rmtree(package_build_dir)

        if not os.path.isdir(package_build_dir):
            os.makedirs(package_build_dir)

        # Make a symlink to the old make infrastructure (and remove any existing one
        # or a symlink recursion will appear).
        p4a_util.run([ "rm", "-Rfv", os.path.join(pips_src_dir, "makes") ])
        p4a_util.run([ "ln", "-sv", os.path.join(nlpmake_src_dir, "makes"), os.path.join(pips_src_dir, "makes") ])

        # Fix the following error:
        # /bin/sed: can't read /lib/libpolylib64.la: No such file or directory
        # libtool: link: `/lib/libpolylib64.la' is not a valid libtool archive
        # make[5]: *** [libpipslibs.la] Error 1
        # make[5]: Leaving directory `/home/gpean/p4a-foo/build/pips/src/Libs'
        #~ p4a_util.run([ "sudo", "ln", "-sfv", os.path.join(install_dir, "lib/libpolylib64.la"), os.path.join(safe_prefix, "lib/libpolylib64.la") ])

        ### FIX for fortran
        fortran = os.path.join(build_dir, "pips/src/Passes/fortran95")
        if not os.path.isdir(fortran):
            os.makedirs(fortran)
        # Copy with a rsync instead of simply symlinking the
        # source directory because the Fortran95 parser build
        # patches the sources and that would mark the files as
        # touched in the git repositiry (if any). Use --delete so
        # that if this setup is run again, the .files are removed
        # to relauch the patch.
        # If we do not build with Fortran 95 support, admit this can fail...
        version = "4.4.5"
        p4a_util.run([ "rsync", "-av", os.path.join(packages_dir, "pips-gfc/."), fortran ], can_fail = True)
        # To cheat the Makefile process that would like to
        # download the sources from the Internet:
        for file in [ "gcc-"+version+".md5", "gcc-core-"+version+".tar.bz2", "gcc-fortran-"+version+".tar.bz2" ]:
            p4a_util.run([ "touch", os.path.join(fortran, file) ])
        fortran2 = os.path.join(fortran, "gcc-"+version)
        if not os.path.isdir(fortran2):
            os.makedirs(fortran2)
        for file in [ ".dir", ".md5-check-core", ".md5-check-core.list",  ".md5-check-core.liste", ".md5-check-fortran", ".md5-check-fortran.liste", ".untar-core", ".untar-fortran", ".untar", ".patched" ]:
            p4a_util.run([ "touch", os.path.join(fortran2, file) ])
        ### End of FIX for fortran

        pips_conf_opts = configure_opts
        if options.pips_conf_options:
            pips_conf_opts.extend(options.pips_conf_options)
        else:
            global default_pips_conf_opts
            pips_conf_opts.extend(default_pips_conf_opts)
        pips_make_opts = make_opts
        if options.pips_make_options:
            pips_make_opts.extend(options.pips_make_options)

        build_package(package_dir = pips_src_dir, build_dir = package_build_dir,
            configure_opts = pips_conf_opts, make_opts = pips_make_opts, dest_dir = dest_dir,
            install = not options.no_install, reconf = options.reconf_pips)


    ##############################

    if options.no_final:
        p4a_util.warn("Skipping final installation steps (--no-final)")
        return


    # Proceed with local scripts and libraries installation.

    # Create directory tree.
    p4a_util.info("Creating dirs")
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
    install_dir_share_scmp = os.path.join(install_dir_share, "p4a_scmp")
    if not os.path.isdir(install_dir_share_scmp):
        os.makedirs(install_dir_share_scmp)
    install_dir_share_astrad = os.path.join(install_dir_share, "p4a_astrad")
    if not os.path.isdir(install_dir_share_astrad):
        os.makedirs(install_dir_share_astrad)
    install_dir_makes = os.path.join(install_dir, "makes")
    if not os.path.isdir(install_dir_makes):
        os.makedirs(install_dir_makes)
    install_dir_stubs = os.path.join(install_dir, "stubs")
    if not os.path.isdir(install_dir_stubs):
        os.makedirs(install_dir_stubs)

    # Install a few scripts.
    p4a_util.info("Installing scripts")

    for file in [
        "src/dev/p4a_git",
        "src/dev/p4a_valgrind",
        "src/simple_tools/p4a_process",
        "src/postprocessor/p4a_recover_includes",
        "src/validation/p4a_validate",
        "src/validation/p4a_validation",
        "src/p4a_accel/p4a_post_processor.py",
        "src/simple_tools/p4a_scpp"
        ]:
        p4a_util.run([ "cp", "-rv", "--remove-destination", os.path.join(root, file), install_dir_bin ])

    for file in [ "src/dev/p4a_git_lib.bash" ]:
        p4a_util.run([ "cp", "-rv", "--remove-destination", os.path.join(root, file), install_dir_etc ])

    # Install accelerator source.
    p4a_util.info("Installing accel files")
    accel_src_dir = os.path.join(root, "src/p4a_accel")
    for file in os.listdir(accel_src_dir):
        ext = os.path.splitext(file)[1]
        if ext == ".h" or ext == ".c" or ext == ".f" or ext == ".mk" or ext == ".cu" or ext == ".cpp" or ext == ".f95" :
            p4a_util.run([ "cp", "-rv", "--remove-destination", os.path.join(accel_src_dir, file), install_dir_share_accel ])

    # Install scmp source.
    p4a_util.info("Installing scmp files")
    scmp_src_dir = os.path.join(root, "src/scmp")
    for file in os.listdir(scmp_src_dir):
        ext = os.path.splitext(file)[1]
        if ext == ".h" or ext == ".c" or ext == ".f" or ext == ".arp" :
            p4a_util.run([ "cp", "-rv", "--remove-destination", os.path.join(scmp_src_dir, file), install_dir_share_scmp ])

    # Install astrad sources.
    p4a_util.info("Installing astrad files")
    astrad_src_dir = os.path.join(root, "src/astrad")
    for file in os.listdir(astrad_src_dir):
        ext = os.path.splitext(file)[1]
        if ext == ".h" or ext == ".c" or ext == ".f" or ext == ".arp" :
            p4a_util.run([ "cp", "-rv", "--remove-destination", os.path.join(astrad_src_dir, file), install_dir_share_astrad ])

    # Copy python dependencies and templates.
    p4a_util.info("Copying python libs")
    install_python_lib_dir = p4a_util.get_python_lib_dir(install_dir)
    #~ for file in os.listdir(install_dir_lib):
        #~ if file.startswith("python") and os.path.isdir(os.path.join(install_dir_lib, file)):
            #~ install_python_lib_dir = os.path.join(install_dir_lib, file, "site-packages/pips")
            #~ if not os.path.isdir(install_python_lib_dir):
                #~ install_python_lib_dir = os.path.join(install_dir_lib, file, "dist-packages/pips")
            #~ break
    #~ if not install_python_lib_dir:
        #~ p4a_util.die("Cannot not determine python lib dir in " + install_dir_lib + ", try --rebuild")
    dir = os.path.join(root, "src/simple_tools")
    for file in os.listdir(dir):
        ext = os.path.splitext(file)[1]
        if ext == ".py" or ext == ".tpl":
            p4a_util.run([ "cp", "-rv", "--remove-destination", os.path.join(dir, file), install_python_lib_dir ])

    dir = os.path.join(root, "src/scmp")
    for file in os.listdir(dir):
        ext = os.path.splitext(file)[1]
        if ext == ".py":
            p4a_util.run([ "cp", "-rv", "--remove-destination", os.path.join(dir, file), install_python_lib_dir ])

    dir = os.path.join(root, "src/astrad")
    for file in os.listdir(dir):
        ext = os.path.splitext(file)[1]
        if ext == ".py":
            p4a_util.run([ "cp", "-rv", "--remove-destination", os.path.join(dir, file), install_python_lib_dir ])

    # installing stubs !
    dir = os.path.join(root, "src/p4a_accel/stubs")
    for file in os.listdir(dir):
        p4a_util.run([ "cp", "-a", "--remove-destination", os.path.join(dir, file), install_dir_stubs ])

    # Create a shortcut name for binaries to the Python file, so that
    # we can type p4a instead of p4a.py. We need the .py versions
    # anyway since they can be imported by other Python programs
    # I guess it does not make sense on Windows... Should use os.symlink() RK
    for file in [ "p4a" ]:
        # Use a relative target for the link so that we can move all
        # install_dir_bin somewhere else and having it still working:
        p4a_util.run([ "ln", "-fs", os.path.relpath(os.path.join(install_python_lib_dir, file + '.py'), install_dir_bin), os.path.join(install_dir_bin, file) ])

    # Install stuff still lacking from PIPS install.
    if not options.skip_pips:
        p4a_util.info("Installing pips scripts")
        dir = os.path.join(pips_src_dir, "src/Scripts/validation")
        for file in os.listdir(dir):
            if file.startswith("pips"):
                p4a_util.run([ "cp", "-rv", "--remove-destination", os.path.join(dir, file), install_dir_bin ])
            p4a_util.run([ "cp", "-rv", "--remove-destination", os.path.join(pips_src_dir, "src/Scripts/misc/logfile_to_tpips"), install_dir_bin ])


    # Fix validation.
    if not options.skip_pips and not options.skip_linear and not options.skip_newgen:
        p4a_util.info("Fixing validation")
        dir = os.path.join(nlpmake_src_dir, "makes")
        for file in os.listdir(dir):
            if file == "arch.sh" or file == "version.sh":
                p4a_util.run([ "cp", "-rv", "--remove-destination", os.path.join(dir, file), install_dir_makes ])

    # Install various files.
    p4a_util.info("Installing release notes")
    p4a_util.run([ "cp", "-rv", "--remove-destination", os.path.join(root, "RELEASE-NOTES.rst"), install_dir ])
    p4a_util.info("Installing license")
    p4a_util.run([ "cp", "-rv", "--remove-destination", os.path.join(root, "LICENSE.txt"), install_dir ])
    if not options.skip_examples:
        p4a_util.info("Installing examples")
        p4a_util.run([ "cp", "-rv", "--remove-destination", os.path.join(root, "examples"), install_dir ])

    # Write the environment shell scripts.
    p4a_util.info("Writing shell rc files")
    fortran = ""
    if p4a_util.which("gfortran"):
        fortran = "gfortran"
    elif p4a_util.which("g77"):
        fortran = "g77"
    else:
        fortran = "false"
    accel_suffix=os.path.relpath(install_dir_share_accel,install_dir)
    scmp_suffix=os.path.relpath(install_dir_share_scmp,install_dir)
    astrad_suffix=os.path.relpath(install_dir_share_astrad,install_dir)
    p4a_rc.p4a_write_rc(install_dir_etc, dict(dist = install_dir,
        accel = accel_suffix, scmp = scmp_suffix, astrad = astrad_suffix, fortran = fortran))

    # Write version file.
    p4a_version.write_VERSION(install_dir, p4a_version.VERSION(root))
    p4a_version.write_GITREV(install_dir, p4a_version.GITREV(root))
    (revision, versiond) = p4a_version.make_full_revision(install_dir)

    p4a_util.done("")
    p4a_util.done("All done. Par4All " + revision + " is ready and has been installed in " + install_dir)
    p4a_util.done("To begin using it, you should source, depending on your shell religion:")
    p4a_util.done("")
    p4a_util.done("  " + os.path.join(install_dir, "etc/par4all-rc.sh") + " (for bash, dash, sh...) or")
    p4a_util.done("  " + os.path.join(install_dir, "etc/par4all-rc.csh") + " (tcsh, csh...)")
    p4a_util.done("")


def main():
    '''The function called when this program is executed by its own.'''

    parser = optparse.OptionParser(description = __doc__, usage = "%prog [options]; run %prog --help for options")

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
