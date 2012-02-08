#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#

'''
Par4All Common Utility Functions
'''

import string, sys, random, logging, logging.handlers, os, re, datetime, shutil
import subprocess, time, tempfile, optparse, StringIO, fcntl, cPickle, glob, platform, traceback
from threading import Thread
import p4a_term


# Disable all fancy output animation and coloring once and for all:
never_fancy = False

# Current state of fancyness, will be set depending on options in p4a_opts.process_common_options:
fancy = not never_fancy

# Global verbosity level:
verbosity = 0

# Logger instance to log to when something is output:
logger = None

# Log file associated with current logger instance:
current_log_file = None


def is_fancy():
    global fancy
    return fancy

def set_fancy(f):
    global never_fancy, fancy
    if not never_fancy:
        fancy = f
        p4a_term.disabled = not fancy


def get_current_log_file():
    '''Helper function which returns the current log file.'''
    global current_log_file
    return current_log_file


def change_file_ext(file, new_ext = None, if_ext = None):
    '''Changes the extension for the given file path if it matches if_ext.
    If if_ext is None, will change the extension every time. '''
    (base, ext) = os.path.splitext(file)
    if new_ext is None:
        new_ext = ""
    if if_ext:
        if ext == if_ext:
            return base + new_ext
        else:
            return file
    else:
        return base + new_ext

def get_file_extension(file):
    '''Returns the extension of the given file.'''
    return os.path.splitext(file)[1]

def get_file_ext(file):
    '''Alias for get_file_extension.'''
    return get_file_extension(file)

def file_add_suffix(file, suffix):
    '''Adds a suffix to the given file (before its extension).'''
    (base, ext) = os.path.splitext(file)
    return base + suffix + ext


def set_verbosity(level):
    '''Sets global verbosity level.'''
    global verbosity
    verbosity = level

def get_verbosity():
    '''Returns global verbosity level.'''
    global verbosity
    return verbosity


(program_dir, program_name) = os.path.split(sys.argv[0])
program_name = change_file_ext(program_name, "")

def get_program_name():
    '''Helper function which returns the calling program name (argv[0], see above).'''
    global program_name
    return program_name

def get_program_dir():
    '''Helper function which returns the calling program directory.'''
    global program_dir
    return program_dir


msg_prefix = program_name + ": "

def debug(msg, log = True, bare = False, level = 2):
    '''Log (and print out if verbosity is high enough (cf. level parameter))
    the passed message msg.'''
    global msg_prefix
    debug_prefix = ""
    debug_suffix = ""
    if get_verbosity() >= level:
        if bare:
            sys.__stderr__.write(str(msg).rstrip("\n") + "\n");
        else:
            sys.__stderr__.write(msg_prefix + debug_prefix + str(msg).rstrip("\n") + debug_suffix + "\n");
    if logger and log:
        logger.debug(msg)

def info(msg, log = True, bare = False, level = 1):
    '''Log (and print out if verbosity is high enough (cf. level parameter))
    the passed message msg.'''
    global msg_prefix
    info_prefix = p4a_term.escape("white", if_tty_fd = 2)
    info_suffix = p4a_term.escape(if_tty_fd = 2)
    if get_verbosity() >= level:
        if bare:
            sys.__stderr__.write(str(msg).rstrip("\n") + "\n");
        else:
            sys.__stderr__.write(msg_prefix + info_prefix + str(msg).rstrip("\n") + info_suffix + "\n");
    if logger and log:
        logger.info(msg)

def suggest(msg, level = 0):
    '''Suggest something at given level. This one never logs.'''
    global msg_prefix
    if get_verbosity() >= level:
        sys.__stderr__.write(msg_prefix + str(msg).rstrip("\n") + "\n");

def cmd(msg, dir = None, log = True, bare = False, level = 1):
    '''Log (and print out if verbosity is high enough (cf. level parameter))
    the passed message msg.
    This function is specific to external commands: it will print out the command
    in a specific color and will print the directory in which the command is run
    if verbosity is high enough.'''
    global msg_prefix
    cmd_prefix = p4a_term.escape("magenta", if_tty_fd = 2)
    cmd_suffix = p4a_term.escape(if_tty_fd = 2)
    if get_verbosity() >= level:
        if get_verbosity() > level and dir:
            if bare:
                sys.__stderr__.write("(in " + dir + ") " + str(msg).rstrip("\n") + "\n");
            else:
                sys.__stderr__.write(msg_prefix + "(in " + dir + ") " + cmd_prefix + str(msg).rstrip("\n") + cmd_suffix + "\n");
        else:
            if bare:
                sys.__stderr__.write(str(msg).rstrip("\n") + "\n");
            else:
                sys.__stderr__.write(msg_prefix + cmd_prefix + str(msg).rstrip("\n") + cmd_suffix + "\n");
    if logger and log:
        logger.info(msg)

def done(msg, log = True, bare = False, level = 0):
    '''Log (and print out if verbosity is high enough (cf. level parameter))
    the passed message msg.
    This function is specific to results (for displaying explicit "ok it's done" messages).'''
    #" <- for Emacs pretty-printer good behaviour
    global msg_prefix
    done_prefix = p4a_term.escape("green", if_tty_fd = 2)
    done_suffix = p4a_term.escape(if_tty_fd = 2)
    if get_verbosity() >= level:
        if bare:
            sys.__stderr__.write(str(msg).rstrip("\n") + "\n");
        else:
            sys.__stderr__.write(msg_prefix + done_prefix + str(msg).rstrip("\n") + done_suffix + "\n");
    if logger and log:
        logger.info(msg)

def warn(msg, log = True, bare = False, level = 0):
    '''Log (and print out if verbosity is high enough (cf. level parameter))
    the passed message msg.'''
    global msg_prefix
    warn_prefix = p4a_term.escape("yellow", if_tty_fd = 2)
    warn_suffix = p4a_term.escape(if_tty_fd = 2)
    if get_verbosity() >= level:
        if bare:
            sys.__stderr__.write(str(msg).rstrip("\n") + "\n");
        else:
            sys.__stderr__.write(msg_prefix + warn_prefix + str(msg).rstrip("\n") + warn_suffix + "\n");
    if logger and log:
        logger.warn(msg)

def error(msg, log = True, bare = False, level = 0):
    '''Log (and print out if verbosity is high enough (cf. level parameter))
    the passed message msg.'''
    global msg_prefix
    error_prefix = p4a_term.escape("red", if_tty_fd = 2)
    error_suffix = p4a_term.escape(if_tty_fd = 2)
    if get_verbosity() >= level:
        if bare:
            sys.__stderr__.write(str(msg).rstrip("\n") + "\n");
        else:
            sys.__stderr__.write(msg_prefix + error_prefix + str(msg).rstrip("\n") + error_suffix + "\n");
    if logger and log:
        logger.error(msg)

def die(msg, exit_code = 254, log = True, bare = False, level = 0):
    '''Issue an error() and then commit suicide.'''
    error(msg, log = log, bare = bare, level = level)
    sys.exit(exit_code)

def p4a_die_env(message):
    "Display a message and die with a message about misconfiguration"

    die(message + "\nIt looks like the Par4All environment has not been properly set.\n Have you sourced par4all-rc.sh?")

def add_list_to_set (l, s):
    """ add all elements of the list to the set"""
    for e in l:
        s.add (e)

default_log_file = os.path.join(os.getcwd(), program_name + ".log")
log_file_handler = None

def setup_logging(file = default_log_file, suffix = "", remove = False):
    '''Setup the logger instance so that all calls to debug, info, warn, etc.
    end up in a file, not only on the screen.'''
    global logger, program_name, current_log_file, log_file_handler
    logger = logging.getLogger(program_name)
    logger.setLevel(logging.DEBUG)
    if suffix:
        file = file_add_suffix(file, suffix)
    if remove and os.path.exists(file):
        os.remove(file)
    log_file_handler = logging.handlers.RotatingFileHandler(file, maxBytes = 20 * 1024 * 1024, backupCount = 10)
    log_file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logger.addHandler(log_file_handler)
    current_log_file = file
    logger.info("*" * 50)
    logger.info("*" * 50)
    logger.info("*" * 50)
    logger.info("Command: " + " ".join(sys.argv))
    logger.info("Python version: " + platform.python_version())
    warn("Logging to " + file)

def flush_log():
    '''Flush the log handler to disk.'''
    global log_file_handler
    if log_file_handler:
        log_file_handler.flush()

def skip_file_up_to_word(o_file,word,fold):
    """ Skip all the file lines until the <fold> occurence of <word>
    For instance, in the wrapper file, the signature is omitted thus
    it the file will be skipped until the second occurence of
    word="P4A_accel_kernel_wrapper"
    """
    n = 0
    src = open (o_file, 'r')
    content = ""
    for line in src:
        if re.search(word,line):
            n = n + 1
        if n == fold:
            content = content + str(line)
    src.close ()
    write_file(o_file,content)

def merge_files (dst_name, src_name_l):
    """ merge the sources file (given as a list) into the dst file. The content
    of the sources is appended to the destination
    """
    dst = open (dst_name, 'a')
    for name in src_name_l:
        src = open (name, 'r')
        for line in src:
            dst.write (line)
        src.close ()
    dst.close ()

class p4a_error(Exception):
    '''Generic base class for exceptions.'''
    def __init__(self, msg = "Generic error", code = 123):
        self.msg = msg
        self.code = code
    def __str__(self):
        #~ return self.msg + " (" + str(self.code) + ")"
        return self.msg


def read_file(file, text = True):
    '''"Slurp" (read whole) file contents.'''
    f = None
    if text:
        f = open(file, "r")
    else:
        f = open(file, "rb")
    content = f.read()
    f.close()
    return content

def write_file(file, content, text = True):
    '''Write some text or binary data to a file.
    This will overwrite any existing data in file!'''
    debug("Writing " + str(len(content)) + " bytes to " + file)
    f = None
    if text:
        f = open(file, "w")
    else:
        f = open(file, "wb")
    f.write(content)
    f.close()


def make_non_blocking(f):
    '''Make a file-like object non-blocking (reads will not block)
    until there is something to actually read). Useful for pipes.'''
    fd = f.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)


class runner(Thread):

    '''This class can be used to spawn an external command and
    capture its output. For a typical usage see the p4a_util.run() function.'''

    def __init__(self, cmd_list, can_fail = False, shell = True,
                 force_locale = "C", working_dir = None, extra_env = {},
                 stdout_handler = None, stderr_handler = None, silent = False):

        Thread.__init__(self)

        self.can_fail = can_fail

        if working_dir:
            self.working_dir = working_dir
        else:
            self.working_dir = os.getcwd()

        if force_locale is not None:
            extra_env["LC_ALL"] = force_locale

        self.env = os.environ
        for e in extra_env:
            self.env[e] = extra_env[e]

        self.cmd = " ".join(cmd_list)
        subp_cmd = None
        if shell:
            subp_cmd = self.cmd
        else:
            subp_cmd = cmd_list

        self.silent = silent

        if self.silent:
            # Output only debugging launch info in silent mode:
            debug("Running '" + self.cmd + "' in " + self.working_dir, level = 3)
        else:
            # Display that a command will be run:
            cmd(self.cmd, dir = self.working_dir)

        try:
            self.process = subprocess.Popen(subp_cmd, shell = shell,
                stdout = subprocess.PIPE, stderr = subprocess.PIPE,
                cwd = self.working_dir, env = self.env)
        except:
            if not can_fail:
                debug("Environment was: " + repr(env))
                raise p4a_error("Command '" + " ".join(cmd_list) + "' in " + w + " failed: "
                    + str(sys.exc_info()[1]))

        self.stdout_handler = stdout_handler
        self.stderr_handler = stderr_handler

        self.redir = (stdout_handler or stderr_handler)

        self.out = ""
        self.err = ""

        self.out_line_chunk = ""
        self.err_line_chunk = ""

        make_non_blocking(self.process.stdout)
        make_non_blocking(self.process.stderr)
        self.spin_text = "-=-          "
        #spin_text = ".·°·..·°·..·°·."
        self.spin_pos = len(self.spin_text)
        self.spin_back = False
        self.spin_after = .8
        #~ self.can_spin = not self.silent and os.isatty(2)
        self.can_spin = is_fancy() and os.isatty(2)
        self.startt = time.time()

        if self.redir:
            self.start()

    def running(self):
        return self.process.poll() == None

    def inc_spinner(self):
        if not self.can_spin or time.time() - self.startt < self.spin_after:
            return
        sys.__stderr__.write("\r" + self.spin_text[self.spin_pos:] + self.spin_text[0:self.spin_pos] + "")
        if self.spin_back:
            self.spin_pos = self.spin_pos - 1
        else:
            self.spin_pos = self.spin_pos + 1
        if self.spin_pos >= len(self.spin_text):
            self.spin_back = True
        elif self.spin_pos <= 3:
            self.spin_back = False

    def hide_spinner(self):
        if not self.can_spin or time.time() - self.startt < self.spin_after:
            return
        sys.__stderr__.write("\r")
        self.startt = time.time()

    def read_output(self, spin = True):
        try:
            while True:
                new_out = self.process.stdout.read()
                lines = new_out.split("\n")
                lines[0] = self.out_line_chunk + lines[0]
                self.out_line_chunk = lines.pop()
                new_out = "\n".join(lines)
                if new_out:
                    self.out += new_out
                    if self.stdout_handler:
                        if spin:
                            self.hide_spinner()
                        for line in lines:
                            self.stdout_handler(line)
                else:
                    break
        except IOError:
            e = sys.exc_info()[1]
            if e.errno != 11:
                raise
        try:
            while True:
                new_err = self.process.stderr.read()
                lines = new_err.split("\n")
                lines[0] = self.err_line_chunk + lines[0]
                self.err_line_chunk = lines.pop()
                new_err = "\n".join(lines)
                if new_err:
                    self.err += new_err
                    if self.stderr_handler:
                        if spin:
                            self.hide_spinner()
                        for line in lines:
                            self.stderr_handler(line)
                else:
                    break
        except IOError:
            e = sys.exc_info()[1]
            if e.errno != 11:
                raise
        if spin:
            self.inc_spinner()

    def run(self):
        self.inc_spinner()
        while self.running():
            self.read_output()
            time.sleep(0.05)
        self.hide_spinner()

    def wait(self, error_code = 0, msg = ""):
        try:
            if self.redir:
                self.join()
            ret = self.process.wait()
            self.read_output(spin = False) # Read remaining output in pipes.
        except KeyboardInterrupt:
            raise p4a_error("Command '" + self.cmd + "' in " + self.working_dir
               + " interrupted", code = -2)
        if ret != 0 and not self.can_fail:
            if self.err and get_verbosity() == 0:
                sys.stderr.write(self.err + "\n")
            self.stderr_handler("Environment was: " + repr(self.env))
            if ret == error_code:
				raise p4a_error("Command '" + self.cmd + "' in " + self.working_dir
					+ " failed with exit code " + str(ret) + msg, code = ret)
            else:
				raise p4a_error("Command '" + self.cmd + "' in " + self.working_dir
					+ " failed with exit code " + str(ret), code = ret)
		#~ stop_master_spinner()
        return [ self.out, self.err, ret ]


def run(cmd_list, can_fail = False, force_locale = "C", working_dir = None,
        shell = True, extra_env = {}, silent = False,
        stdout_handler = None, stderr_handler = None,
        error_code = 0, retry = 1, msg = ""):
    '''Helper function to spawn an external command and wait for it to finish.'''
    if stdout_handler is None and stderr_handler is None:
        if silent:
            # Log output even in silent mode.
            stdout_handler = lambda s: debug(s, bare = True, level = 4)
            stderr_handler = lambda s: info(s, bare = True, level = 3)
        else:
            stdout_handler = lambda s: debug(s, bare = True)
            stderr_handler = lambda s: info(s, bare = True)

    # To gather the output on multiple iterations:
    out = ""
    err = ""
    # For funny case we do not iterate at all:
    ret = 0
    # Iterate on the command if asked to insist:
    for i in range(1, retry + 1):
        # On the last retry, command failure launches an exception, if not
        # allowed to fail:
        i_can_fail = (i != retry) | can_fail
        r = runner(cmd_list, can_fail = i_can_fail,
                   force_locale = force_locale, working_dir = working_dir,
                   extra_env = extra_env, stdout_handler = stdout_handler,
                   stderr_handler = stderr_handler, silent = silent)
        (i_out, i_err, ret) = r.wait(error_code, msg)
        # Keep track of the command output:
        out += i_out
        err += i_err
        if ret == 0:
            # No error during the execution, so stop retrying:
            break
    # Return all the outputs but only the last return code error:
    return (out, err, ret)


def which(cmd, silent = True):
    '''Calls the "which" UNIX utility for the given program.'''
    return run([ "which", cmd ], can_fail = True, silent = silent, force_locale = None)[0].rstrip("\n")

def whoami(silent = True):
    '''Calls the whoami UNIX utility.'''
    return run([ "whoami" ], can_fail = True, silent = silent)[0].rstrip("\n")

def hostname(silent = True): # platform.node()???
    '''Calls the hostname UNIX utility.'''
    return run([ "hostname", "--fqdn" ], can_fail = True, silent = silent)[0].rstrip("\n")

def ping(host, silent = True):
    '''Calls the ping utility. Returns True if remote host answers within 1 second.'''
    return 0 == run([ "ping", "-w1", "-q", host ], can_fail = True, silent = silent)[2]


def get_distro():
    '''Returns currently running Linux distribution name (ubuntu, debian,
    redhat, etc.) always in lower case.'''
    return str.lower(platform.linux_distribution()[0])


def pkg_config(dist_dir, variable):
    pkg_config_path = os.path.join(dist_dir, "lib", "pkgconfig")
    if not os.path.isdir(pkg_config_path):
        pkg_config_path = os.path.join(dist_dir, "lib64", "pkgconfig")
        if not os.path.isdir(pkg_config_path):
            raise p4a_error("Could not determine PKG_CONFIG_PATH in " + dist_dir + ", try reinstalling Par4All")
    return run([ "pkg-config", "pips", "--variable=" + variable ], extra_env = dict(PKG_CONFIG_PATH = pkg_config_path))[0].rstrip("\n")


def get_python_lib_dir(dist_dir):
    dir = pkg_config(dist_dir, "pkgpythondir")
    if not dir or not os.path.isdir(dir):
        raise p4a_error("Could not determine Python modules installation path in " + dist_dir + ", try reinstalling Par4All")
    return dir

def get_machine_arch():
    return platform.machine()

def gen_name(length = 4, prefix = "P4A", suffix = "", chars = string.ascii_letters + string.digits):
    '''Generates a random name or password.'''
    return prefix + "".join(random.choice(chars) for x in range(length)) + suffix


def is_system_dir(dir):
    '''Returns True if dir is a system directory (any directory which matters to the system).'''
    for s in [ "/", "/boot", "/etc", "/dev", "/opt", "/sys", "/srv", "/proc", "/usr", "/home",
        "/var", "/lib", "/lib64", "/sbin", "/bin", "/root", "/tmp",
        "/var/backups", "/var/cache", "/var/crash", "/var/games", "/var/lib", "/var/local", "/var/lock", "/var/log", "/var/mail", "/var/opt", "/var/run", "/var/spool", "/var/tmp",
        "/usr/bin", "/usr/etc", "/usr/include", "/usr/lib", "/usr/sbin", "/usr/share", "/usr/src", "/usr/local",
        "/usr/local/bin", "/usr/local/etc", "/usr/local/games", "/usr/local/include", "/usr/local/lib", "/usr/local/man", "/usr/local/sbin", "/usr/local/share", "/usr/local/src" ]:
        if dir == s:
            return True
    return False


def rmtree(dir, can_fail = False, remove_top = True):
    '''Removes a directory recursively, alternative to shutil.rmtree()'''
    dir = os.path.realpath(os.path.abspath(os.path.expanduser(dir)))
    if not os.path.isdir(dir):
        if can_fail:
            return
        raise p4a_error("Directory does not exist: " + dir)
    if not remove_top and not glob.glob(os.path.join(dir, "*")):
        return
    if is_system_dir(dir): # Prevent deletion of major system dirs...
        raise p4a_error("Will not remove protected directory: " + dir)
    ret = 0
    if remove_top:
        if run([ "rm", "-rf", dir + "/" ], can_fail = can_fail)[2]:
            warn("Could not remove " + dir + " recursively")
    else:
        if run([ "rm", "-rf", dir + "/*" ], can_fail = can_fail)[2]:
            warn("Could not remove everything in " + dir)


def find(file_re, dir = None, abs_path = True, match_files = True,
    match_dirs = False, match_whole_path = False, can_fail = True):
    '''Lookup files matching the regular expression file_re underneath dir.
    If dir is empty, os.getcwd() will be looked up.
    If full_path is true, absolute path names of matching file/dir names will be returned.
    If match_whole_path is True, whole paths will be tested against file_re.'''
    matches = []
    compiled_file_re = re.compile(file_re)
    if dir:
        if not os.path.isdir(dir):
            raise p4a_error("Invalid directory: " + dir)
    else:
        dir = os.getcwd()
    dir = os.path.abspath(os.path.expanduser(dir))
    #debug("Looking for files matching '" + file_re + "' in " + dir)
    try:
        for root, dirs, files in os.walk(dir, topdown = False):
            files_dirs = []
            if match_files:
                files_dirs += files
            if match_dirs:
                files_dirs += dirs
            for name in files:
                whole_path = os.path.join(root, name)
                matched = False
                if match_whole_path:
                    if compiled_file_re.match(whole_path):
                        matched = True
                else:
                    if compiled_file_re.match(name):
                        matched = True
                if matched:
                    if abs_path:
                        matches += [ whole_path ]
                    else:
                        matches += [ whole_path[len(dir):] ]
    except:
        if not can_fail:
            raise e
    return matches


def fortran_file_p(file):
    '''Tests if a file has a Fortran name.'''
    ext = get_file_extension(file)
    return ext == '.f' or ext == '.f77' or ext == '.f90' or ext == '.f95' or ext == '.f03' or ext == '.f08' or ext == '.F'

def c_file_p(file):
    '''Tests if a file has a C name.'''
    ext = get_file_extension(file)
    return ext == '.c'

def cxx_file_p(file):
    '''Tests if a file has a C++ name.'''
    ext = get_file_extension(file)
    return ext == '.cpp' or ext == '.cxx'
def cpp_file_p(file):
    return cxx_file_p(file)

def cuda_file_p(file):
    '''Tests if a file has a CUDA name.'''
    ext = get_file_extension(file)
    return ext == '.cu'

def opencl_file_p(file):
    '''Tests if a file has a OPENCL name.'''
    ext = get_file_extension(file)
    return ext == '.cl'

def sharedlib_file_p(file):
    '''Tests if a file has a shared library name.'''
    ext = get_file_extension(file)
    return ext == '.so'

def lib_file_p(file):
    '''Tests if a file has a static library name.'''
    ext = get_file_extension(file)
    return ext == '.a'

def exe_file_p(file):
    '''Tests if a file has an executable binary name.'''
    ext = get_file_extension(file)
    return ext == ''

def header_file_p(file):
    '''Tests if a file has an header name.'''
    ext = get_file_extension(file)
    return ext == '.h' or ext == '.hpp'


def subs_template_file(template_file, map = {}, output_file = None, trim_tpl_ext = True):
    '''Substitute keys with values from map in template designated by template_file.
    output_file can be empty, in which case the original template will be overwritten with the substituted file.
    It can also be a directory, in which case the name of the original template file is kept.'''
    content = string.Template(read_file(template_file)).substitute(map)
    if not output_file:
        output_file = template_file
    elif os.path.isdir(output_file):
        output_file = os.path.join(output_file, os.path.split(template_file)[1])
    write_file(output_file, content)
    if trim_tpl_ext:
        (base, ext) = os.path.splitext(output_file)
        if ext == ".tpl":
            shutil.move(output_file, base)
            output_file = base
    debug("Template " + template_file + " subsituted to " + output_file)
    return output_file


def file_lastmod(file):
    '''Returns file's last modification date/time.'''
    #'
    return datetime.datetime.fromtimestamp(os.path.getmtime(file))

def utc_datetime(sep = "T", dsep = "", tsep = ""):
    '''Returns the current UTC date/time in ISO format.'''
    return time.strftime("%Y" + dsep + "%m" + dsep + "%d" + sep + "%H" + tsep + "%M" + tsep + "%S", time.gmtime())

def utc_date(dsep = "-"):
    '''Returns the current UTC date.'''
    return time.strftime("%Y" + dsep + "%m" + dsep + "%d", time.gmtime())


def sh2csh(file, output_file = None):
    '''Attempts to convert a sh file to csh.'''
    if not output_file:
        output_file = change_file_ext(file, ".csh")
    content = read_file(file)
    # XXX: probably more to filter out (if ... else etc.)
    content = re.sub("export\s+(\S+?)\s*=\s*(.+?)(\n?)(;?)", "setenv \\1 \\2\\3\\4", content)
    content = re.sub("(\S+?)\s*=\s*(.+?)(\n?)(;?)", "set \\1=\\2\\3\\4", content)
    content += "\n\nrehash\n";
    write_file(output_file, content)


def add_to_path(new_value, var = "PATH", after = False):
    '''Adds a new value to the PATH environment variable (or any other var working the same way).
    Returns the previous whole value for the variable.'''
    values = []
    if var in os.environ:
        for v in os.environ[var].split(os.pathsep):
            if v != new_value:
                values += [ v ]
    old_values = values
    if after:
        values += [ new_value ]
    else:
        values = [ new_value ] + values
    os.environ[var] = os.pathsep.join(values)
    debug("New " + var + " value: " + os.environ[var])
    return os.pathsep.join(old_values)

def env(var, default = ""):
    '''Helper to return a variable environment value if it is defined.'''
    if var in os.environ:
        return os.environ[var]
    else:
        return default


def quote(s):
    '''Quote the string if necessary and escape dangerous characters.
    In other words, make the string suitable for using in a shell command as a single argument.
    This function could be optimized a little.

    I guess this function already exists...
    '''
    if not s:
        return '""'
    enclose = False
    if s.find(" ") >= 0: # and previous characters is not \\ ...
        enclose = True
    if s.find("\\") >= 0:
        s = s.replace("\\", "\\\\")
        enclose = True
    if s.find('"') >= 0:
        s = s.replace('"', '\\"')
        enclose = True
    if s.find('`') >= 0:
        s = s.replace('`', '\\`')
        enclose = True
    if s.find('$') >= 0:
        s = s.replace('$', '\\$')
        enclose = True
    if s.find('!') >= 0:
        s = s.replace('!', '\\!')
        enclose = True
    if enclose:
        return '"' + s + '"'
    else:
        return s


def relativize(file_dir = None, dirs = [], base = os.getcwd()):
    '''Make a file or directory relative to the base directory,
    if they start with this base directory (this is the difference
    with os.path.relpath()).'''
    # Make sure it is absolute first:
    file_dir = os.path.abspath(os.path.expanduser(file_dir))
    if file_dir.startswith(base) and len(file_dir) > len(base):
        file_dir = file_dir[len(base):]
        if file_dir[0] == os.path.sep:
            file_dir = file_dir[1:]
    return file_dir


def save_pickle(file, obj):
    '''Serizializes/marshalls object in obj in file.'''
    f = open(file, "wb")
    cPickle.dump(obj, f)
    f.close()

def load_pickle(file):
    '''Deserializes/unmarshalls object contained in file,
    previously serialized using pickle or cPickle modules.
    Returns the new object.'''
    f = open(file, "rb")
    obj = cPickle.load(f)
    f.close()
    return obj

def generate_c_header (in_c_file, out_h_file, additional_args= []):
    """ generate the header file for a given c file produced by p4a using
    cproto.
    in_c_file : The c file to be processed by cproto
    out_h_file : The h file to be produced  by cproto
    additional_ args : The additional arguments to the cproto command (for
    example specific defines)
    """
    args = ["cproto"]
    args.append ("-I")
    args.append (os.environ["P4A_ACCEL_DIR"])
    args.extend (additional_args)
    args.append ("-o")
    args.append (out_h_file)
    args.append (in_c_file)
    run (args, force_locale = None)


def quote_fname(fname):
    """quote file name or any string to avoid problems with file/directory names
	that contain spaces or any other kind of nasty characters
	"""
    return '"%s"' % (
		fname
		.replace('\\', '\\\\')
		.replace('"', '\"')
		.replace('$', '\$')
		.replace('`', '\`')
		.replace('!', '\!')
		)

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
