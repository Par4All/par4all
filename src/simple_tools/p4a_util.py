#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#

'''
Par4All Common Utility Functions
'''

import string, sys, random, logging, logging.handlers, os, re, datetime, shutil, subprocess, time, tempfile, optparse, StringIO, fcntl, cPickle, glob
from threading import Thread
import p4a_term

def save_pickle(file, obj):
    f = open(file, "wb")
    cPickle.dump(obj, f)
    f.close()

def load_pickle(file):
    f = open(file, "rb")
    obj = cPickle.load(f)
    f.close()
    return obj

# Global variables.
verbosity = 0
logger = None
current_log_file = None

def get_current_log_file():
    global current_log_file
    return current_log_file

def change_file_ext(file, new_ext = None, if_ext = None):
    '''Changes the extension for the given file path if it matches if_ext.'''
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
    return get_file_extension(file)

def file_add_suffix(file, suffix):
    '''Adds a suffix to the given file (before its extension).'''
    (base, ext) = os.path.splitext(file)
    return base + suffix + ext

def set_verbosity(level):
    '''Sets global verbosity level'''
    global verbosity
    verbosity = level

def get_verbosity():
    '''Returns global verbosity level'''
    global verbosity
    return verbosity

(program_dir, program_name) = os.path.split(sys.argv[0])
program_name = change_file_ext(program_name, "")

def get_program_name():
    global program_name
    return program_name

def get_program_dir():
    global program_dir
    return program_dir

#~ all_spinners = []

#~ class spinner(Thread):

    #~ def __init__(self, start_it = True):
        #~ Thread.__init__(self)
        #~ self.stopped = True
        #~ global all_spinners
        #~ all_spinners.append(self)
        #~ if start_it:
            #~ self.start_spinning()

    #~ def start_spinning(self):
        #~ if not self.stopped:
            #~ return
        #~ # Not a tty? return now
        #~ if not os.isatty(2):
            #~ return
        #~ self.startt = time.time()
        #~ self.start()
        #~ self.stopped = False

    #~ def stop(self):
        #~ if not self.stopped:
            #~ self.stopped = True
            #~ self.join()

    #~ def __del__(self):
        #~ self.stop()

    #~ def run(self):
        #~ while time.time() - self.startt < 1:
            #~ time.sleep(0.05)
            #~ if self.stopped:
                #~ return
        #~ while not self.stopped:
            #~ for item in "-\|/":
                #~ sys.__stderr__.write("\rPlease wait... " + item)
                #~ time.sleep(0.05)
        #~ sys.__stderr__.write("\r")

#~ def stop_all_spinners():
    #~ global all_spinners
    #~ for spin in all_spinners:
        #~ spin.stop()

msg_prefix = program_name + ": "
#~ master_spin = None

debug_prefix = ""
debug_suffix = ""

#~ def stop_master_spinner():
    #~ global master_spin
    #~ if master_spin:
        #~ master_spin.stop()
        #~ master_spin = None

def debug(msg, log = True, bare = False, level = 2):
    global debug_prefix, debug_suffix, msg_prefix
    if get_verbosity() >= level:
        #~ stop_master_spinner()
        #~ if msg and msg[len(msg) - 1] == "\n":
            #~ msg = msg[0:len(msg) - 2]
        if bare:
            sys.__stderr__.write(str(msg).rstrip("\n") + "\n");
        else:
            sys.__stderr__.write(msg_prefix + debug_prefix + str(msg).rstrip("\n") + debug_suffix + "\n");
        #~ if spin:
            #~ master_spin = spinner()
    if logger and log:
        logger.debug(msg)

info_prefix = p4a_term.escape("white", if_tty_fd = 2)
info_suffix = p4a_term.escape(if_tty_fd = 2)

def info(msg, log = True, bare = False, level = 1):
    global info_prefix, info_suffix, msg_prefix
    if get_verbosity() >= level:
        if bare:
            sys.__stderr__.write(str(msg).rstrip("\n") + "\n");
        else:
            sys.__stderr__.write(msg_prefix + info_prefix + str(msg).rstrip("\n") + info_suffix + "\n");
    if logger and log:
        logger.info(msg)

def suggest(msg, level = 0):
    global msg_prefix
    if get_verbosity() >= level:
        sys.__stderr__.write(msg_prefix + str(msg).rstrip("\n") + "\n");

cmd_prefix = p4a_term.escape("magenta", if_tty_fd = 2)
cmd_suffix = p4a_term.escape(if_tty_fd = 2)

def cmd(msg, dir = None, log = True, bare = False, level = 1):
    global cmd_prefix, cmd_suffix, msg_prefix
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

done_prefix = p4a_term.escape("green", if_tty_fd = 2)
done_suffix = p4a_term.escape(if_tty_fd = 2)

def done(msg, log = True, bare = False, level = 0):
    global done_prefix, done_suffix, msg_prefix
    if get_verbosity() >= level:
        if bare:
            sys.__stderr__.write(str(msg).rstrip("\n") + "\n");
        else:
            sys.__stderr__.write(msg_prefix + done_prefix + str(msg).rstrip("\n") + done_suffix + "\n");
    if logger and log:
        logger.info(msg)

warn_prefix = p4a_term.escape("yellow", if_tty_fd = 2)
warn_suffix = p4a_term.escape(if_tty_fd = 2)

def warn(msg, log = True, bare = False, level = 0):
    global warn_prefix, warn_suffix, msg_prefix
    if get_verbosity() >= level:
        if bare:
            sys.__stderr__.write(str(msg).rstrip("\n") + "\n");
        else:
            sys.__stderr__.write(msg_prefix + warn_prefix + str(msg).rstrip("\n") + warn_suffix + "\n");
    if logger and log:
        logger.warn(msg)

error_prefix = p4a_term.escape("red", if_tty_fd = 2)
error_suffix = p4a_term.escape(if_tty_fd = 2)

def error(msg, log = True, bare = False, level = 0):
    global error_prefix, error_suffix, msg_prefix
    if get_verbosity() >= level:
        if bare:
            sys.__stderr__.write(str(msg).rstrip("\n") + "\n");
        else:
            sys.__stderr__.write(msg_prefix + error_prefix + str(msg).rstrip("\n") + error_suffix + "\n");
    if logger and log:
        logger.error(msg)

def die(msg, exit_code = 254, log = True, bare = False, level = 0):
    error(msg, log = log, bare = bare, level = level)
    sys.exit(exit_code)

default_log_file = os.path.join(os.getcwd(), program_name + ".log")
log_file_handler = None

def setup_logging(file = default_log_file, suffix = "", remove = False):
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
    warn("Logging to " + file)

def flush_log():
    if log_file_handler:
        log_file_handler.flush()

class p4a_error(Exception):
    '''Generic base class for exceptions'''
    def __init__(self, msg = "Generic error", code = 123):
        self.msg = msg
        self.code = code
    def __str__(self):
        #~ return self.msg + " (" + str(self.code) + ")"
        return self.msg

def read_file(file, text = True):
    '''Slurp file contents.'''
    f = None
    if text:
        f = open(file, "r")
    else:
        f = open(file, "rb")
    content = f.read()
    f.close()
    return content

def write_file(file, content, text = True):
    debug("Writing " + str(len(content)) + " bytes to " + file)
    f = None
    if text:
        f = open(file, "w")
    else:
        f = open(file, "wb")
    f.write(content)
    f.close()

def make_non_blocking(f):
    fd = f.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

class runner(Thread):

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
            debug("Running '" + self.cmd + "' in " + self.working_dir)
        else:
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
        self.can_spin = os.isatty(2)
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
                    self.out += new_err
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

    def wait(self):
        try:
            if self.redir:
                self.join()
            ret = self.process.wait()
            self.read_output(spin = False) # Read remaining output in pipes.
        except KeyboardInterrupt:
            raise p4a_error("Command '" + self.cmd + "' in " + self.working_dir
               + " interrupted", code = -2)
        if ret != 0 and not self.can_fail:
            #~ if self.err:
                #~ sys.stderr.write(self.err)
            #~ if not self.silent:
            debug("Environment was: " + repr(self.env))
            raise p4a_error("Command '" + self.cmd + "' in " + self.working_dir
                + " failed with exit code " + str(ret), code = ret)
        #~ stop_master_spinner()
        return [ self.out, self.err, ret ]

def run(cmd_list, can_fail = False, force_locale = "C", working_dir = None, 
        shell = True, extra_env = {}, silent = False, 
        stdout_handler = None, stderr_handler = None):
    if stdout_handler is None and stderr_handler is None:
        if silent:
            # Log output even in silent mode.
            stdout_handler = lambda s: debug(s, bare = True, level = 4)
            stderr_handler = lambda s: info(s, bare = True, level = 3)
        else:
            stdout_handler = lambda s: debug(s, bare = True)
            stderr_handler = lambda s: info(s, bare = True)
    r = runner(cmd_list, can_fail = can_fail, 
        force_locale = force_locale, working_dir = working_dir, extra_env = extra_env, 
        stdout_handler = stdout_handler, stderr_handler = stderr_handler, silent = silent)
    return r.wait()

def which(cmd, silent = True):
    return run([ "which", cmd ], can_fail = True, silent = True)[0]

def whoami(silent = True):
    return run([ "whoami" ], can_fail = True, silent = True)[0]

def hostname(silent = True):
    return run([ "hostname", "--fqdn" ], can_fail = True, silent = True)[0]

def ping(host, silent = True):
    return 0 == run([ "ping", "-w1", "-q", host ], can_fail = True, silent = True)[2]

def gen_name(length = 4, prefix = "P4A", suffix = "", chars = string.ascii_letters + string.digits):
    '''Generates a random name or password'''
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
    #~ debug("Removing tree: " + dir)
    ret = 0
    if remove_top:
        if run([ "rm", "-rf", dir + "/" ], can_fail = can_fail)[2]:
            warn("Could not remove " + dir + " recursively")
    else:
        if run([ "rm", "-rf", dir + "/*" ], can_fail = can_fail)[2]:
            warn("Could not remove everything in " + dir)
    #~ try:
        #~ for root, dirs, files in os.walk(dir, topdown = False):
            #~ for name in files:
                #~ os.remove(os.path.join(root, name))
            #~ for name in dirs:
                #~ os.rmdir(os.path.join(root, name))
        #~ if remove_top:
            #~ os.rmdir(dir)
    #~ except:
        #~ if can_fail:
            #~ warn("Could not remove directory " + dir + ": " + str(sys.exc_info()[1]))
        #~ else:
            #~ raise

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

#def get_python_lib_dir(dist_dir = None):
#	lib_dir = ""
#	if dist_dir:
#		lib_dir = os.path.join(dist_dir, "lib")
#	else:
#		global script_dir
#		return script_dir
#	python_dir = find(r"python\d\.\d", dir = dist_dir)
#	
#	for file in os.listdir(dist_dir):
#		if file.startswith("python") and os.path.isdir(os.path.join(install_dir_lib, file)):
#			install_python_lib_dir = os.path.join(install_dir_lib, file, "site-packages/pips")
#			if not os.path.isdir(install_python_lib_dir):
#				install_python_lib_dir = os.path.join(install_dir_lib, file, "dist-packages/pips")
#			break

def fortran_file_p(file):
    '''Tests if a file has a Fortran name.'''
    ext = get_file_extension(file)
    return ext == '.f' or ext == '.f77' or ext == '.f90' or ext == '.f95'

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

def get_machine_arch():
    '''Returns current machine architecture'''
    (sysname, nodename, release, version, machine) = os.uname()
    return machine

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
    return datetime.datetime.fromtimestamp(os.path.getmtime(file))

def utc_datetime():
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())

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

# XXX: make it cross platform
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

def quote(s):
    '''Quote the string if necessary and escape dangerous characters.
    In other words, make the string suitable for using in a shell command as a single argument.
    This function could be optimized a little.'''
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

def env(var, default = ""):
    if var in os.environ:
        return os.environ[var]
    else:
        return default

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
