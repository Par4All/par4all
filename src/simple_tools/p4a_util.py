#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#

'''
Par4All Common Utility Functions
'''

import string, sys, random, logging, os, re, datetime, shutil, subprocess, time
from threading import Thread
import p4a_term

# Global variables.
verbosity = 0
logger = None

def set_verbosity(level):
    '''Sets global verbosity level'''
    global verbosity
    verbosity = level

def get_verbosity():
    '''Returns global verbosity level'''
    global verbosity
    return verbosity

msg_prefix = os.path.split(sys.argv[0])[1] + ": "

# Printing/logging helpers.
def debug(msg):
    global msg_prefix
    if verbosity >= 2:
        sys.stderr.write(msg_prefix + str(msg).rstrip("\n") + "\n");
    if logger:
        logger.debug(msg)

def info(msg):
    global msg_prefix
    if verbosity >= 1:
        sys.stderr.write(msg_prefix + p4a_term.escape("white") + str(msg).rstrip("\n") + p4a_term.escape() + "\n");
    if logger:
        logger.info(msg)

def done(msg):
    global msg_prefix
    if verbosity >= 0:
        sys.stderr.write(msg_prefix + p4a_term.escape("green") + str(msg).rstrip("\n") + p4a_term.escape() + "\n");
    if logger:
        logger.info(msg)

def warn(msg):
    global msg_prefix
    if verbosity >= 0:
        sys.stderr.write(msg_prefix + p4a_term.escape("yellow") + str(msg).rstrip("\n") + p4a_term.escape() + "\n");
    if logger:
        logger.warn(msg)

def error(msg):
    global msg_prefix
    sys.stderr.write(msg_prefix + p4a_term.escape("red") + str(msg).rstrip("\n") + p4a_term.escape() + "\n");
    if logger:
        logger.error(msg)

def die(msg, exit_code = 255):
    error(msg)
    #error("aborting")
    sys.exit(exit_code)

class p4a_error(Exception):
    '''Generic base class for exceptions'''
    msg = "error"
    def __init__(self, msg):
        self.msg = msg
        #error(msg)
    def __str__(self):
        return self.msg

spinners = []

class spinner(Thread):

    def __init__(self):
        Thread.__init__(self)
        self.flag = False
        self.startt = time.time()
        global spinners
        spinners.append(self)
        self.start()

    def stop(self):
        if not self.flag:
            self.flag = True
            self.join()
    
    #~ def __del__(self):
        #~ self.stop()

    def run(self):
        while time.time() - self.startt < 1:
            time.sleep(0.05)
            if self.flag:
                return
        #~ sys.stderr.write("\n")
        while not self.flag:
            for item in "-\|/":
                sys.stderr.write("\b\b" + item)
                time.sleep(0.05)
        sys.stderr.write("\b\b")

def stop_all_spinners():
    global spinners
    for spin in spinners:
        spin.stop()

def run(cmd_list, can_fail = False, force_locale = "C", working_dir = None, extra_env = {}):
    '''Runs a command and dies if return code is not zero.
    NB: cmd_list must be a list with each argument to the program being an element of the list.'''
    if verbosity >= 1:
        sys.stderr.write(sys.argv[0] + ": " + p4a_term.escape("magenta") 
            + " ".join(cmd_list) + p4a_term.escape() + "\n");
    if force_locale is not None:
        extra_env["LC_ALL"] = force_locale
    prev_env = {}
    for e in extra_env:
        if e in os.environ:
            prev_env[e] = os.environ[e]
        else:
            prev_env[e] = ""
        os.environ[e] = extra_env[e]
    old_cwd = ""
    w = ""
    try:
        if working_dir:
            old_cwd = os.getcwd()
            os.chdir(working_dir)
            w = working_dir
        else:
            w = os.getcwd()
        ret = os.system(" ".join(cmd_list))
        if old_cwd:
            os.chdir(old_cwd)
    except:
        if not can_fail:
            raise p4a_error("Command '" + " ".join(cmd_list) + "' in " + w + " failed: " 
                + str(sys.exc_info()[1]) + " (env = " + repr(os.environ) + ")")
    for e in prev_env:
        if e in os.environ:
            if len(e):
                os.environ[e] = prev_env[e]
            else:
                del os.environ[e]
    if ret != 0 and not can_fail:
        raise p4a_error("Command '" + " ".join(cmd_list) + "' in " + w 
            + " failed with exit code " + str(ret) + " (env = " + repr(os.environ) + ")")
    return ret

def run2(cmd_list, can_fail = False, force_locale = "C", working_dir = None, shell = True, capture = False, extra_env = {}):
    '''Runs a command and dies if return code is not zero.
    Returns the final stdout and stderr output as a list.
    NB: cmd_list must be a list with each argument to the program being an element of the list.'''
    #if cmd_list.__class__.__name__ != "list":
    #    cmd_list = [ cmd_list ]
    w = ""
    if working_dir:
        w = working_dir
    else:
        w = os.getcwd()
    if verbosity >= 1:
        sys.stderr.write(sys.argv[0] + ": (in " + w + ") " + p4a_term.escape("magenta") 
            + " ".join(cmd_list) + p4a_term.escape() + "\n");
    if force_locale is not None:
        extra_env["LC_ALL"] = force_locale
    env = os.environ
    for e in extra_env:
        env[e] = extra_env[e]    
    redir = subprocess.PIPE
    if verbosity >= 2 and not capture:
        redir = None
    spin = None
    if redir and verbosity >= 1: # Display a spinner if we are hiding output and we displayed command.
        spin = spinner()
    try:
        if shell:
            process = subprocess.Popen(" ".join(cmd_list), shell = True, 
                stdout = redir, stderr = redir, cwd = working_dir, env = env)
        else:
            process = subprocess.Popen(cmd_list, shell = False, 
                stdout = redir, stderr = redir, cwd = working_dir, env = env)
    except:
        if not can_fail:
            raise p4a_error("Command '" + " ".join(cmd_list) + "' in " + w + " failed: " 
                + str(sys.exc_info()[1]) + " (env = " + repr(env) + ")")
    out = ""
    err = ""
    while True:
        try:
            new_out, new_err = process.communicate()
            out += new_out
            err += new_err
        except:
            break
    ret = process.wait()
    if spin is not None:
        spin.stop()
    if ret != 0 and not can_fail:
        if err:
            error(err)
        raise p4a_error("Command '" + " ".join(cmd_list) + "' in " + w 
            + " failed with exit code " + str(ret) + " (env = " + repr(env) + ")")
    return [ out, err, ret ]

# Not portable!
def which(cmd):
    return run2([ "which", cmd ], can_fail = True, capture = True)[0]

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

def rmtree(dir, can_fail = 0, remove_top = True):
    '''Removes a directory recursively, alternative to shutil.rmtree()'''
    dir = os.path.abspath(os.path.realpath(os.path.expanduser(dir)))
    if not dir or not os.path.isdir(dir):
        raise p4a_error("Not a directory: " + dir)
    if is_system_dir(dir): # Prevent deletion of major system dirs...
        raise p4a_error("Will not remove protected directory: " + dir)
    debug("Removing tree: " + dir)
    try:
        for root, dirs, files in os.walk(dir, topdown = False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        if remove_top:
            os.rmdir(dir)
    except:
        if can_fail:
            warn("Could not remove directory " + dir + ": " + str(sys.exc_info()[1]))
        else:
            raise

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
    dir = os.path.abspath(os.path.realpath(os.path.expanduser(dir)))
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

def slurp(file):
    '''Slurp file contents.'''
    f = open(file)
    content = f.read()
    f.close()
    return content

def dump(file, content):
    '''Dump contents to file.'''
    debug("Writing " + str(len(content)) + " bytes to " + file)
    f = open(file, "w")
    f.write(content)
    f.close()

def subs_template_file(template_file, map = {}, output_file = None, trim_tpl_ext = True):
    '''Substitute keys with values from map in template designated by template_file.
    output_file can be empty, in which case the original template will be overwritten with the substituted file.
    It can also be a directory, in which case the name of the original template file is kept.'''
    content = string.Template(slurp(template_file)).substitute(map)
    if not output_file:
        output_file = template_file
    elif os.path.isdir(output_file):
        output_file = os.path.join(output_file, os.path.split(template_file)[1])
    dump(output_file, content)
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

def sh2csh(file, output_file = None):
    '''Attempts to convert a sh file to csh.'''
    if not output_file:
        output_file = change_file_ext(file, ".csh")
    content = slurp(file)
    # XXX: probably more to filter out (if ... else etc.)
    content = re.sub("export\s+(\S+?)\s*=\s*(.+?)(\n?)(;?)", "setenv \\1 \\2\\3\\4", content)
    content = re.sub("(\S+?)\s*=\s*(.+?)(\n?)(;?)", "set \\1=\\2\\3\\4", content)
    content += "\n\nrehash\n";
    dump(output_file, content)

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
