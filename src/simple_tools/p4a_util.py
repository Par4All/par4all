#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#

'''
Par4All Common Utility Functions
'''

import string, sys, random, logging, os, re, datetime, shutil, subprocess
import term

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

# Printing/logging helpers.
def debug(msg):
    if verbosity >= 2:
        sys.stderr.write(sys.argv[0] + ": " + str(msg).rstrip("\n") + "\n");
    if logger:
        logger.debug(msg)

def info(msg):
    if verbosity >= 1:
        sys.stderr.write(sys.argv[0] + ": " + term.escape("white") + str(msg).rstrip("\n") + term.escape() + "\n");
    if logger:
        logger.info(msg)

def warn(msg):
    if verbosity >= 0:
        sys.stderr.write(sys.argv[0] + ": " + term.escape("yellow") + str(msg).rstrip("\n") + term.escape() + "\n");
    if logger:
        logger.warn(msg)

def error(msg):
    sys.stderr.write(sys.argv[0] + ": " + term.escape("red") + str(msg).rstrip("\n") + term.escape() + "\n");
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
        error(msg)
    def __str__(self):
        return self.msg

def run(cmd_list, can_fail = False, force_locale = "C", working_dir = None):
    '''Runs a command and dies if return code is not zero.
    NB: cmd_list must be a list with each argument to the program being an element of the list.'''
    if verbosity >= 1:
        sys.stderr.write(sys.argv[0] + ": " + term.escape("magenta") + " ".join(cmd_list) + term.escape() + "\n");
    old_locale = ""
    if force_locale is not None:
        if "LC_ALL" in os.environ:
            old_locale = os.environ["LC_ALL"]
        os.environ["LC_ALL"] = force_locale
    old_cwd = ""
    if working_dir:
        old_cwd = os.getcwd()
        os.chdir(working_dir)
    ret = os.system(" ".join(cmd_list))
    if old_cwd:
        os.chdir(old_cwd)
    if old_locale:
        os.environ["LC_ALL"] = old_locale
    if ret != 0 and not can_fail:
        raise p4a_error("command failed with exit code " + str(ret))
    return ret

def run2(cmd_list, can_fail = False, force_locale = "C", working_dir = None):
    '''Runs a command and dies if return code is not zero.
    Returns the final stdout and stderr output as a list.
    NB: cmd_list must be a list with each argument to the program being an element of the list.'''
    if verbosity >= 1:
        w = os.getcwd()
        if working_dir:
            w = working_dir
        sys.stderr.write(sys.argv[0] + ": (in " + w + ") " + term.escape("magenta") + " ".join(cmd_list) + term.escape() + "\n");
    old_locale = ""
    if force_locale is not None:
        if "LC_ALL" in os.environ:
            old_locale = os.environ["LC_ALL"]
        os.environ["LC_ALL"] = force_locale
    redir = subprocess.PIPE
    if verbosity >= 1:
        redir = None
    try:
        #print repr(os.environ)
        process = subprocess.Popen(" ".join(cmd_list), shell = True, 
            stdout = redir, stderr = redir, cwd = working_dir, env = os.environ)
    except:
        raise p4a_error("command '"+ " ".join(cmd_list)  +"' failed: " + str(sys.exc_info()))
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
    if old_locale:
        os.environ["LC_ALL"] = old_locale
    if ret != 0 and not can_fail:
        if err:
            error(err)
        raise p4a_error("command '"+ " ".join(cmd_list)  +"' failed with exit code " + str(ret))
    return [ out, err, ret ]

# Not portable!
def which(cmd):
    return run2([ "which", cmd ], can_fail = True)[0]

def gen_name(length = 4, prefix = "P4A", chars = string.letters + string.digits):
    '''Generates a random name or password'''
    return prefix + "".join(random.choice(chars) for x in range(length))

def rmtree(dir, can_fail = 0):
    '''Removes a directory recursively, alternative to shutil.rmtree()'''
    #(base, ext) = os.path.splitext(dir)
    #if ext != ".database" and ext != ".build":
    #    raise p4a_error("Cannot remove unknown directory: " + dir)
    debug("removing tree " + dir)
    try:
        for root, dirs, files in os.walk(dir, topdown = False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(dir)
    except:
        if can_fail:
            warn("could not remove directory " + dir + ": " + str(sys.exc_info()))
        else:
            raise e

def change_file_ext(file, new_ext):
    '''Changes the extension for the given file path'''
    (base, ext) = os.path.splitext(file)
    return base + new_ext

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
    debug("template " + template_file + " subsituted to " + output_file)
    return output_file

def file_lastmod(file):
    '''Returns file's last modification date/time.'''
    return datetime.datetime.fromtimestamp(os.path.getmtime(file))

def sh2csh(file, output_file = None):
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
        for v in os.environ[var].split(":"):
            if v != new_value:
                values += [ v ] 
    old_values = values
    if after:
        values += [ new_value ]
    else:
        values = [ new_value ] + values
    os.environ[var] = ":".join(values)
    return ":".join(old_values)

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
