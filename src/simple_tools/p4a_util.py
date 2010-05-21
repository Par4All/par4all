#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Par4All Common Utility Functions
'''

import string, sys, random, logging, os, re
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
		sys.stderr.write(sys.argv[0] + ": " + msg.rstrip("\n") + "\n");
	if logger:
		logger.debug(msg)

def info(msg):
	if verbosity >= 1:
		sys.stderr.write(sys.argv[0] + ": " + term.escape("white") + msg.rstrip("\n") + term.escape() + "\n");
	if logger:
		logger.info(msg)

def warn(msg):
	if verbosity >= 0:
		sys.stderr.write(sys.argv[0] + ": " + term.escape("yellow") + msg.rstrip("\n") + term.escape() + "\n");
	if logger:
		logger.warn(msg)

def error(msg):
	sys.stderr.write(sys.argv[0] + ": " + term.escape("red") + msg.rstrip("\n") + term.escape() + "\n");
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

def run(cmd, can_fail = 0):
	'''Runs a command and dies if return code is not zero'''
	if verbosity >= 0:
		sys.stderr.write(sys.argv[0] + ": " + term.escape("magenta") + cmd + term.escape() + "\n");
	ret = os.system(cmd)
	if ret != 0 and not can_fail:
		raise p4a_error("command failed with exit code " + str(ret))
	return ret

def gen_name(length = 4, prefix = "P4A", chars = string.letters + string.digits):
	'''Generates a random name or password'''
	return prefix + "".join(random.choice(chars) for x in range(length))

def rmtree(dir, can_fail = 0):
	'''Removes a directory recursively, alternative to shutil.rmtree()'''
	#(base, ext) = os.path.splitext(dir)
	#if ext != ".database" and ext != ".build":
	#	raise p4a_error("Cannot remove unknown directory: " + dir)
	try:
		for root, dirs, files in os.walk(dir, topdown = False):
			for name in files:
				os.remove(os.path.join(root, name))
			for name in dirs:
				os.rmdir(os.path.join(root, name))
		os.rmdir(dir)
	except Exception as e:
		if can_fail:
			warn("could not remove directory " + dir + ": " + repr(e))
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

# XXX: Not really cross-platform. Should use Git extension for Python instead.
def git_is_dirty(file_or_directory):
	'''Returns True if the Git repository in which the file_or_directory lies has uncommited changes, 
	False otherwise or if the file_or_directory is not versioned in a Git repository.'''
	if not os.path.isdir(file_or_directory):
		(file_or_directory, name) = os.path.split(file_or_directory)
		if not os.path.isdir(file_or_directory):
			return False
	output = os.popen("cd " + file_or_directory + " && git status 2>&1").read()
	if re.search("Changes to be committed:", output) or re.search("Changed but not updated:", output):
		return True
	return False

# XXX: Not really cross-platform. Should use Git extension for Python instead.
def git_get_revision(file_or_directory):
	'''Returns the current revision for the Git repository in which the file_or_directory lies.'''
	if not os.path.isdir(file_or_directory):
		(file_or_directory, name) = os.path.split(file_or_directory)
		if not os.path.isdir(file_or_directory):
			return None
	output = os.popen("cd " + file_or_directory + " && git log --abbrev-commit --pretty=oneline -n 1 2>/dev/null").read().strip()
	if not output:
		return None
	short_rev = output.split(" ")[0]
	if not short_rev:
		return None
	#tag = os.popen("cd " + file_or_directory + " && git describe --tags 2>/dev/null").read().strip()
	if git_is_dirty(file_or_directory):
		short_rev += "~dirty"
	return short_rev

# XXX: Not really cross-platform. Should use Git extension for Python instead.
def git_get_tag(file_or_directory):
	'''Returns the current tag for the Git repository in which the file_or_directory lies.'''
	if not os.path.isdir(file_or_directory):
		(file_or_directory, name) = os.path.split(file_or_directory)
		if not os.path.isdir(file_or_directory):
			return None
	tag = os.popen("cd " + file_or_directory + " && git describe --tags 2>/dev/null").read().strip()
	if tag:
		return tag
	return None

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
	'''Substitute keys with values from map in template designated by template_file.'''
	content = string.Template(slurp(template_file)).substitute(map)
	if not output_file:
		output_file = template_file
	dump(output_file, content)
	if trim_tpl_ext:
		(base, ext) = os.path.splitext(output_file)
		if ext == ".tpl":
			shutil.move(output_file, base)
			output_file = base
	return output_file

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
