#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Par4All Common Utility Functions
'''

import string, sys, random, logging, os, gc
import term

# Global variables.
verbosity = 0
logger = None

def set_verbosity(level):
	global verbosity
	verbosity = level

def get_verbosity():
	global verbosity
	return verbosity

# Printing/logging helpers.
def debug(msg):
	if verbosity >= 3:
		sys.stderr.write(sys.argv[0] + ": " + msg.rstrip("\n") + "\n");
	if logger:
		logger.debug(msg)

def info(msg):
	if verbosity >= 2:
		sys.stderr.write(sys.argv[0] + ": " + term.escape("white") + msg.rstrip("\n") + term.escape() + "\n");
	if logger:
		logger.info(msg)

def warn(msg):
	if verbosity >= 1:
		sys.stderr.write(sys.argv[0] + ": " + term.escape("yellow") + msg.rstrip("\n") + term.escape() + "\n");
	if logger:
		logger.warn(msg)

def error(msg):
	sys.stderr.write(sys.argv[0] + ": " + term.escape("red") + msg.rstrip("\n") + term.escape() + "\n");
	if logger:
		logger.error(msg)

def die(msg, exit_code = 255):
	error(msg)
	error("aborting")
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
	if verbosity >= 2:
		sys.stderr.write(sys.argv[0] + ": " + term.escape("magenta") + cmd + term.escape() + "\n");
	ret = os.system(cmd)
	if ret != 0 and not can_fail:
		raise p4a_error("command failed with exit code " + str(ret))
	return ret

def gen_name(length = 4, prefix = "P4A", chars = string.letters + string.digits):
	'''Generates a random name / password'''
	return prefix + "".join(random.choice(chars) for x in range(length))

def rmtree(dir, can_fail = 0):
	'''Removes a directory recursively, because sometimes shutil.rmtree() does not want to'''
	(base, ext) = os.path.splitext(dir)
	if ext != ".database" and ext != ".build":
		raise p4a_error("Cannot remove unknown directory: " + dir)
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

if __name__ == "__main__":
	print(__doc__)
	print("This module is not directly executable")

# What? People still use emacs? :-)
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
