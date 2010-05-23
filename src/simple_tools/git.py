#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#

'''
Git Repositories Manipulation Class
'''

import sys, os, string
from p4a_util import *

actual_script = os.path.abspath(os.path.realpath(os.path.expanduser(__file__)))
script_dir = os.path.split(actual_script)[0]

class git():
	
	_git_ext = None
	_git_dir = None
	_dir = None
	
	def is_valid_git_dir(self, dir):
		'''Returns True if the directory appears to be a valid Git repository.'''
		result = ((os.path.splitext(dir)[1] == self._git_ext or os.path.split(dir)[1] == self._git_ext) 
			and os.path.exists(os.path.join(dir, "index")))
		#debug("is_valid_git_dir("+ dir +") = " + str(result))
		return result
	
	def __init__(self, any_file_inside_target_repos, git_ext = ".git"):
		'''Construct a class for manipulating a Git repository in which "any_file_inside_target_repos" lies.'''
		self._git_ext = git_ext
		git_dir = os.path.abspath(os.path.realpath(os.path.expanduser(any_file_inside_target_repos)))
		while True:
			if not os.path.isdir(git_dir):
				(git_dir, name) = os.path.split(git_dir)
				if not os.path.isdir(git_dir):
					break
			git_dir = os.path.realpath(git_dir)
			(drive, tail) = os.path.splitdrive(git_dir)
			if not tail or tail == "/":
				break
			if self.is_valid_git_dir(git_dir):
				self._git_dir = git_dir
				break
			maybe_dir = os.path.join(git_dir, self._git_ext)
			if self.is_valid_git_dir(maybe_dir):
				self._git_dir = maybe_dir
				break
			maybe_dir = git_dir + self._git_ext
			if self.is_valid_git_dir(maybe_dir):
				self._git_dir = maybe_dir
				break
			(git_dir, name) = os.path.split(git_dir)
		if self._git_dir:
			self._dir = os.path.normpath(re.sub(re.escape(self._git_ext) + "$", "", self._git_dir))
		else:
			raise p4a_error("git ext " + self._git_ext + " not found for " + any_file_inside_target_repos)
		#debug("git dir for " + any_file_inside_target_repos + ": " + self._git_dir + " (" + self._dir + ")")
	
	def fix_input_file_path(self, path):
		'''Function for fixing input paths: make it relative to the repository root or carp if it is not in there.'''
		if not path:
			return ""
		path = os.path.abspath(os.path.realpath(os.path.expanduser(path)))
		if path[0:len(self._dir)] != self._dir:
			raise p4a_error("file is outside repository in " + self._dir + ": " + path)
		path = path[len(self._dir) + 1:]
		return path
	
	def cmd(self, git_command, can_fail = True):
		'''Runs a git command with correct environment and path settings.'''
		old_git_dir = ""
		if "GIT_DIR" in os.environ:
			old_git_dir = os.environ["GIT_DIR"]
		old_work_tree = ""
		if "GIT_WORK_TREE" in os.environ:
			old_work_tree = os.environ["GIT_WORK_TREE"]
		os.environ["GIT_DIR"] = self._git_dir
		os.environ["GIT_WORK_TREE"] = self._dir
		output = run2([ "git" ] + git_command, can_fail = can_fail, working_dir = self._dir)[0].strip()
		os.environ["GIT_DIR"] = old_git_dir
		os.environ["GIT_WORK_TREE"] = old_work_tree
		return output
	
	def is_dirty(self, file = None):
		'''Returns True if the file in the repository has been altered since last revision.
		If file is None or empty, it will return True if any file in the repository has been modified.'''
		file = self.fix_input_file_path(file)
		args = [ "status" ]
		if file:
			args += [ file ]
		output = self.cmd(args)
		result = False
		if re.search("Changes to be committed:", output) or re.search("Changed but not updated:", output):
			result = True
		#debug("is_dirty("+ file +") = " + str(result))
		return result
	
	def current_revision(self, file = None, test_dirty = True):
		'''Returns the current revision for the currently checked out branch, for the given file.
		file can be None or empty, in which case, the current revision for the whole branch will be returned.'''
		short_rev = None
		file = self.fix_input_file_path(file)
		args = [ "log", "--abbrev-commit", "--pretty=oneline", "-n 1" ]
		if file:
			args += [ file ]
		output = self.cmd(args)
		if output:
			short_rev = output.split(" ")[0]
			if test_dirty and self.is_dirty(file):
				short_rev += "~dirty"
		#debug("current_revision("+ file +") = " + short_rev)
		return short_rev
	
	def git_dir(self):
		'''Returns the absolute path for the .git directory.'''
		return self._git_dir
	
	def dir(self):
		'''Returns the absolute path for the working tree directory.'''
		return self._dir

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
