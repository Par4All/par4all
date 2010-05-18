#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Par4All Configuration Management
'''

import sys, os, string

export_keys = ["P4A_ROOT", "P4A_DIST", "P4A_SRC", "P4A_ACCEL_DIR"]
valid_keys = export_keys + []

class p4a_config(dict):
	def __init__(self, kv_dict = {}, from_env = True):
		dict.__init__(self)
		if from_env:
			for k in os.environ:
				if k in valid_keys:
					kv_dict[k] = os.environ[k]
		for k in kv_dict:
			if k not in valid_keys:
				print("invalid key: " + k)
		self.update(kv_dict)

	def __getitem__(self, k):
		return self.get(k, None)

	def __delitem__(self, k):
		if self.has_key(k):
			dict.__delitem__(self, k)

	def print_var(self, k, v, csh = False):
		v = v.replace("'", "\\'")
		s = ""
		if csh:
			if k in export_keys:
				s += "setenv " + k + " '" + v + "'"
			else:
				s += "" + k + "='" + v + "'"
		else:
			if k in export_keys:
				s += "export " + k + "='" + v + "'" 
			else:
				s += "" + k + "='" + v + "'"
		return s

	def __str__(self, csh = False):
		s = ""
		for k in self: 
			s += self.print_var(k, self[k], csh)
		return s

if __name__ == "__main__":
	print("# Current P4A configuration")
	print(p4a_config())

# What? People still use emacs? :-)
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
