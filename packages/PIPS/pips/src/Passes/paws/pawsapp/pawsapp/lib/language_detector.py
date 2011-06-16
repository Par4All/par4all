# -*- coding: utf-8 -*-

import re

__all__ = ['CDecoder', 'FortranDecoder']

class CDecoder:

	tokens = ['{', '}', ';', 'int ', 'float ', 'char ', 'void ', '[', ']', '#include', 'case', 'null']
	name = "C"
	
	def analyze(self, code):
		if code.find(';') != -1:
			return sum([1 for x in self.tokens if code.lower().find(x) != -1]) * 100 / float(len(self.tokens))
		else:
			return 0

class FortranDecoder:

	tokens = ['^c','subroutine ', 'end', 'program ', 'print ', 'integer ', 'call ', 'allocate', 'endif', 'read', 'write', 'real', '(*,*)']
	name = "Fortran"

	def analyze(self, code):
		return sum([1 for x in self.tokens if code.lower().find(x) != -1]) * 100 / float(len(self.tokens))



