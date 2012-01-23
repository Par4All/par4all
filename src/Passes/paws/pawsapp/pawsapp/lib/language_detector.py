# -*- coding: utf-8 -*-


__all__ = ['CDecoder', 'FortranDecoder']


class CDecoder(object):

    tokens = [ '{', '}', ';', 'int ', 'float ', 'char ', 'void ', '[', ']', '#include', 'case', 'null']
    name   = "C"
	
    def analyze(self, code):
        return len(filter(lambda t: t in code.lower(), self.tokens)) * 100.0 / len(self.tokens) \
            if code.find(';') != -1 else 0


class FortranDecoder(object):

    tokens = [ '^c', 'subroutine ', 'end', 'program ', 'print ', 'integer ', 'call ', 'allocate',
               'endif', 'read', 'write', 'real', '(*,*)']
    name   = "Fortran"

    def analyze(self, code):
        return len(filter(lambda t: t in code.lower(), self.tokens)) * 100.0 / len(self.tokens)



