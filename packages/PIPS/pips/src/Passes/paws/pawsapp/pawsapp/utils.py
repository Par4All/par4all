# -*- coding: utf-8 -*-

"""
PAWS utility functions

"""

# Supported languages
languages = {
    "C"         : dict(cmd='gcc',      ext='.c'),
    "Fortran77" : dict(cmd='f77',      ext='.f'),
    "Fortran95" : dict(cmd='gfortran', ext='.f90'),
    "Fortran"   : dict(cmd='f77',      ext='.f'),
    }


# Language detection

class CDecoder(object):
    """
    """
    tokens = [ '{', '}', ';', 'int ', 'float ', 'char ', 'void ', '[', ']', '#include', 'case', 'null']
    name   = "C"

    @classmethod
    def analyze(cls, code):
        return len(filter(lambda t: t in code.lower(), cls.tokens)) * 100.0 / len(cls.tokens) \
            if code.find(';') != -1 else 0


class FortranDecoder(object):
    """
    """
    tokens = [ '^c', 'subroutine ', 'end', 'program ', 'print ', 'integer ', 'call ', 'allocate',
               'endif', 'read', 'write', 'real', '(*,*)']
    name   = "Fortran"

    @classmethod
    def analyze(cls, code):
        return len(filter(lambda t: t in code.lower(), cls.tokens)) * 100.0 / len(cls.tokens)
