# -*- coding: utf-8 -*-

"""
PAWS utility functions

"""
import os
from ConfigParser import SafeConfigParser


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

#
#
#

def getSiteSections(request):
    """Compute site sections for the menu bar and home page.
    """
    cfg = SafeConfigParser()
    val_path = request.registry.settings['paws.validation']

    cfg.read(os.path.join(val_path, 'main', 'site_sections.ini'))    
    sections = [ dict(title=s, path=cfg.get(s, 'path'), advmode=cfg.getboolean(s, 'advmode')) for s in cfg.sections() ]

    for s in sections:
        path = os.path.join(val_path, os.path.basename(s['path']))
        s['entries'] = []
        for t in os.listdir(path):
            if not t.startswith('.'):
                cfg.read(os.path.join(path, t, 'info.ini'))
                s['entries'].append({'name':t, 'title':cfg.get('info', 'title'), 'descr':cfg.get('info', 'description')})

    return sections
