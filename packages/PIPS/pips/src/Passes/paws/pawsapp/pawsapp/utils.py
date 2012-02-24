# -*- coding: utf-8 -*-

"""
PAWS utility functions for language handling etc.

"""
import os, re
from ConfigParser        import SafeConfigParser

from pygments            import highlight
from pygments.lexers     import CLexer, FortranLexer
from pygments.formatters import HtmlFormatter


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


# Extract first comment
def extractFirstComment(code, lang, html=True):
    """Extract and return first comment block from code. Supported
    languages are Fortran and C.

    :code: source cide
    :lang: language of source code
    :html: output <br/>'s instead of line breaks ?
    """

    # Extraction of first Fortran comment
    if lang in languages and lang.startswith('Fortran'):

        start = False
        end   = False
        out   = []

        for l in code.split('\n'):
            if re.match('\s*$', l): # blank line: skip
                continue
            match = re.match(r'c     (.*)', l, re.I) # Fortran comment
            if match:
                start = True
                if not end:
                    out.append(match.group(1))
            elif start:
                end = True

    # Extraction of first C comment
    elif lang == 'C':

        inside  = False
        outside = False
        out     = []

        for l in code.split('\n'):

            if re.match('\s*$', l): # blank line: skip
                continue

            match = re.match(r'/\*\s*(.*?)\s*\*/', l)
            if match: # single-line comment /* ... */
                return match.group(1)

            match = re.match(r'//\s*(.*)', l)
            if match: # single-line comment // ...
                return match.group(1).strip()

            if not inside: 
                match = re.match(r'/\*\s*(.*)', l)
                if match: # start of multi-line comment
                    if match.group(1).strip():
                        out.append(match.group(1).strip())
                    inside=True
                    continue

            if inside:

                match = re.match(r'(.*)\s*\*/\s*$', l)
                if match: # end of multi-line comment
                    if match.group(1).strip():
                        out.append(match.group(1).strip())
                    break

                match = re.match(r'\*\s*(.*)', l)
                if match: # continuation of multi-line comment
                    out.append(match.group(1).strip())

    return ('<br/>' if html else '\n').join(out)


# Source code highlighting
def htmlHighlight(code, lang):
    """Apply HTML formatting to highlight code fragment.

    :code:  Source code
    :lang:  Language of source code
    """
    lexer = None
    if lang == 'C':
        lexer = CLexer()
    elif lang in ('Fortran77', 'Fortran95'):
        lexer = FortranLexer()

    if lexer:
        code  = highlight(code, lexer, HtmlFormatter()).replace('<pre>', '<pre>\n')
        lines = [ '<li>%s</li>' % l for l in code.split('\n') if l[:4] != "<div" and l[:5]!="</pre" ]
        return '<pre class="prettyprint linenums"><ol class="highlight linenums">%s</ol></pre>' % ''.join(lines) # absolutely NO blank spaces!
    else:
        return code


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
