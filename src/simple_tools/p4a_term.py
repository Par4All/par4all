#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Set ANSI Terminal Color and Attributes.
Originally found on http://code.activestate.com/recipes/574451.
'''

import os


# Set to True to disable module (coloring etc.).
disabled = False


esc = '%s[' % chr(27)
reset = '%s0m' % esc
format = '1;%dm'
fgoffset, bgoffset = 30, 40
for k, v in dict(
    attrs = 'none bold faint italic underline blink fast reverse concealed',
    colors = 'grey red green yellow blue magenta cyan white'
).items(): globals()[k] = dict((s, i) for i, s in enumerate(v.split()))


def escape(arg = '', sep = ' ', end = '\n', if_tty_fd = -1):
    '''
    "arg" is a string or None
    If "arg" is None : the terminal is reset to his default values.
    If "arg" is a string it must contain "sep" separated values.
    If args are found in globals "attrs" or "colors", or start with "@" 
    they are interpreted as ANSI commands else they are output as text.
    colors, if any, must be first (foreground first then background)
    you can not specify a background color alone ; 
    if you specify only one color, it will be the foreground one.
        @* commands handle the screen and the cursor :
            @x;y : go to xy
            @	: go to 1;1
            @@   : clear screen and go to 1;1
    Examples:
    escape('red')				: set red as the foreground color
    escape('red blue')			: red on blue
    escape('red blink')			: blinking red
    escape()					: restore terminal default values
    escape('reverse')			: swap default colors
    escape('cyan blue reverse')	: blue on cyan <=> escape('blue cyan')
    escape('red reverse')		: a way to set up the background only
    escape('red reverse blink')	: you can specify any combinaison of 
            attributes in any order with or without colors
    escape('blink Python')		: output a blinking 'Python'
    escape('@@ hello')			: clear the screen and print 'hello' at 1;1
    '''
    if disabled:
        return ''
    if if_tty_fd != -1 and not os.isatty(if_tty_fd):
        return ''
    cmd, txt = [reset], []
    if arg:
        arglist = arg.split(sep)
        for offset in (fgoffset, bgoffset):
            if arglist and arglist[0] in colors:
                cmd.append(format % (colors[arglist.pop(0)] + offset))
        for a in arglist:
            c = format % attrs[a] if a in attrs else None
            if c and c not in cmd:
                cmd.append(c)
            else:
                if a.startswith('@'):
                    a = a[1:]
                    if a == '@':
                        cmd.append('2J')
                        cmd.append('H')
                    else:
                        cmd.append('%sH' % a)
                else:
                    txt.append(a)
    if txt and end:
        txt[-1] += end
    return esc.join(cmd) + sep.join(txt)


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
