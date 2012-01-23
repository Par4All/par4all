"""Helper functions

Consists of functions to typically be used within templates, but also
available to Controllers. This module is available to templates as 'h'.
"""
# Import helpers as desired, or define your own, ie:
#from webhelpers.html.tags import checkbox, password

from webhelpers.pylonslib import Flash as _Flash

from webhelpers.html import literal
from webhelpers.html.tags import *

from webhelpers.paginate import Page

flash = _Flash()
