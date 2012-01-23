# -*- coding: utf-8 -*-

"""

Generic tool controller

"""

import os

from pylons import request, response, session, tmpl_context as c, url, config
from pylons.controllers.util import abort, redirect
from pawsapp.lib.base import BaseController, render


toolName = dict(
    preconditions = u'Preconditions over scalar integer variables',
    openmp        = u'Openmp demo page',
    in_regions    = u'IN regions',
    out_regions   = u'OUT regions',
    regions       = u'Array regions',
    )

def _get_examples(tool):
    path = os.path.join(config['paws.validation'], 'tools', tool)
    return [ f for f in sorted(os.listdir(path))
             if os.path.isdir(f) == False
             and (f.endswith('.c') or f.endswith('.f'))]


class ToolsController(BaseController):

    def index(self, tool, level=None):

        c.id       = tool
        c.title    = toolName[tool]
        c.adv      = bool(level == 'advanced')
        c.examples = _get_examples(tool)

        return render('/tool.mako')

