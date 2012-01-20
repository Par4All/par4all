# -*- coding: utf-8 -*-

from pylons import request, response, session, tmpl_context as c, url
from pylons.controllers.util import abort, redirect
from pawsapp.lib.base import BaseController, render


class ToolsPreconditionsController(BaseController):

    def index(self):

        c.id     = 'preconditions'
        c.title  = 'Preconditions over scalar integer variables'
        c.link   = '/tools_preconditions/advanced'

        return render('/tool.mako')


    def advanced(self):

        return render('/tools_preconditions_advanced.mako')
