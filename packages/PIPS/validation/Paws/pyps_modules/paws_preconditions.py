# -*- coding: utf-8 -*-

from __future__ import with_statement # this is to work with python2.5
from pyps       import workspace, module


def invoke_function(fu, ws):
    return fu._get_code(activate = module.print_code_preconditions)


if __name__=="__main__":
    workspace.delete('paws_preconditions')
    with workspace('paws_preconditions.c', name='paws_preconditions', deleteOnClose=True) as ws:
        for fu in ws.fun:
            print invoke_function(fu, ws)

