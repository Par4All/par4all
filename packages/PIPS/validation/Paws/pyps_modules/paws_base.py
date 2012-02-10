# -*- coding: utf-8 -*-

from __future__ import with_statement # this is to work with python2.5
from pyps       import workspace, module
from pyrops     import pworkspace


def import_module(op):
    return __import__('paws_' + op, None, None, ['__all__'])

def perform(source_path, op, advanced=False, **params):
    mod = import_module(op)
    try:
        ws = pworkspace(str(source_path), deleteOnClose=True)

        if advanced:

            # Set properties
            for props in params.get('properties', {}).values():
                for p in props:
                    if p['checked']:
                        val = str(p['val']) if isinstance(p['val'], unicode) else p['val'] ##TODO
                        setattr(ws.props, p['id'], val)

            # Activate analyses
            for a in params.get('analyses', []):
                if a['checked']:
                    ws.activate(str(a['val']))

            # Apply phases
            for p in params.get('phases', []):
                if p['checked']:
                    getattr(ws.all, p['id'])()

        functions = ''
        for fu in ws.fun:
            functions += mod.invoke_function(fu, ws)

        ws.close()
        return functions

    except:
        ws.close()
        raise

def perform_multiple(sources, op, function_name, advanced=False, **params):
    mod = import_module(op)
    try:
        ws = pworkspace(*sources, deleteOnClose=True) ##TODO: cast source file names to str
        if advanced:
            if (properties != ""): set_properties(ws, str(properties).split(';')[: -1])
            if (analysis != ""): activate(ws, analysis)
            if (phases != ""): apply_phases(ws, phases)		
        result = mod.invoke_function(ws.fun.__getattr__(function_name), ws)
        ws.close()
        return result
    except:
        ws.close()
        raise

def get_functions(sources):
    ws = pworkspace(*sources, deleteOnClose=True) ##TODO: cast source file names to str
    functions = []
    for fu in ws.fun:
        functions.append(fu.name)
    ws.close()
    return functions

