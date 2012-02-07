# -*- coding: utf-8 -*-

from __future__ import with_statement # this is to work with python2.5
from pyps       import workspace, module
from pyrops     import pworkspace


def import_module(operation):
    return __import__('paws_' + operation, None, None, ['__all__'])

def perform(source_path, operation, advanced=False, properties=None, analysis=None, phases=None):
    mod = import_module(operation)
    try:
        ws = pworkspace(str(source_path), deleteOnClose=True)
        if advanced:
            if (properties != ""): set_properties(ws, str(properties).split(';')[: -1])
            if (analysis != ""): activate(ws, analysis)
            if (phases != ""): apply_phases(ws, phases)
        functions = ''
        for fu in ws.fun:
            functions += mod.invoke_function(fu, ws)
        ws.close()
        return functions
    except:
        ws.close()
        raise

def perform_multiple(sources, operation, function_name, advanced=False, properties=None, analysis=None, phases=None):
    mod = import_module(operation)
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

def activate(ws, analyses):
    for analysis in analyses[ : -1].split(';'):
        ws.activate(str(analysis))

def apply_phases(ws, phases):
    for phase in phases[ : -1].split(';'):
        if phase.split()[1] == 'true':
            getattr(ws.all, phase.split()[0])()

def set_properties(ws, properties):
    for prop in properties:
        pair = prop.split()
        default = getattr(ws.props, pair[0])
        if type(default) == int:
            value = int(pair[1])
        elif type(default) == bool:
            value = False if pair[1] == 'false' else True
        else:
            value = pair[1]
        if default != value:
            setattr(ws.props, pair[0], value)

def get_functions(sources):
    ws = pworkspace(*sources, deleteOnClose=True) ##TODO: cast source file names to str
    functions = []
    for fu in ws.fun:
        functions.append(fu.name)
    ws.close()
    return functions

