from __future__ import with_statement # this is to work with python2.5
from pyps import workspace, module
from pyrops import pworkspace

def import_module(operation):
	return __import__('paws_' + operation, None, None, ['__all__'])

def perform(source_path, operation, advanced=False, properties=None, analysis=None):
	mod = import_module(operation)
        try:
                ws = pworkspace(source_path, deleteOnClose=True)
                if advanced:
                        set_properties(ws, str(properties).split(';')[: -1])
                        activate(ws, analysis)
                functions = ''
                for fu in ws.fun:
                        functions += mod.invoke_function(fu, ws)
                ws.close()
                return functions
        except:
                ws.close()
                raise

def perform_multiple(sources, operation, function_name):
	mod = import_module(operation)
        try:
                ws = pworkspace(*sources, deleteOnClose=True)
                result = mod.invoke_function(ws.fun.__getattr__(function_name), ws)
                ws.close()
                return result
        except:
                ws.close()
                raise

def activate(ws, operation_type):
        ws.activate(str(operation_type))

def set_properties(ws, properties):

        for prop in properties:
                pair = prop.split()
                default = getattr(ws.props, pair[0])
                if type(default) == int:
                        value = int(pair[1])
                elif type(default) == bool:
                        if pair[1] == 'false':
                                value = False
                        else:
                                value = True
                else:
                        value = pair[1]
                if default != value:
                        setattr(ws.props, pair[0], value)

def get_functions(sources):
        ws = pworkspace(*sources, deleteOnClose=True)
        functions = []
        for fu in ws.fun:
                functions.append(fu.name)
        ws.close()
        return functions

