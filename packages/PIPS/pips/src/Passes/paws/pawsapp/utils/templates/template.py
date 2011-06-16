
from pyps import workspace, module

def invoke_function(fu, ws):
	return fu._get_code(activate = module.<template_module>)

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

if __name__=="__main__":
	workspace.delete('<template_name>')
	with workspace('<template_name>.c',name='<template_name>',deleteOnClose=True) as ws:
		for fu in ws.fun:
			print invoke_function(fu, ws)

