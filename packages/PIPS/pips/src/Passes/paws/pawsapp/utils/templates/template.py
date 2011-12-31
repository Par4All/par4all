from __future__ import with_statement # this is to work with python2.5
from pyps import workspace, module

def invoke_function(fu, ws):
	return fu._get_code(activate = module.{{template_module}})

if __name__=="__main__":
	workspace.delete('{{template_name}}')
	with workspace('{{template_name}}.c',name='{{template_name}}',deleteOnClose=True) as ws:
		for fu in ws.fun:
			print invoke_function(fu, ws)

