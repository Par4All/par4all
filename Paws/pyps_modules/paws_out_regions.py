from __future__ import with_statement # this is to work with python2.5
from pyps import workspace, module

def invoke_function(fu, ws):
        return fu._get_code(activate = module.print_code_out_regions)

if __name__=="__main__":
	workspace.delete('paws_out_regions')
	with workspace('paws_out_regions.c',name='paws_out_regions',deleteOnClose=True) as ws:
        	for fu in ws.fun:
                	print invoke_function(fu, ws)

