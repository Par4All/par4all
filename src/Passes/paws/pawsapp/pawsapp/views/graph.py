# -*- coding: utf-8 -*-

import os, sys, shutil
import Image
from subprocess import Popen, PIPE

from pyps       import workspace, module
from pyrops     import pworkspace


class Images(object):

    size = (512, 512)

    def create_thumbnail(self, image):
        outfile = os.path.splitext(image)[0] + '.thumbnail.png'
	try:
            ii = Image.open(image)
            ii.thumbnail(self.size)
            ii.save(outfile, 'PNG')
	except IOError:
            print 'ERROR'
        return outfile[len(paws.public) - 1 : ]
 
    def create_zoom_image(self, image):
        outfile = self.create_thumbnail(paws.results + image)
	print 'o', outfile
	return '<a href="/' + paws.result_dir + image + '" class=' + paws.images_class + ' ><img src="' + outfile + '" /></a>'


def dependence_graph(self):
    """
    """
    source_file = self.create_file('', request.params['code'], request.params['language'])
    ws = pworkspace(str(source_file), name=session['directory'], deleteOnClose=True)
    functions = []
    images = ''
    for fu in ws.fun:
        try:
            functions.append(fu.name)
            fu.print_dot_dependence_graph()
        except:
            ws.close()
            raise
    for fu in functions:
        filename = 'files/' + session['directory'] + '/' + fu + '.png'
        p = Popen(['dot', '-Tpng', '-o', filename, session['directory'] + '.database/' + fu + '/' + fu + '.dot'], stdout = PIPE, stderr = PIPE)
        p.wait()
        self.create_result_graphs(filename)
        images += '<div style="clear: both;width:100%"><b>' + fu + ':</b><br/>' + self.create_zoom_image(session['directory'] + '/' + fu + '.png') + '</div>'
    self.delete_dir(source_file)
    ws.close()
    return images

def dependence_graph_multi(self):
    """
    """
    sources = session['sources']
    print sources[0][0]
    ws = pworkspace(*sources, name=session['directory'], deleteOnClose=True)
    functions = []
    images = ''
    for fu in ws.fun:
        try:
            functions.append(fu.name)
            fu.print_dot_dependence_graph()
        except:
            ws.close()
            raise
    for fu in functions:
        filename = 'files/' + session['directory'] + '/' + fu + '.png'
        p = Popen(['dot', '-Tpng', '-o', filename, session['directory'] + '.database/' + fu + '/' + fu + '.dot'], stdout = PIPE, stderr = PIPE)
        p.wait()
        self.create_result_graphs(filename)
        images += '<div style="clear: both;width:100%"><b>' + fu + ':</b><br/>' + self.create_zoom_image(session['directory'] + '/' + fu + '.png') + '</div>'
    ws.close()
    return images
