# -*- coding: utf-8 -*-

import os, sys, shutil
import Image
from subprocess   import Popen, PIPE
from pyps         import workspace, module
from pyrops       import pworkspace
from pyramid.view import view_config
from .operations  import _create_file, _create_result_graphs, _delete_dir


_size = (512, 512)


def _create_thumbnail(request, image):
    """
    """
    publicdir = request.registry.settings['paws.publicdir']
    outfile   = '%s.thumbnail.png' % os.path.splitext(image)[0]
    try:
        ii = Image.open(image)
        ii.thumbnail(_size)
        ii.save(outfile, 'PNG')
    except IOError:
        print 'ERROR'
    return '/' + os.path.relpath(outfile, os.path.join(publicdir, '..'))


def _create_zoom_image(request, image):
    """
    """
    resultdir = request.registry.settings['paws.resultdir']
    publicdir = request.registry.settings['paws.publicdir']

    outfile   = _create_thumbnail(request, os.path.join(resultdir, image))
    link      = '/' + os.path.relpath(os.path.join(resultdir, image), os.path.join(publicdir, '..'))

    print 'o', link, outfile
    return '<a href="%s" class="ZOOM_IMAGE"><img src="%s"/></a>' % (link, outfile)


@view_config(route_name='dependence_graph', renderer='string')
def dependence_graph(request):
    """
    """
    source_file = _create_file(request, '', request.params['code'], request.params['language'])
    workdir     = request.session['workdir']
    ws = pworkspace(str(source_file), name=workdir, deleteOnClose=True)
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
        filename = 'files/%s/%s.png' % (workdir, fu)
        p = Popen( ['dot', '-Tpng', '-o', filename, '%s.database/%s/%s.dot' % (workdir, fu, fu)],
                   stdout=PIPE, stderr=PIPE)
        p.wait()
        _create_result_graphs(request, filename)
        images += '<div style="clear: both;width:100%%"><b>%s:</b><br/>%s</div>' % (fu, _create_zoom_image(request, '%s/%s.png' % (workdir, fu)))
    _delete_dir(source_file)
    ws.close()
    return images

@view_config(route_name='dependence_graph_multi', renderer='string')
def dependence_graph_multi(request):
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
        _create_result_graphs(request, filename)
        images += '<div style="clear:both; width:100%%"><b>%s:</b><br/>%s</div>' % (fu, _create_zoom_image(request, '%s/%s.png' % (workdir, fu)))
    ws.close()
    return images
