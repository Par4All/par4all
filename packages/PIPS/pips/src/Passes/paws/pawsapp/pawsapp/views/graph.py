# -*- coding: utf-8 -*-

import os, sys, shutil
import Image
from subprocess   import Popen, PIPE
from pyps         import workspace, module
from pyrops       import pworkspace
from pyramid.view import view_config
from .operations  import _create_file, _create_result_graphs, _delete_dir


# Default image size
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

    return dict(link=link, image=outfile)


def _create_dot_images(request, functions):
    workdir = request.session['workdir']
    images  = []
    for fu in functions:
        filename = 'files/%s/%s.png' % (workdir, fu)
        p = Popen( ['dot', '-Tpng', '-o', filename, '%s.database/%s/%s.dot' % (workdir, fu, fu)], stdout=PIPE, stderr=PIPE)
        p.wait()
        _create_result_graphs(request, filename)
        image = _create_zoom_image(request, '%s/%s.png' % (workdir, fu))
        image['fu'] = fu
        images.append(image)
    return images


def _get_ws_functions(ws):
    functions = []
    for fu in ws.fun:
        try:
            functions.append(fu.name)
            fu.print_dot_dependence_graph()
        except:
            ws.close()
            raise
    return functions


@view_config(route_name='dependence_graph', renderer='pawsapp:templates/lib/images.mako', permission='view')
def dependence_graph(request):
    """
    """
    source    = _create_file(request, '', request.params['code'], request.params['language'])
    workdir   = request.session['workdir']
    ws        = pworkspace(str(source), name=workdir, deleteOnClose=True)
    functions = _get_ws_functions(ws)
    images    = _create_dot_images(request, functions)
    _delete_dir(source)
    ws.close()
    return dict(images=images)


@view_config(route_name='dependence_graph_multi', renderer='pawsapp:templates/lib/images.mako', permission='view')
def dependence_graph_multi(request):
    """
    """
    sources   = request.session['sources']
    workdir   = request.session['workdir']
    ws        = pworkspace(*sources, name=workdir, deleteOnClose=True)
    functions = _get_ws_functions(ws)
    images    = _create_dot_images(request, functions)
    ws.close()
    return dict(images=images)
