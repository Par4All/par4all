# -*- coding: utf-8 -*-

import os, sys, shutil
from subprocess   import Popen, PIPE

import Image
from pyps         import workspace, module
from pyrops       import pworkspace

from pyramid.view import view_config

from .operations  import create_workdir, get_resultdir, _create_file


# Default image size
_size = (512, 512)


def create_graph_images(request, functions, db=None):
    """Create a pair of images (full-size, thumbnail) from a dot graph.

    :request  : Pyramid request
    :functions: List of functions to graph
    :db:      : PIPS database directory
    """
    resdir  = get_resultdir(request)
    workdir = os.path.basename(request.session['workdir']) # sanitized
    
    if db is None:
        db = workdir + '.database'

    images  = []

    for fu in functions:

        # Full-size image
        imgname = os.path.basename('%s-%s.png' % (workdir, fu)) # sanitized
        imgpath = os.path.join(resdir, imgname)
        imgurl  = request.route_url('results_name', tool='tool', name=imgname)        
        p = Popen( ['dot', '-Tpng', '-o', imgpath, os.path.join(db, fu, '%s.dot' % fu)],
                   stdout=PIPE, stderr=PIPE)
        p.wait()

        # Thumbnail image
        thumbname = os.path.basename('%s-%s.thumbnail.png' % (workdir, fu)) # sanitized
        thumbpath = os.path.join(resdir, thumbname)
        thumburl  = request.route_url('results_name', tool='tool', name=thumbname)
        
        zoom = False
        try:
            im = Image.open(imgpath)
            if im.size[0] > _size[0] or im.size[1] > _size[1]:
                zoom = True
                im.thumbnail(_size, Image.ANTIALIAS)
                im.save(thumbpath, 'PNG')
        except IOError, msg:
            print 'ERROR:', msg

        images.append(dict(full=imgurl, thumb=thumburl, fu=fu, zoom=zoom))

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
    form = request.params
    code = form.get('code')
    lang = form.get('lang')

    source    = _create_file(request, '', code, lang)
    dirname   = create_workdir(request)        
    ws        = pworkspace(str(source), name=dirname, deleteOnClose=True)
    functions = _get_ws_functions(ws)
    images    = create_graph_images(request, functions)
    #_delete_dir(source)
    ws.close()
    return dict(images=images)


@view_config(route_name='dependence_graph_multi', renderer='pawsapp:templates/lib/images.mako', permission='view')
def dependence_graph_multi(request):
    """
    """
    dirname   = create_workdir(request)        
    sources   = request.session['sources']

    ws        = pworkspace(*sources, name=dirname, deleteOnClose=True)
    functions = _get_ws_functions(ws)
    images    = create_graph_images(request, functions)
    ws.close()
    return dict(images=images)
