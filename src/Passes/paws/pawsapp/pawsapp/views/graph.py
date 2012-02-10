# -*- coding: utf-8 -*-

import os, sys, shutil
from subprocess   import Popen, PIPE

import Image
from pyps         import workspace, module
from pyrops       import pworkspace

from pyramid.view import view_config

from .operations  import _create_workdir, _create_file, _get_resdir


# Default image size
_size = (512, 512)


def _create_dot_images(request, functions):
    """
    """
    resdir  = _get_resdir(request)
    workdir = os.path.basename(request.session['workdir']) # sanitized

    images  = []

    for fu in functions:

        # Full-size image
        imgname = os.path.basename('%s.png' % fu) # sanitized
        imgpath = os.path.join(resdir, imgname)
        imgurl  = request.route_url('tool_results_name', tool='tool', name=imgname)

        p = Popen( ['dot', '-Tpng', '-o', imgpath, '%s.database/%s/%s.dot' % (workdir, fu, fu)],
                   stdout=PIPE, stderr=PIPE)
        p.wait()

        # Thumbnail image
        thumbname = '%s.thumbnail.png' % fu
        thumbpath = os.path.join(resdir, thumbname)
        thumburl  = request.route_url('tool_results_name', tool='tool', name=thumbname)
        
        zoom = False
        try:
            im = Image.open(imgpath)
            if im.size[0] > _size[0] or im.size[1] > _size[1]:
                zoom = True
                im.thumbnail(_size, Image.ANTIALIAS)
                im.save(thumbpath, 'PNG')
        except IOError:
            print 'ERROR'

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
    workdir   = request.session['workdir'] if 'workdir' in request.session else _create_workdir(request)        
    ws        = pworkspace(str(source), name=workdir, deleteOnClose=True)
    functions = _get_ws_functions(ws)
    images    = _create_dot_images(request, functions)
    #_delete_dir(source)
    ws.close()
    return dict(images=images)


@view_config(route_name='dependence_graph_multi', renderer='pawsapp:templates/lib/images.mako', permission='view')
def dependence_graph_multi(request):
    """
    """
    workdir   = request.session['workdir'] if 'workdir' in request.session else _create_workdir(request)        
    sources   = request.session['sources']

    ws        = pworkspace(*sources, name=workdir, deleteOnClose=True)
    functions = _get_ws_functions(ws)
    images    = _create_dot_images(request, functions)
    ws.close()
    return dict(images=images)
