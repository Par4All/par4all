# -*- coding: utf-8 -*-

"""
Generic tutorial controller

"""
import os, re
from subprocess   import Popen, PIPE

from pyramid.view import view_config

from .operations  import _get_directory, _highlight_code
from .graph       import _create_zoom_image


_dependence_graph = 'apply PRINT_DOT_DEPENDENCE_GRAPH'
_paws_marker      = '<END OF THE STEP>'


def _get_path(request, tutorial):
    """Tutorial files directory
    """
    return os.path.join( request.registry.settings['paws.validation'], 'tutorials', os.path.basename(tutorial))

def _parse_tpips(tpips):
    operations = graphs = {}
    index = 1
    for line in file(tpips):
        if index not in operations:
            operations[index] = []
        operations[index].append(line)
        if line.startswith('display') or line.startswith(_dependence_graph):
            operations[index].append('echo %s\n' % _paws_marker)
            index += 1
            if line.startswith(_dependence_graph):
                graphs[index-1] = line[line.find('[') + 1 : line.find(']')]
    return operations, graphs


def _create_script(request, tutorial, operations):
    path = _get_path(request, tutorial)
    f = file(os.path.join(path, 'temporary.tpips'), 'w')
    for step in operations:
        for line in operations[step]:
            f.write(line + '\n')
    f.close()
    return f.name


def _create_results(request, tutorial, operations, graphs):
    results = {}
    script = _create_script(request, tutorial, operations)
    p = Popen(['tpips', script, '1'], stdout=PIPE, stderr=PIPE)
    p.wait()
    result = p.communicate()[0].split(_paws_marker)
    for index in range(1, len(result) + 1):
        results[index] = result[index-1]
    for index in graphs:
        results[index] = _create_png(request, graphs[index], tutorial) 
    os.remove(script)
    return results

def _create_png(request, function, tutorial):
    workdir = request.session['workdir']
    p = Popen([ 'dot', '-Tpng', '-o', workdir + '.png',
                os.path.join(tutorial+'.database', function, function+'.dot')],
                stdout=PIPE, stderr=PIPE)
    p.wait()
    _create_result_graphs(workdir + '.png')
    return _create_zoom_image(request, os.path.join(workdir, workdir+'.png'))


@view_config(route_name='tutorial', renderer='pawsapp:templates/tutorial.mako', permission='view')
def tutorial(request):

    tutorial = os.path.basename(request.matchdict['tutorial'])  # (sanitized)
    name     = "tutorialName[tutorial]" ##TODO
    path     = _get_path(request, tutorial)

    source   = os.path.join(path, name)
    tpips    = os.path.join(path, tutorial + '.tpips')

    _get_directory(request)
    operations, graphs = _parse_tpips(tpips)
    results = _create_results(request, tutorial, operations, graphs)
    _delete_files(tpips)
    steps   = len(operations) - 1

    return dict(tutorial = tutorial,
                name     = name,
                source   = file(source).read(),
                tpips    = _highlight_code(request, re.sub("(echo.*\n)", "", file(tpips).read()), "C", demo=True),
                )
                
