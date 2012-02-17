# -*- coding: utf-8 -*-

"""
Generic tutorial controller

"""
import os, re
from collections  import OrderedDict
from ConfigParser import SafeConfigParser
from subprocess   import Popen, PIPE

from pyramid.view import view_config

from .detector    import detect_language
from .operations  import get_workdir, get_resultdir
from ..utils      import htmlHighlight


_paws_marker      = '<END OF THE STEP>'


def get_tutorialdir(request, tut):
    """Get tutorial directory full path.

    :request: Pyramid request
    :tut:     Tutorial name
    """
    return os.path.join(request.registry.settings['paws.validation'], 'tutorials', os.path.basename(tut))


def get_info(request, tut):
    """Get information about the tutorial from the 'info.ini' file.

    :request:  Pyramid request
    :tutorial: Tutorial name
    """
    cfg = SafeConfigParser()
    cfg.optionxform = str # case-sensitive keys
    cfg.read(os.path.join(get_tutorialdir(request, tut), 'info.ini'))

    info = { 'title' : cfg.get('info', 'title'),
             'descr' : cfg.get('info', 'description'),
             }
    return info


re_display = re.compile(r'\s*display')
re_graph   = re.compile(r'\s*apply PRINT_DOT_DEPENDENCE_GRAPH\s*\[(\w+)\]')

def parse_tpips(tpips):
    """Split tpips file into individual steps
    """
    ops    = OrderedDict()
    graphs = {}
    index  = 1
    for line in file(tpips):
        if index not in ops:
            ops[index] = ''
        ops[index] += line
        m = re_graph.match(line)
        if m:
            graphs[index] = m.group(1)
        if re_display.match(line) or re_graph.match(line):
            index += 1
    return ops, graphs


def create_png(request, function, tutname):
    workdir = request.session['workdir']
    p = Popen([ 'dot', '-Tpng', '-o', workdir + '.png',
                os.path.join(tutname+'.database', function, function+'.dot')],
                stdout=PIPE, stderr=PIPE)
    p.wait()
    _create_result_graphs(workdir + '.png')
    return _create_zoom_image(request, os.path.join(workdir, workdir+'.png'))


@view_config(route_name='tutorial', renderer='pawsapp:templates/tutorial.mako', permission='view')
#@view_config(route_name='tutorial', renderer='string', permission='view')
def tutorial(request):
    """
    """
    form    = request.params
    step    = int(form.get('step', 0))

    tutname = os.path.basename(request.matchdict['tutorial'])  # sanitized
    info    = get_info(request, tutname)
    tutdir  = get_tutorialdir(request, tutname)

    dirname = get_workdir(request, reuse=False if step==0 else True)
    tempdir = request.registry.settings['paws.tempdir']
    resdir  = get_resultdir(request)

    # Step 1 : do some scaffolding to prepare the step-by-step files

    if step == 0:

        source  = os.path.join(tutdir, os.path.basename(info['title']))
        tpips   = os.path.join(tutdir, '%s.tpips' % tutname)

        lang    = detect_language(file(source).read())

        ops, graphs = parse_tpips(tpips)

        # Full tpips script, with end-of-step markers
        full_tpips = os.path.join(tutdir, 'full-markers.tpips')
        file(full_tpips, 'w').write(('echo %s\n' % _paws_marker).join(ops.values()))

        # Step-by-step tpips chunks (in result dir)
        for k,v in ops.items():
            step_tpips = os.path.join(resdir, 'step-%d.tpips' % k)
            file(step_tpips, 'w').write(v)

        # Execute full tpips script
        p = Popen(['tpips', full_tpips, '1'], stdout=PIPE, stderr=PIPE)
        p.wait()

        # Split results along markers and save to file
        results = p.communicate()[0].split(_paws_marker)[:-1]
        for i in range(len(results)):
            step_result = os.path.join(resdir, 'step-%d.txt' % (i+1))
            file(step_result, 'w').write(results[i])

        for index in graphs:
            #results[index] = create_png(request, graphs[index], tutname) 
            pass

        request.session['nb_steps'] = len(results)
        request.session['lang']     = lang

        return dict(tutorial = tutname,
                    step     = step,
                    nb_steps = len(results),
                    lang     = lang,
                    name     = tutname,
                    info     = info,
                    source   = file(source).read(),
                    tpips    = htmlHighlight(re.sub("(echo.*\n)", "", file(tpips).read()), "C"),
                    )

    # Subsequent steps

    else:

        step_result = os.path.join(resdir, 'step-%d.txt' % step)
        step_tpips  = os.path.join(resdir, 'step-%d.tpips' % step)
        lang        = request.session['lang']

        return dict(tutorial = tutname,
                    step     = step,
                    nb_steps = request.session['nb_steps'],
                    lang     = lang,
                    name     = tutname,
                    info     = info,
                    source   = htmlHighlight(file(step_result).read(), lang),
                    tpips    = htmlHighlight(re.sub("(echo.*\n)", "", file(step_tpips).read()), 'C'),
                    )
        
        
