from __future__ import with_statement
import re
import os
import fileinput
import pypsbase
import pypsconfig

""" Pyps tool box
it contains various utilities used in pyps.
It also hides the internal of pyps in another module,
to make user interface cleaner
"""

guard_begin = "PIPS include guard begin:"
guard_begin_re = re.compile(r"^// %s (.+) $" % guard_begin)
guard_end = "PIPS include guard end:"
include_re = re.compile(r"^\s*#\s*include\s*(\S+)\s*.*$")

def mkguard(guard, line):
    p = line.rstrip("\r\n").split("//", 1)
    str =  "// %s %s \n" % (guard, p[0])
    if(len(p) == 2):
        str += "//" + p[1] + "\n"
    return str

def guardincludes(fname):
    """ Adds guards around includes."""
    for l in fileinput.FileInput([fname], inplace = True):
        is_include = include_re.match(l)
        if is_include:
            print mkguard(guard_begin, l),
        print l,
        if is_include:
            print mkguard(guard_end, l),

def addBeginnning(fname, text):
    """Adds a line of text at the beginning of fname"""
    fi = fileinput.FileInput([fname], inplace = True)
    for l in fi:
        if fi.isfirstline():
            print text
        print l,
    fi.close()

def unincludes(fname):
    """remove the contents of included files"""
    fi = fileinput.FileInput([fname], inplace = True)
    inside_include = False
    included = None
    end_included = None
    for l in fi:
        match = guard_begin_re.match(l)
        if match:
            included = match.group(1)
            inside_include = True
            end_included = mkguard(guard_end, included)
            print included
            continue
        if l == end_included:
            inside_include = False
            included = None
            end_included = None
            continue
        if inside_include:
            continue
        print l,

def string2file(string, fname):
    f = open(fname, "w")
    f.write(string)
    f.close()
    return fname

def file2string(fname):
    with open(fname, "r") as f:
        s = f.read()
    return s

def nameToTmpDirName(name): return "." + name + ".tmp"

def formatprop(value):
    if type(value) is bool:
        return str(value).upper()
    elif type(value) is str:
        def stringify(s): return '"'+s+'"'
        return stringify(value)
    else:
        return str(value)

def capply(ms,phase):
    """ concurrently apply a phase to all contained modules"""
    if len(ms) > 0:
        ms.workspace.cpypips.capply(phase.upper(),map(lambda m:m.name,ms))

def apply(m, phase, *args, **kwargs):
    """ apply a phase to a module. The method pre_phase and post_phase
    of the originate workspace will be called. """
    m.workspace.pre_phase(phase,m)
    m.workspace.cpypips.apply(phase.upper(),m.name)
    m.workspace.post_phase(phase,m)

def update_props(passe,props):
    """Change a property dictionary by appending the pass name to the property when needed """
    for name,val in props.iteritems():
        if name.upper() not in pypsbase.workspace.Props.all:
            del props[name]
            props[str.upper(passe+"_"+name)]=val
            #print "warning, changing ", name, "into", passe+"_"+name
    return props


def get_property(ws, name):
    name = name.upper()
    """return property value"""
    t = type(ws.props.all[name])

    if t == str:     return ws.cpypips.get_string_property(name)
    elif t == int:   return ws.cpypips.get_int_property(name)
    elif t == bool : return ws.cpypips.get_bool_property(name)
    else :
        raise TypeError( 'Property type ' + str(t) + ' is not supported')

def get_properties(ws, props):
    """return a list of values of props list"""
    res = []
    for prop in props.iteritems():
        res.append(get_property(ws, prop))
    return res

def set_property(ws, prop,value):
    """change property value and return the old one"""
    prop = prop.upper()
    old = get_property(ws,prop)
    if value == None:
        return old
    val=formatprop(value)
    ws.cpypips.set_property(prop.upper(),val)
    return old

def set_properties(ws,props):
    """set properties based on the dictionary props and returns a dictionary containing the old state"""
    old = dict()
    for prop,value in props.iteritems():
        old[prop] = set_property(ws,prop, value)
    return old

def patchIncludes(s):
    if not re.search(r"-I.\s",s) and not re.search(r"-I.$",s):
        s+=" -I."
    return s

def get_runtimefile(fname,subdir=None,isFile=True):
    """Returns runtime file path"""
    searchdirs=[pypsconfig.pypsruntime] # removed "." from the search dir because it leads to complicated situations
    if subdir: searchdirs.insert(1,os.path.join(pypsconfig.pypsruntime,subdir))
    for d in searchdirs:
        f=os.path.join(d,fname)
        if isFile and os.path.isfile(f): return f
        if not isFile and os.path.isdir(f):return f
    raise RuntimeError, "Cannot find runtime file : " + fname + "\nsearch path: "+":".join(searchdirs)

def get_runtimedir(fname,subdir=None):
    return get_runtimefile(fname,subdir=subdir,isFile=False)


def gen_compile_command(rep,makefile,outfile,rule,**opts):
    #Moved here because of code duplication
    commandline = ["make",]
    commandline+=["-C",rep]
    commandline+=["-f",makefile]
    commandline.append("TARGET="+outfile)
    for (k,v) in opts.iteritems():
        commandline.append(k+'='+str(v))
    commandline.append(rule)
    return commandline
