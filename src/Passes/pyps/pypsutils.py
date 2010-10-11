import re
import os
import fileinput
import sys
import pyps

""" Pyps tool box
it contains various utilities used in pyps.
It also hides the internal of pyps in another module,
to make user interface cleaner
"""

guard_begin = "PIPS include guard begin:"
guard_begin_re = re.compile(r"^/\* %s (.+) \*/$" % guard_begin)
guard_end = "PIPS include guard end:"
include_re = re.compile(r"^\s*#\s*include\s*(\S+)\s*.*$")

def mkguard(guard, line):
    return "/* %s %s */\n" % (guard, line.rstrip("\r\n"))

def guardincludes(fname):
    """ Adds guards around includes."""
    for l in fileinput.FileInput([fname], inplace = True):
        is_include = include_re.match(l)
        if is_include:
            print mkguard(guard_begin, l),
        print l,
        if is_include:
            print mkguard(guard_end, l),

define_MAX0 = """
/* Header automatically inserted by PYPS for defining MAX, MIN, MOD and others */
#ifndef MAX0
# define MAX0(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MAX
# define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
# define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MOD
# define MOD(a, b) ((a) % (b))
#endif

#ifndef DBLE
# define DBLE(a) ((double)(a))
#endif

#ifndef INT
# define INT(a) ((int)(a))
#endif

#ifdef WITH_TRIGO
#  include <math.h>
#  ifndef COS
#    define COS(a) (cos(a))
#  endif

#  ifndef SIN
#    define SIN(a) (sin(a))
#  endif
#endif
/* End header automatically inserted by PYPS for defining MAX, MIN, MOD and others */
"""

def addMAX0(fname):
    """ Adds #define's for MAX0 and MOD."""
    addBeginnning(fname, define_MAX0)

def addBeginnning(fname, text):
    """Adds a line of text at the beginning of fname"""
    fi = fileinput.FileInput([fname], inplace = True)
    for l in fi:
        if fi.isfirstline():
            print text
        print l,
    
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
            print l,
            print included
            continue
        if l == end_included:
            inside_include = False
            included = None
            end_included = None
            print l,
            continue
        if inside_include:
            continue
        print l,

def string2file(string, fname):
    f = open(fname, "w")
    f.write(string)
    f.close()
    
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
	if ms._modules:
		ms._ws.cpypips.capply(phase.upper(),map(lambda m:m.name,ms._modules))

def apply(m, phase, *args, **kwargs):
	""" apply a phase to a module"""
	m._ws.cpypips.apply(phase.upper(),m._name)

def update_props(passe,props):
	"""Change a property dictionnary by appending the pass name to the property when needed """
	for name,val in props.iteritems():
		if name.upper() not in pyps.workspace.props.all:
			del props[name]
			props[str.upper(passe+"_"+name)]=val
			#print "warning, changing ", name, "into", passe+"_"+name
	return props

def build_module_list(ws):
	""" update workspace module list """
	for m in ws.info("modules"):
		ws._modules[m]=pyps.module(ws,m,ws._sources[0])

# A regex matching compilation unit names ending with a "!":
re_compilation_units = re.compile("^.*!$")
def filter_compilation_units(ws):
	""" retreive compilation unit """
	return ws.filter(lambda m: re_compilation_units.match(m.name))

def filter_all_functions(ws):
	""" retreive function, all non compilation unit """
	return ws.filter(lambda m: not re_compilation_units.match(m.name))

def get_property(ws, name):
	name = name.upper()
	"""return property value"""
	t = type(ws.props.all[name])

	if t == str:     return ws.cpypips.get_string_property(name)
	elif t == int:   return ws.cpypips.get_int_property(name)
	elif t == bool : return ws.cpypips.get_bool_property(name)
	else : 
		raise TypeError( 'Property type ' + str(t) + ' isn\'t supported')

def get_properties(ws, props):
	"""return a list of values of props list"""
	res = []
	for prop in props.iteritems():
		res.append(get_property(ws, prop))
	return res

def _set_property(ws, prop,value):
	"""change property value and return the old one"""
	prop = prop.upper()
	old = get_property(ws,prop)
	if value == None:
		return old
	val=formatprop(value)
	ws.cpypips.set_property(prop.upper(),val)
	return old

def set_properties(ws,props):
	"""set properties based on the dictionnary props and returns a dictionnary containing the old state"""
	old = dict()
	for prop,value in props.iteritems():
		old[prop] = _set_property(ws,prop, value)
	return old

def set_property(ws, **props):
	"""set properties and return a dictionnary containing the old state"""
	return ws.set_properties(props)
