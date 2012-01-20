import os, re

#path = '/home/local/szymczak/MYPIPS/prod/pips/src/Libs/semantics/'
#path1 = '/home/local/szymczak/MYPIPS/prod/pips/src/Libs/transformer/'

def search(directory):
	files = os.listdir(directory)
	props = set()
	for f in files:
		if f.endswith('.c'):
			text = file(directory + f).read()
			p = re.compile('get_\w+_property\("\w+"\)')
			matches =  p.findall(text)
			for m in matches:
				pi = re.compile('"\w+"')
				props.add(pi.findall(m)[0][1:-1])
	return props

def parse_tex():
	new_set = set()
	text = file('/home/local/szymczak/MYPIPS/prod/pips/src/Documentation/pipsmake/pipsmake-rc.tex').read()
	sections = text.split('\\section')
	for sect in sections:
		if sect.find('{Preconditions}') != -1:
			for line in sect.split('\n'):
				if line.find('begin{PipsProp}') != -1:
					line = line.strip()
					new_set.add(line[line.rindex('{') + 1:-1])
	return new_set

all_props = list(search(path) | search(path1) | parse_tex())
all_props.sort()

fl = open('preconditions_properties_list.lst', 'w')
for p in all_props:
	fl.write(p + '\n')
fl.close()

#parse_tex()
