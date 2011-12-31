import logging
import re, os, shutil
import tempfile

import pawsapp.config.paws as paws

from pygments import highlight
from pygments.lexers import CLexer, FortranLexer
from pygments.formatters import HtmlFormatter

from pylons import session

log = logging.getLogger(__name__)

class FileUtils:

    extensions = { "C" : ".c", "Fortran95" : ".f90", "Fortran77" : ".f", "Fortran" : ".f"}
    temporary_files_path = "files/"

    def create_directory(self):

        mkd = tempfile.mkdtemp(dir=self.temporary_files_path)
        session['directory'] = mkd[mkd.rindex('/') + 1 : ]
        session.save()
        return session['directory']


    def create_file(self, functionality, code, language):

        file_name = self.temporary_files_path + session['directory'] + '/' + functionality + "_code" + self.extensions[language]
	f = open(file_name, 'w')
	f.write(code)
	f.close()
	return file_name


    def get_includes(self, code):

        lines = code.split('\n')
	imports = ''
	for line in lines:
		if re.match("#include", line):
			imports += line + '\n'
	return imports

    def delete_dir(self, file_name):
	file_to_delete = file_name[ : file_name.rindex('/')]
	if os.path.exists(file_to_delete):
		shutil.rmtree(file_to_delete)
	
    def create_result_file(self, code):

	if os.path.exists(paws.results + session['directory']) == False:
		os.mkdir(paws.results + session['directory'])
	f = open(paws.results + session['directory'] + '/' + session['directory'], 'w')
	f.write(code)
	f.close
		
    def create_result_graphs(self, graph):

	path = paws.results + session['directory'] + '/'
	print 'path', path
	if os.path.exists(path) == False:
		os.mkdir(path)
	graph_name = graph[ graph.rfind('/') + 1 : ] if graph.rfind('/') != -1 else graph
	f = open(path + graph_name, 'w')
	f.write(file(graph).read())
	f.close()
	
    def add_lines(self, code):

	code = code.replace('<pre>', '<pre>\n')
	code_with_lines = ''.join(['<li> %s </li>' % x for x in code.split('\n') if x[:4] != '<div'  and x[:5]!='</pre'])
	code_with_lines = '<div class="highlight"><pre><ol>' + code_with_lines
	code_with_lines = code_with_lines + '</ol></pre></div>'
	return code_with_lines

    def highlight_code(self, code, language, demo=False):
	
	code = code.replace('\n\n', '\n')
	if not demo:
		self.create_result_file(code)
	values = {
		"C": self.add_lines(highlight(code, CLexer(), HtmlFormatter())),
		"Fortran77": self.add_lines(highlight(code, FortranLexer(), HtmlFormatter())),
		"Fortran95": self.add_lines(highlight(code, FortranLexer(), HtmlFormatter()))
	}
	return values.get(language, None)
