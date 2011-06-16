import sys, traceback

from paste.script.command import Command
from paste.script.filemaker import FileOp
from tempita import paste_script_template_renderer

# 0 - name
# 1 - header

class ToolTemplate(Command):

	summary = "Creation of new functionality of basic tools"
	usage = "--NO USAGE--"
	group_name = "pawsapp"
	parser = Command.standard_parser()

	def command(self):
		print 'header', self.args[1]
		try:
			file_op = FileOp(source_dir=('pylons', 'templates'))
			file_op.template_vars.update(
				{'name': self.args[0],
				 'header': self.args[1]})
		except:
			traceback.print_exc(file=sys.stdout)
			print traceback.format_exc()
		file_op.copy_file(template='site.mako_tmpl',
				dest='templates',
				filename='tools_' + self.args[0] + '.mako',
				template_renderer=paste_script_template_renderer)

