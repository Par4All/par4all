import sys

module = sys.argv[1]
name = sys.argv[2]

text = file('templates/template.py', 'r').read()
text = text.replace('{{template_module}}', module).replace('{{template_name}}', name)
f = open('templates/paws_' + name + '.py', 'w')
f.write(text)
f.close()
