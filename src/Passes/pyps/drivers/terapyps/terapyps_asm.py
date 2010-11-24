import re as RE
import operator
from logging import info, getLogger

getLogger().setLevel(1)

def conv(input,output):
	info("reading input file "+input)
	fd=open(input)
	line=reduce(operator.add,fd.readlines())
	fd.close()
	
	info("pruning trailing comments, declarations and indentation")
	# this is for the comments
	line=RE.sub(r'\n\s*/\*(.*?)\*/\s*\n',r'\n*\1\n',line)
	line=RE.sub(r'^\s*/\*(.*?)\*/\s*\n',r'\n*\1\n',line)
	line=RE.sub(r'\s*//.*\n',r'\n',line)
	line=RE.sub(r'(/\*.*?\*/)|(;)',r'',line)
	line=RE.sub(r'[ \t]*\n+[ \t]*',r'\n',line)
	# this is for the declarations
	line=RE.sub(r'(short|int)\s*\*?\s*(re|im|ma)[0-9]+(\s*,\s*\*?\s*(re|im|ma)[0-9]+)*',r'',line)
	
	info("managing substitution")
	def print_fields(fields):
		padding=24
		res=fields[0].center(padding)
		for field in fields[1:]:
			res+='||'+field.center(padding)
		return res+'\n'
	
	def print_im(field): 		return print_fields([field,' ',' ',' ',' '])
	def print_ma(field): 		return print_fields([' ',field,' ',' ',' '])
	def print_re(field): 		return print_fields([' ',' ',field,' ',' '])
	def print_do(field): 		return print_fields([' ',' ',' ',field,' '])
	def print_return(field): 	return print_fields([' ',' ',' ',' ',field])
	
	def re(m,i):return "re("+m.group(i)+")"
	def lre(m,i):return 'P,'+re(m,i)
	def im(m,i):return m.group(i)
	def lim(m,i):return 'P,'+im(m,i)
	
	
	#reX+=reY
	line=RE.sub(r'addi\(re([0-9]+), re([0-9]+)\)',
			lambda m:print_re(lre(m,1)+'='+re(m,1)+'+'+re(m,2)),
			line)
	#reX*=reY
	line=RE.sub(r'muli\(re([0-9]+), re([0-9]+)\)',
			lambda m:print_re(lre(m,1)+'='+re(m,1)+'*'+re(m,2)),
			line)
	
	#reX+=im
	line=RE.sub(r'addi\(re([0-9]+), (im[0-9]+)\)',
			lambda m:print_re(lre(m,1)+'='+re(m,1)+ +"+"+im(m,2)),
			line)
	
	#reX+=\d+
	line=RE.sub(r'addi\(re([0-9]+), ([0-9]+)\)',
			lambda m:print_re(lre(m,1)+'='+re(m,1)+'+'+im(m,2)),
			line)
	
	#reX-=\d+
	line=RE.sub(r'subi\(re([0-9]+), ([0-9]+)\)',
			lambda m:print_re(lre(m,1)+'='+re(m,1)+'-'+im(m,2)),
			line)
	#reX/=\d+
	line=RE.sub(r'divi\(re([0-9]+), ([0-9]+)\)',
			lambda m:print_re(lre(m,1)+'='+re(m,1)+'/'+im(m,2)),
			line)
	
	#reX+=imY[reZ]
	line=RE.sub(r'addi\(re([0-9]+), (im[0-9]+)\[re([0-9]+)\]\)',
			lambda m: print_re(lim(m,2)+"+"+re(m,3))+print_re(lre(m,1)+'='+re(m,1)+'+im'),
			line)
	#reX+=imY[\d+]
	line=RE.sub(r'addi\(re([0-9]+), (im[0-9]+)\[([0-9]+)\]\)',
			lambda m: print_re(lim(m,2)+"+"+m.group(3))+print_re(lre(m,1)+'+im'),
			line)
	#reX=imY
	line=RE.sub(r'seti\(re([0-9]+), (im[0-9]+)\)',
			lambda m:print_re(lre(m,1)+"="+im(m,2)),
			line)
	#imX=imY
	line=RE.sub(r'seti\((im[0-9]+), (im[0-9]+)\)',
			lambda m:print_im(im(m,1)+"="+im(m,2)),
			line)
	#*imX=*imY
	line=RE.sub(r'psetpi\(\*(im[0-9]+), \*(im[0-9]+)\)',
			lambda m:print_re(lim(m,1)+"="+im(m,2)),
			line)
	#*imX+=*imY
	line=RE.sub(r'addi\(\*(im[0-9]+), \*(im[0-9]+)\)',
			lambda m:print_re(lim(m,1)+"="+im(m,1))+print_re(lim(m,1)+"=P+"+im(m,2)),
			line)
	#inX+=1
	line=RE.sub(r'paddi\((im[0-9]+), 1\)',
			lambda m:print_im(lim(m,1)+"=E"),
			line)
	#reX=re(Z)
	line=RE.sub(r'seti\(re([0-9]+), re([0-9]+)\)',
			lambda m:print_re(lre(m,1)+"="+re(m,2)),
			line)
	#reX=\d+
	line=RE.sub(r'seti\(re([0-9]+), ([0-9]+)\)',
			lambda m:print_re(lre(m,1)+'='+m.group(2)),
			line)
	
	#imX=FIFOY
	line=RE.sub(r'seti\((im[0-9]+), (FIFO[0-9]+)\)',
			lambda m:print_im(lim(m,1)+"="+m.group(2)),
			line)
	
	#imX[reY]=reZ !! to check !!
	line=RE.sub(r'seti\((im[0-9]+)\[re([0-9]+)\], (re([0-9]+))\)',
			lambda m:print_im(lim(m,1)+"+"+re(m,2))+print_im("im="+re(m,3)),
			line)
	
	# void microcode(short *FIFO0, short *FIFO1, short *FIFO2, short iter1, short iter2)
	line=RE.sub(r'void ([a-zA-Z]\w*)\(.*?\)\s*{',
			r'sub \1\n',
			line)
	line=RE.sub(r'}\s*$',r'endsub\n',line)
	
	# for(re0 = 0 re0 <= N0 re0 += 1) {
	line=RE.sub(r'for\s*\([^N]*(N[0-9])[^\n]*',
			lambda m:print_do("do_"+m.group(1)),
			line)
	line=RE.sub(r'}',
			lambda m:print_do("loop"),
			line) # only work under the assumption that indetion for while loop adds at least a tab before } and that no {} left
	
	#prune duplicated newlines
	line=RE.sub(r'\n+',r'\n',line)
	
	info("resulting file")
	output.write(line)
