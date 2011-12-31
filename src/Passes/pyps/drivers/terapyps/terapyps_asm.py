import re as RE
import operator
from logging import info, getLogger

getLogger().setLevel(1)

def conv(input,output):
    info("reading input file "+input)
    fd=open(input)
    line=reduce(operator.add,fd.readlines())
    fd.close()
    
    info("pruning trailing comments, labels, declarations and indentation")
    # this is for the comments
    line=RE.sub(r'\n\s*/\*(.*?)\*/\s*\n',r'\n*\1\n',line)
    line=RE.sub(r'^\s*/\*(.*?)\*/\s*\n',r'\n*\1\n',line)
    line=RE.sub(r'\s*//.*\n',r'\n',line)
    line=RE.sub(r'\s*#pragma.*\n',r'\n',line)
    line=RE.sub(r'(/\*.*?\*/)|(;)',r'',line)
    line=RE.sub(r'[ \t]*\n+[ \t]*',r'\n',line)
    # this is for the declarations
    line=RE.sub(r'(short|int)\s*\*?\s*(\w+)(\s*,\s*\*?\s*(\w+))*',r'',line)
    # this is for labels
    line=RE.sub(r'\n\s*\w+:\s*',r'',line)

    # are we in degraded mode ?
    degraded_mode=line.find("P=")>=0
    
    info("managing substitution")
    if output.name[-4:] == ".asm":
        def print_fields(fields):
            padding=24
            max_delay=5
            res=fields[0].center(padding)
            for field in fields[1:]:
                res+='||'+field.center(padding)
            res+='\n'
            for i in range(max_delay):
                res+="".center(padding)
                for field in ["","","",""]:
                    res+='||'+field.center(padding)
                res+='\n'
            return res
    elif output.name[-4:] == ".asl":
        def print_fields(fields):
            res=''
            for field in fields:
                if field: res+= field
            return res.strip() + '\n'

    
    def print_im(lhs,rhs):   return print_fields(["im"+(','+lhs if lhs else '') +'='+rhs,' ',' ',' ',' '])
    def print_ma(lhs,rhs):   return print_fields([' ',"ma"+(","+lhs if lhs else '') +'='+rhs,' ',' ',' '])
    def print_re(lhs, rhs):  return print_fields([' ',' ',"P"+ (','+lhs if lhs else '') + "=" + rhs,' ',' '])
    def print_do(field):     return print_fields([' ',' ',' ',field,' '])
    def print_return(field): return print_fields([' ',' ',' ',' ',field])
    
    def re(m,i):return "re("+m.group(i)+")"
    def im(m,i):return "i"+m.group(i)
    def ma(m,i):return "m"+m.group(i)
    
    #imX = FIFOY
    line=RE.sub(r'im([0-9]+) = (FIFO[0-9]+)',
            lambda m:print_im(im(m,1),m.group(2)),
            line)
    #maX = FIFOY
    line=RE.sub(r'ma([0-9]+) = (FIFO[0-9]+)',
            lambda m:print_ma(ma(m,1),m.group(2)),
            line)
    #imX = imY+Z*S
    line=RE.sub(r'im([0-9]+) = im([0-9]+)([\+-])([0-9]+\*)S',
            lambda m:print_im(im(m,1),im(m,2)+"+"+m.group(4)+("S" if m.group(3)=="+" else "N")),
            line)
    #maX = maY+Z
    line=RE.sub(r'ma([0-9]+) = ma([0-9]+)([\+\-])([0-9]+)',
            lambda m:print_ma(ma(m,1),ma(m,2)+"+"+m.group(4)+"*"+("E" if m.group(3)=="+" else "W" ) ),
            line)
    #imX = imY+Z
    line=RE.sub(r'im([0-9]+) = im([0-9]+)([\+\-])([0-9]+)',
            lambda m:print_im(im(m,1),im(m,2)+"+"+m.group(4)+"*"+("E" if m.group(3)=="+" else "W" ) ),
            line)
    #*++imX = reY
    line=RE.sub(r'\*\+\+im([0-9]+) = re([0-9]+)',
            lambda m:print_im(im(m,1),im(m,1)+"+1*E")+print_re("im",re(m,2)+"*re(0)"),
            line)
    #*imX = reY
    line=RE.sub(r'\*im([0-9]+) = re([0-9]+)',
            lambda m:print_im("",im(m,1))+print_re("",re(m,2))+print_re("im","P"),
            line)
    #*++imX = reY+reZ
    line=RE.sub(r'\*\+\+im([0-9]+) = re([0-9]+)\+re([0-9]+)',
            lambda m:print_re("",re(m,2))+print_im(im(m,1),im(m,1)+"+1*E")+print_re("im","P+"+re(m,3)+"*re(0)"),
            line)
    #*++imX = P
    line=RE.sub(r'\*\+\+im([0-9]+) = P',
            lambda m:print_im(im(m,1),im(m,1)+"+1*E")+print_re("im","P"),
            line)
    #*imX = P
    line=RE.sub(r'\*im([0-9]+) = P',
            lambda m:print_im("",im(m,1))+print_re("im","P"),
            line)
    #reX = *++imY*reZ
    line=RE.sub(r're([0-9]+) = \*\+\+im([0-9]+)\*re([0-9]+)',
            lambda m:print_re("",re(m,3))+print_im(im(m,2),im(m,2)+"+1*E")+print_re(re(m,1),"P+im*re(0)"),
            line)
    #P = *++imY*reZ
    line=RE.sub(r'P = \*\+\+im([0-9]+)\*re([0-9]+)',
            lambda m:print_im(im(m,1),im(m,1)+"+1*E")+print_re("","im*"+re(m,2)),
            line)
    #reX = *++imY**++maZ
    line=RE.sub(r're([0-9]+) = \*\+\+im([0-9]+)\*\*\+\+ma([0-9]+)',
            lambda m:print_im(im(m,2),im(m,2)+"+1*E")+print_ma(ma(m,3),ma(m,3)+"+1*E")+print_re(re(m,1),"im*ma"),
            line)
    #reX = reY+*imZ**maW
    line=RE.sub(r're([0-9]+) = re([0-9]+)([\+\*])\*im([0-9]+)([\+\*])\*ma([0-9]+)',
            lambda m:print_im("",im(m,4))+print_ma("",ma(m,6))+print_re("","im"+m.group(5)+"ma")+print_re(re(m,1),"P"+m.group(3)+re(m,2)+"*re(0)"),
            line)
    #*imX = *imY+*imZ
    line=RE.sub(r'\*im([0-9]+) = \*im([0-9]+)([\+\*])\*im([0-9]+)',
            lambda m:(print_re("re(1)","P") if degraded_mode else "" )+print_im("",im(m,4))+print_re("","im*re(0)")+print_im("",im(m,2))+print_re("","P"+m.group(3)+"im*re(0)")+print_im("",im(m,1))+print_re("im","P")+(print_re("","re(1)") if degraded_mode else ""),
            line)
    #*imX = *imY+*maZ
    line=RE.sub(r'\*im([0-9]+) = \*im([0-9]+)([\+\*])\*ma([0-9]+)',
            lambda m:(print_re("re(1)","P") if degraded_mode else "" )+print_im("",im(m,2))+print_ma("",ma(m,4))+print_re("","P"+m.group(3)+"im*ma")+print_im("",im(m,1))+print_re("im","P")+(print_re("","re(1)") if degraded_mode else ""),
            line)
    #*(++imX) = *(++imY)+*maZ
    line=RE.sub(r'\*\+\+im([0-9]+) = \*\+\+im([0-9]+)([\+\*])\*ma([0-9]+)',
            lambda m:print_im(im(m,2),im(m,2)+"+1*E")+print_re("","im*re(0)")+print_ma("",ma(m,4))+print_im(im(m,1),im(m,1)+"+1*E")+print_re("im","P+re(1)*ma"),
            line)
    #P = P+*imZ**maW
    line=RE.sub(r'P = P([\+\*])\*im([0-9]+)([\+\*])\*ma([0-9]+)',
            lambda m:print_im("",im(m,2))+print_ma("",ma(m,4))+print_re("","P"+m.group(1)+"im"+m.group(3)+"ma"),
            line)
    #P = P+*++imZ**++maW
    line=RE.sub(r'P = P([\+\*])\*\+\+im([0-9]+)([\+\*])\*\+\+ma([0-9]+)',
            lambda m:print_im(im(m,2),im(m,2)+"+1*E")+print_ma(ma(m,4),ma(m,4)+"+1*E")+print_re("","P"+m.group(1)+"im"+m.group(3)+"ma"),
            line)

    #reX = *(maY+X)
    line=RE.sub(r're([0-9]+) = \*\(ma([0-9]+)\+([0-9]+)\)',
            lambda m:print_ma("",ma(m,2)+"+"+m.group(3)+"*E")+print_re(re(m,1),"re(0)*ma"),
            line)

    #reX = *maY
    line=RE.sub(r're([0-9]+) = \*ma([0-9]+)',
            lambda m:print_ma("",ma(m,2))+print_re(re(m,1),"re(0)*ma"),
            line)

    #imX = imY
    line=RE.sub(r'im([0-9]+) = im([0-9]+)',
            lambda m:print_im(im(m,1),im(m,2)),
            line)

    #reX = Y (special case for Y=0)
    line=RE.sub(r're([0-9]+) = ([0-9]+)',
            lambda m:print_re(re(m,1),(m.group(2) if m.group(2)!="0" else "P-P")),
            line)
    #P = Y (special case for Y=0)
    line=RE.sub(r'P = ([0-9]+)',
            lambda m:print_re("",(m.group(1) if m.group(1)!="0" else "P-P")),
            line)
    #reX = reY + reZ
    line=RE.sub(r're([0-9]+) = re([0-9]+)\+re([0-9]+)',
            lambda m:print_re("",re(m,2))+print_re(re(m,1),"P+"+re(m,3)+"*re(0)"),
            line)
    #reX = reY
    line=RE.sub(r're([0-9]+) = re([0-9]+)',
            lambda m:print_re(re(m,1),re(m,2)),
            line)
    #P = reY
    line=RE.sub(r'P = re([0-9]+)',
            lambda m:print_re("",re(m,1)),
            line)
    #maX = maY
    line=RE.sub(r'ma([0-9]+) = ma([0-9]+)',
            lambda m:print_ma(ma(m,1),ma(m,2)),
            line)
    #*imX = MIN(reY,*imZ)
    def handler0(m):
        # it's ok to use re(0-5) they are reserved for internal use
        s=print_im("",im(m,4))
        if m.group(2) == "MIN":
            s+=print_re("As",re(m,3)+"-im*re(0)")
        else:
            s+=print_re("As","im*re(0)-"+re(m,3))
        s+=print_re("re(2)","im*re(0)")
        s+=print_re("",re(m,3))
        s+=print_im("",im(m,1))
        s+=print_re("im","if(As=1,P,re(2))")
        return s
    line=RE.sub(r'\*im([0-9]+) = (MIN|MAX)\(re([0-9]+), \*im([0-9]+)\)',
            handler0,
            line)
    #*++imX = MIN(reY,*++imZ)
    def handler0(m):
        # it's ok to use re(0-5) they are reserved for internal use
        s=print_im("",im(m,4)+"+1*E")
        if m.group(2) == "MIN":
            s+=print_re("As",re(m,3)+"-im*re(0)")
        else:
            s+=print_re("As","im*re(0)-"+re(m,3))
        s+=print_re("re(2)","im*re(0)")
        s+=print_re("",re(m,3))
        s+=print_im("",im(m,1)+"+1*E")
        s+=print_re("im","if(As=1,P,re(2))")
        return s
    line=RE.sub(r'\*\+\+im([0-9]+) = (MIN|MAX)\(re([0-9]+), \*\+\+im([0-9]+)\)',
            handler0,
            line)
    #P = MIN(P,*imZ)
    def handler1(m):
        # it's ok to use re(0-5) they are reserved for internal use
        s=print_im("",im(m,2))
        s+=print_re("re(1)","P")
        s+=print_re("As","P-im*re(0)")
        if m.group(1) == "MIN":
            s+=print_re("","if(As=1,P,re(1))")
        else:
            s+=print_re("","if(As=0,P,re(1))")
        return s
    line=RE.sub(r'P = (MIN|MAX)\(P, \*im([0-9]+)\)',
            handler1,
            line)


    #reX = MIN(reY,*imZ)
    def handler1(m):
        # it's ok to use re(0-5) they are reserved for internal use
        s=print_im("",im(m,4))
        if m.group(2) == "MIN":
            s+=print_re("As",re(m,3)+"-im*re(0)")
        else:
            s+=print_re("As","im*re(0)-"+re(m,3))
        s+=print_re(re(m,1),"im*re(0)")
        s+=print_re("",re(m,3))
        s+=print_re(re(m,1),"if(As=1,P,"+re(m,1)+")")
        return s
    line=RE.sub(r're([0-9]+) = (MIN|MAX)\(re([0-9]+), \*im([0-9]+)\)',
            handler1,
            line)

    #reX = MIN(reY,*++imZ)
    def handler1(m):
        # it's ok to use re(0-5) they are reserved for internal use
        s=print_im("",im(m,4)+"+1*E")
        if m.group(2) == "MIN":
            s+=print_re("As",re(m,3)+"-im*re(0)")
        else:
            s+=print_re("As","im*re(0)-"+re(m,3))
        s+=print_re(re(m,1),"im*re(0)")
        s+=print_re("",re(m,3))
        s+=print_re(re(m,1),"if(As=1,P,"+re(m,1)+")")
        return s
    line=RE.sub(r're([0-9]+) = (MIN|MAX)\(re([0-9]+), \*\+\+im([0-9]+)\)',
            handler1,
            line)

    #reX=MIN(*imY,*imZ)
    def handler2(m):
        # it's ok to use re(0-5) they are reserved for internal use
        s =print_im("",im(m,3))
        s+=print_re("re(1)","im*re(0)")
        s+=print_im("",im(m,4))
        if m.group(2) == "MIN":
            s+=print_re("As","P-im*re(0)")
        else:
            s+=print_re("As","im*re(0)-P")
        s+=print_re("re(2)","im*re(0)")
        s+=print_re("","re(1)")
        s+=print_re(re(m,1),"if(As=1,P,re(2))")
        return s

    line=RE.sub(r're([0-9]+) = (MIN|MAX)\(\*im([0-9]+), \*im([0-9]+)\)',
            handler2,
            line)
    #reX=MIN(*++imY,*++imZ)
    def handler2(m):
        # it's ok to use re(0-5) they are reserved for internal use
        s =print_im("",im(m,3)+"+1*E")
        s+=print_re("re(1)","im*re(0)")
        s+=print_im("",im(m,4)+"+1*E")
        if m.group(2) == "MIN":
            s+=print_re("As","P-im*re(0)")
        else:
            s+=print_re("As","im*re(0)-P")
        s+=print_re("re(2)","im*re(0)")
        s+=print_re("","re(1)")
        s+=print_re(re(m,1),"if(As=1,P,re(2))")
        return s

    line=RE.sub(r're([0-9]+) = (MIN|MAX)\(\*\+\+im([0-9]+), \*\+\+im([0-9]+)\)',
            handler2,
            line)

    # void microcode(short *FIFO0, short *FIFO1, short *FIFO2, short iter1, short iter2)
    line=RE.sub(r'void ([a-zA-Z]\w*)\(.*?\)\s*{',
            lambda m: 'prog '+m.group(1)+ "\nsub "+m.group(1) +'\n' + print_re("re(0)","1"), #re(0) is reserved for this value
            line)
    line=RE.sub(r'}\s*$',print_return("return")+r'endsub\nendprog\n',line)
    
    # for(re0 = 0 re0 <= N0 re0 += 1) {
    line=RE.sub(r'for\s*\([^N]*(N[0-9])[^\n]*',
            lambda m:print_do("do_"+m.group(1)),
            line)
    line=RE.sub(r'}',
            lambda m:print_do("loop"),
            line) # only work under the assumption that indetion for while loop adds at least a tab before } and that no {} left
    
    #prune duplicated newlines
    line=RE.sub(r'\n+',r'\n',line)
    if output.name[-4:] == ".asl":
        info("add some tabbing") 
        tline=""
        tab=0
        for l in line.split('\n'):
            if l[:4] == "loop":
                tab-=1
            tline+=" "*tab*4 + l + '\n'
            if l[:4] == "do_N":
                tab+=1
        line=tline
    # small optimization step
    line=RE.sub(r'((im,i([0-9]+)=[^\n]*\n)\s*im=i([0-9]+)\n)',
            lambda m:m.groups(0)[1] if m.groups(0)[2] == m.groups(0)[3] else m.groups(0)[0],
            line)
    line=RE.sub(r'\n+',r'\n',line)
    info("resulting file")
    output.write(line)
