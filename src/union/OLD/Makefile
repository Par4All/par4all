# Suffix rule added to compile on the RS6000.
# .c.o:
#	$(CC) $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c $<

#PROFILING= -pg -DNDEBUG -O3

#
# you can add your own options behind pips default values
# 
# example: CFLAGS=		$(PIPS_CFLAGS) -DSYSTEM=BSD4.2
#
AR=		$(PIPS_AR)
ARFLAGS=	$(PIPS_ARFLAGS)
CC=		$(PIPS_CC)
CFLAGS=		$(PIPS_CFLAGS) $(PROFILING) -I../include.dir
CPP=		/lib/cpp 
CPPFLAGS=	-I../include.dir $(PIPS_CPPFLAGS)
LD=		$(PIPS_LD)
LDFLAGS=	$(PIPS_LDFLAGS) $(PROFILING)
LEX=		$(PIPS_LEX)
LFLAGS=		$(PIPS_LFLAGS) 
LINT=		$(PIPS_LINT)
LINTFLAGS=	$(PIPS_LINTFLAGS) -DLINT_IS_USED -q -DDEBUG
RANLIB=		$(PIPS_RANLIB)
YACC=		$(PIPS_YACC) 
YFLAGS=		$(PIPS_YFLAGS) -d

#
# do not move or modify the next six lines
#
COMPILE=	$(CC) $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c
LINK=		$(LD) $(LDFLAGS) -o
SCAN=		$(LEX) $(LFLAGS) -t
TYPECHECK=	$(LINT) $(LINTFLAGS) $(CPPFLAGS) $(TARGET_ARCH)
PARSE=		$(YACC) $(YFLAGS)
ARCHIVE=	$(AR) $(ARFLAGS)

MAKEFILE= Makefile
ROOTDIR= 	$(PIPSDIR)
TARGET=		union.a
INCLUDE= 
LIBS=

#	Directory where library is installed
# INSLIBDIR:sh	= echo $LIBDIR/`directory-name ..`

#	Directory where source files are copied
INSSRCDIR=  ../../Production/`directory-name .`

HEADERS	= union-types.h 
# Pour la doc : on ne prend pas les commentaires du lexer/yaccer
DOCCFILES = sc_list.c  disjunct.c path.c reduc.c
FFILES	= 
ASFILES	= 
YACCFILES= sl_gram.y
LEXFILES= sl_lex.l	
SHFILES	= 
MANFILES= 
DOCFILES= 
# OTHERS	= UNION.w disjunct.w path.w sl_io.w sc_list.w 
CFILES	= sc_list.c sl_gram.c sl_lex.c disjunct.c path.c reduc.c
OBJECTS	= sc_list.o sl_gram.o sl_lex.o disjunct.o path.o reduc.o

SOURCE	= $(MAKEFILE) $(MANFILES) $(DOCFILES) $(SHFILES) $(OTHERS) \
	$(HEADERS) $(YACCFILES) $(LEXFILES) $(CFILES) \
	$(FFILES) $(ASFILES)
LISTFILES= $(MAKEFILE) $(SHFILES) $(HEADERS) \
	$(YACCFILES) $(LEXFILES) $(CFILES) $(FFILES) $(ASFILES)

# add here all dependencies that cannot be automatically computed
sl_gram.o: 	sl_gram.c y.tab.h
sl_gram.c: sl_gram.y
	$(YACC) sl_gram.y
	sed -e 's/YY/SLYY/g;s/yy/slyy/g' <y.tab.c >sl_gram.c

sl_lex.o: y.tab.h sl_lex.c
sl_lex.c: sl_lex.l y.tab.h
	$(LEX) $(LEXFLAGS) sl_lex.l
	sed -e 's/YY/SLYY/g;s/yy/slyy/g' <lex.yy.c >sl_lex.c

y.tab.h: sl_gram.y
	$(YACC) -d sl_gram.y



# dependencies on sub-types arithmetique and vecteur and Co:
all:	arithmetique \
	vecteur \
	contrainte \
	sc \
	polyedre \
	ray_dte \
	sommet \
	sg \
	$(TARGET)

# global recompilation of linear; consistency assumed
linear:	$(TARGET)

sg:
	(cd ../sg.dir; $(MAKE) linear)

sommet:
	(cd ../sommet.dir; $(MAKE) linear)

ray_dte:
	(cd ../ray_dte.dir; $(MAKE) linear)

polyedre:
	(cd ../polyedre.dir; $(MAKE) linear)

sc:
	(cd ../sc.dir; $(MAKE) linear)

contrainte:
	(cd ../contrainte.dir; $(MAKE) linear)

vecteur:
	(cd ../vecteur.dir; $(MAKE) linear)

arithmetique:
	(cd ../arithmetique.dir; $(MAKE) linear)

$(TARGET): ../include.dir/union.h $(OBJECTS) 
	-rm -f $(TARGET) 
	$(ARCHIVE) $(TARGET) `lorder $(OBJECTS) | tsort`
	-ranlib $(TARGET)

install: $(TARGET)
	rm -rf $(INSSRCDIR)
	mkdir $(INSSRCDIR)
	cp $(SOURCE) $(INSSRCDIR)
	(cd $(INSSRCDIR) ; $(MAKE) compile)

debug_c3:
	-rm -f $(OBJECTS)
	make $(TARGET) CFLAGS="$(CFLAGS) -DDEBUG_UNION_C3"

debug_pips:
	-rm -f $(OBJECTS)
	make $(TARGET) CFLAGS="$(CFLAGS) -DDEBUG_UNION_PIPS -DTRACE_PIPS_PATH"

pips_install: ../include.dir/union.h $(OBJECTS)
	-rm -f path.o
	$(CC) $(CFLAGS) $(CPPFLAGS) -DTRACE_PIPS_PATH  $(TARGET_ARCH) -c path.c
	-rm -f pips_union.a
	$(ARCHIVE) pips_union.a `lorder $(OBJECTS) | tsort`
	$(RANLIB) pips_union.a
	rm -f path.o
	cp pips_union.a ${EXTEDIR}/libunion.a
	cp union.h      ${EXTEDIR}/union.h

bc_install: ../include.dir/union.h $(OBJECTS)
	-rm -f path.o
	$(CC) $(CFLAGS) $(CPPFLAGS) -DDEBUG_UNION_PIPS -DTRACE_PIPS_PATH  $(TARGET_ARCH) -c sc_list.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -DDEBUG_UNION_PIPS -DTRACE_PIPS_PATH  $(TARGET_ARCH) -c path.c
	-rm -f pips_union.a
	$(ARCHIVE) pips_union.a `lorder $(OBJECTS) | tsort`
	-ranlib pips_union.a
	rm -f path.o
	cp pips_union.a ${DEVEDIR}/Lib/regions/libunion.a


export: $(CFILES)
	(cd .. ; rm -rf union-export.dir )
	sed -e 's/^#line.*$$//g' < sc_list.c >_sc_list.c
	rm sc_list.c 
	mv _sc_list.c sc_list.c
	sed -e 's/^#line.*$$//g' < disjunct.c >_disjunct.c
	rm disjunct.c 
	mv _disjunct.c disjunct.c
	sed -e 's/^#line.*$$//g' < path.c >_path.c
	rm path.c 
	mv _path.c path.c
	(cd .. ; mkdir union-export.dir)
	cp $(CFILES)      ../union-export.dir
	cp union-types.h  ../union-export.dir
	cp Makefile       ../union-export.dir


compile: $(TARGET)

union.h: ../include.dir/auto.h union-types.h doc.h
	@cproto $(CPPFLAGS) $(CFILES) | \
                sed -e 's/int __builtin_va_alist//' | \
	cat ../include.dir/auto.h union-types.h - doc.h > union.h.new
	@if cmp -s union.h union.h.new; \
		then \
		echo union.h is up to date >/dev/null; \
		rm -f union.h.new; \
		else \
		echo updating union.h; \
		mv union.h.new union.h; \
		fi

../include.dir/union.h: union.h
	@if cmp -s union.h ../include.dir/union.h; \
		then \
		echo ../include.dir/union.h is up to date >/dev/null; \
		else \
		echo cp union.h ../include.dir/union.h; \
		cp union.h ../include.dir/union.h; \
		fi

extern.h:
	@make-extern $(CFILES) > extern.h

doc.h: $(DOCCFILES)
	@awk -f $(UTILDIR)/extract-doc.awk $(DOCCFILES) > doc.h

tar:
	tar rfcb /dev/rht0 20 $(SOURCE)
cpio:
	ls $(SOURCE) | cpio -oB
netsend:
	netsend $(dest) $(SOURCE)
llib-lpolyedre.ln: $(CFILES) $(HEADERS)
	$(TYPECHECK)  -Cpolyedre $(CFILES) \
		 | sed '/possible pointer alignment/d'
lint: llib-lpolyedre.ln
	$(TYPECHECK) $(CFILES) ../arithmetique.dir/llib-larithmetique.ln \
	../vecteur.dir/llib-lvecteur.ln  \
	../contrainte.dir/llib-lcontrainte.ln \
	../sc.dir/llib-lsc.ln \
	../ray_dte.dir/llib-lray_dte.ln \
	../sommet.dir/llib-lsommet.ln \
	../sg.dir/llib-lsg.ln \
	 | sed '/possible pointer alignment/d'
clean:
	-rm -f $(OBJECTS) TAGS polyedre.h doc.h extern.h llib-lpolyedre.ln
clobber: clean
	-rm -f $(TARGET)
touch:
	touch $(TARGET)
print:	list60
list:	list66
list66:
	@pr -n -w132 -l66 $(LISTFILES)
list60:
	@pr -n -w132 -l60 $(LISTFILES)
list51:
	@pr -n -w132 -l51 $(LISTFILES)
TAGS: $(CFILES) union-types.h
	etags -t $(CFILES) union-types.h
vgrind:
	@vgrind $(LISTFILES)

depend:
	sed -n -e '1,/^### DO NOT DELETE THIS LINE./p' < $(MAKEFILE) > $(MAKEFILE).new
	-for i in $(YACCFILES) $(LEXFILES) $(CFILES) $(FFILES) ; do\
		base=`expr $$i ':' '\(.*\).[cylf]$$'`;\
		suffix=`expr $$i ':' '.*\.\([cylf]\)$$'`;\
		if test $$suffix = l ; then\
			lex $$i;\
			mv lex.yy.c $$base.c;\
			suffix=c;\
			/bin/echo "$$base.c:	$$base.l" >> $(MAKEFILE).new;\
		elif test $$suffix = y ; then\
			yacc $(YFLAGS) $$i;\
			mv y.tab.c $$base.c;\
			suffix=c;\
			/bin/echo "$$base.c:	$$base.y" >> $(MAKEFILE).new;\
			/bin/echo "y.tab.h:	$$base.y" >> $(MAKEFILE).new;\
		fi;\
		$(CPP) -I../include.dir -M $$base.$$suffix >> $(MAKEFILE).new;\
	done;
	cp $(MAKEFILE).new $(MAKEFILE)
	rm -f $(MAKEFILE).new



### The following dependancies are/can be generated automatically
### by 'make depend'. Listen to this warning
###
### Do NOT put any of your own dependancies below this line,
### they will be removed
### DO NOT DELETE THIS LINE. USED FOR MAKE DEPEND
sl_gram.c:	sl_gram.y
y.tab.h:	sl_gram.y
sl_gram.o: sl_gram.c
sl_gram.o: /usr/include/stdio.h
sl_gram.o: /usr/include/string.h
sl_gram.o: /usr/include/sys/stdtypes.h
sl_gram.o: ../include.dir/boolean.h
sl_gram.o: ../include.dir/arithmetique.h
sl_gram.o: ../include.dir/vecteur.h
sl_gram.o: ../include.dir/contrainte.h
sl_gram.o: ../include.dir/sc.h
sl_gram.o: ../include.dir/arithmetique.h
sl_gram.o: /usr/include/setjmp.h
sl_gram.o: /usr/include/machine/setjmp.h
sl_gram.o: /usr/include/malloc.h
sl_gram.o: ../include.dir/union.h
sl_gram.o: /usr/include/malloc.h
sl_gram.o: /usr/include/memory.h
sl_gram.o: /usr/include/values.h
sl_lex.c:	sl_lex.l
sl_lex.o: sl_lex.c
sl_lex.o: /usr/include/stdio.h
sl_lex.o: /usr/include/stdio.h
sl_lex.o: ./y.tab.h
sc_list.o: sc_list.c
sc_list.o: /usr/include/stdlib.h
sc_list.o: /usr/include/sys/stdtypes.h
sc_list.o: /usr/include/stdio.h
sc_list.o: /usr/include/string.h
sc_list.o: /usr/include/sys/stdtypes.h
sc_list.o: /usr/include/setjmp.h
sc_list.o: /usr/include/machine/setjmp.h
sc_list.o: ../include.dir/assert.h
sc_list.o: /usr/include/time.h
sc_list.o: /usr/include/sys/stdtypes.h
sc_list.o: /usr/include/sys/time.h
sc_list.o: /usr/include/time.h
sc_list.o: ../include.dir/boolean.h
sc_list.o: ../include.dir/arithmetique.h
sc_list.o: ../include.dir/vecteur.h
sc_list.o: ../include.dir/contrainte.h
sc_list.o: ../include.dir/sc.h
sc_list.o: ../include.dir/arithmetique.h
sc_list.o: /usr/include/setjmp.h
sc_list.o: ../include.dir/sommet.h
sc_list.o: ../include.dir/polyedre.h
sc_list.o: ../include.dir/ray_dte.h
sc_list.o: ../include.dir/sg.h
sc_list.o: ../include.dir/union.h
sl_gram.o: sl_gram.c
sl_gram.o: /usr/include/stdio.h
sl_gram.o: /usr/include/string.h
sl_gram.o: /usr/include/sys/stdtypes.h
sl_gram.o: ../include.dir/boolean.h
sl_gram.o: ../include.dir/arithmetique.h
sl_gram.o: ../include.dir/vecteur.h
sl_gram.o: ../include.dir/contrainte.h
sl_gram.o: ../include.dir/sc.h
sl_gram.o: ../include.dir/arithmetique.h
sl_gram.o: /usr/include/setjmp.h
sl_gram.o: /usr/include/machine/setjmp.h
sl_gram.o: /usr/include/malloc.h
sl_gram.o: ../include.dir/union.h
sl_gram.o: /usr/include/malloc.h
sl_gram.o: /usr/include/memory.h
sl_gram.o: /usr/include/values.h
sl_lex.o: sl_lex.c
sl_lex.o: /usr/include/stdio.h
sl_lex.o: /usr/include/stdio.h
sl_lex.o: ./y.tab.h
disjunct.o: disjunct.c
disjunct.o: /usr/include/stdlib.h
disjunct.o: /usr/include/sys/stdtypes.h
disjunct.o: /usr/include/stdio.h
disjunct.o: /usr/include/string.h
disjunct.o: /usr/include/sys/stdtypes.h
disjunct.o: /usr/include/setjmp.h
disjunct.o: /usr/include/machine/setjmp.h
disjunct.o: ../include.dir/assert.h
disjunct.o: /usr/include/time.h
disjunct.o: /usr/include/sys/stdtypes.h
disjunct.o: /usr/include/sys/time.h
disjunct.o: /usr/include/time.h
disjunct.o: ../include.dir/boolean.h
disjunct.o: ../include.dir/arithmetique.h
disjunct.o: ../include.dir/vecteur.h
disjunct.o: ../include.dir/contrainte.h
disjunct.o: ../include.dir/sc.h
disjunct.o: ../include.dir/arithmetique.h
disjunct.o: /usr/include/setjmp.h
disjunct.o: ../include.dir/sommet.h
disjunct.o: ../include.dir/polyedre.h
disjunct.o: ../include.dir/ray_dte.h
disjunct.o: ../include.dir/sg.h
disjunct.o: ../include.dir/union.h
path.o: path.c
path.o: /usr/include/stdlib.h
path.o: /usr/include/sys/stdtypes.h
path.o: /usr/include/stdio.h
path.o: /usr/include/string.h
path.o: /usr/include/sys/stdtypes.h
path.o: /usr/include/setjmp.h
path.o: /usr/include/machine/setjmp.h
path.o: ../include.dir/assert.h
path.o: /usr/include/time.h
path.o: /usr/include/sys/stdtypes.h
path.o: /usr/include/sys/time.h
path.o: /usr/include/time.h
path.o: ../include.dir/boolean.h
path.o: ../include.dir/arithmetique.h
path.o: ../include.dir/vecteur.h
path.o: ../include.dir/contrainte.h
path.o: ../include.dir/sc.h
path.o: ../include.dir/arithmetique.h
path.o: /usr/include/setjmp.h
path.o: ../include.dir/sommet.h
path.o: ../include.dir/polyedre.h
path.o: ../include.dir/ray_dte.h
path.o: ../include.dir/sg.h
path.o: ../include.dir/union.h
