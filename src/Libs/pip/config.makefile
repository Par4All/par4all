#
# List of other libraries used to build the test main program
MAIN_LIBS=	-lri-util -lproperties -ltext-util -ldg-util -lmisc \
                -lproperties -lgenC -lplint -lmatrice -lpolyedre \
                -lsparse_sc -lsc -lcontrainte -lsg -lsommet -lray_dte \
                -lpolynome -lvecteur -larithmetique -lreductions -lm \
                /usr/lib/debug/malloc.o

# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=	pip.c ps_max_to_min.c ps_to_fic_pip.c solpip.c \
		pip_interface.c sol.c traiter.c tab.c integrer.c maind.c
LIB_HEADERS=	pip-local.h solpip_parse.y solpip_scan.l type.h tab.h sol.h
LIB_OBJECTS=	pip.o ps_max_to_min.o ps_to_fic_pip.o solpip.o \
		solpip_parse.o solpip_scan.o pip_interface.o sol.o traiter.o tab.o integrer.o maind.o

solpip_parse.o : solpip_parse.c
	${CC} $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c -g $< -o $@

solpip_scan.o : solpip_scan.c solpip_parse.h
	${CC} $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c -g $< -o $@

solpip_parse.c solpip_parse.h : solpip_parse.y
	${YACC} -d solpip_parse.y
	sed -e '/extern char \*malloc/d;s/YY/QUAYY/g;s/yy/quayy/g' \
		y.tab.c > solpip_parse.c
	rm y.tab.c
	sed -e 's/YY/QUAYY/g;s/yy/quayy/g' y.tab.h > solpip_parse.h
	rm y.tab.h

solpip_scan.c : solpip_scan.l
	${LEX} solpip_scan.l
	sed '/^FILE* yyin/s/=[^,;]*//g;s/YY/QUAYY/g;s/yy/quayy/g;' \
		lex.yy.c > solpip_scan.c
	rm lex.yy.c

