#
# List of other libraries used to build the test main program
MAIN_LIBS=	-lri-util -lproperties -ltext-util -ldg-util -lmisc \
                -lproperties -lgenC -lplint -lmatrice -lpolyedre \
                -lsparse_sc -lsc -lcontrainte -lsg -lsommet -lray_dte \
                -lpolynome -lvecteur -larithmetique -lreductions -lm \
                /usr/lib/debug/malloc.o

# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=	prgm_mapping.c broadcast.c utils.c print.c vvs.c
LIB_HEADERS=	prgm_mapping-local.h
LIB_OBJECTS=	prgm_mapping.o broadcast.o utils.o print.o vvs.o

