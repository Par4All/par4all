#
# List of other libraries used to build the test main program
MAIN_LIBS=	-lri-util -lproperties -ltext-util -ldg-util -lmisc \
                -lproperties -lgenC -lplint -lmatrice -lpolyedre \
                -lsparse_sc -lsc -lcontrainte -lsg -lsommet -lray_dte \
                -lpolynome -lvecteur -larithmetique -lreductions -lm \
                /usr/lib/debug/malloc.o

# to enable local parsing...
# CPPFLAGS+=-DHAS_BDTYY

# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES =	adg_read_paf.c bdt_read_paf.c utils.c print.c
LIB_HEADERS =	paf-util-local.h
LIB_OBJECTS =	$(LIB_CFILES:.c=.o)

