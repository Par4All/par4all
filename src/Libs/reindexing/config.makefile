#
# List of other libraries used to build the test main program
MAIN_LIBS=	-lri-util -lproperties -ltext-util -ldg-util -lmisc \
                -lproperties -lgenC -lplint -lmatrice -lpolyedre \
                -lsparse_sc -lsc -lcontrainte -lsg -lsommet -lray_dte \
                -lpolynome -lvecteur -larithmetique -lreductions -lm \
                /usr/lib/debug/malloc.o

# Source, header and object files used to build the library.
# Do not include the main program source file.
LIB_CFILES=	reindexing.c reindexing_utils.c prettyprint.c single_assign.c \
                bounds.c delay.c cell.c
LIB_HEADERS=	reindexing-local.h
LIB_OBJECTS=	reindexing.o reindexing_utils.o prettyprint.o single_assign.o \
                bounds.o delay.o cell.o
