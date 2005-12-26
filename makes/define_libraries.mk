# 
# $Id$
#

######################################################################## NEWGEN

NEWGEN_LIBS	= -lgenC

######################################################################## LINEAR

LINEAR_LIBS	= -lmatrice -lunion -lpolyedre -lsparse_sc -lsc -lcontrainte \
		  -lsg -lsommet -lray_dte -lpolynome -lmatrix -lvecteur \
		  -larithmetique

######################################################################## OTHERS

OTHER_LIBS = 	-lm

##################################################################### EXTERNALS

EXTERN_LIBS =	-lpolylib

################################################################### PIPS COMMON

# old stuff: 
# -lprgm_mapping -lscheduling -lreindexing -larray_dfg 
# -lpaf-util -lstatic_controlize -lpip

PIPS_LIBS	= \
	-ltop-level -lpipsmake -lwp65 -lhpfc -lhyperplane \
	-linstrumentation -lstatistics -lexpressions -ltransformations \
	-lmovements -lbootstrap -lcallgraph -licfg -lchains -lcomplexity \
	-lconversion -lprettyprint -latomizer -lsyntax -lc_syntax \
	-leffects-simple -leffects-convex -leffects-generic -lalias-classes \
	-lcomp_sections -lsemantics -lcontrol -lcontinuation -lrice -lricedg \
	-lpipsdbm -ltransformer -lpreprocessor -lri-util -lproperties \
	-ltext-util -lmisc -lproperties -lreductions -lflint -lsac -lphrase \
	-lnewgen $(NEWGEN_LIBS) $(LINEAR_LIBS) $(EXTERN_LIBS) $(OTHER_LIBS)

########################################################################## PIPS

PIPS_MAIN	= main_pips.o

######################################################################### TPIPS

TPIPS_LIBS	= -lreadline -ltermcap
TPIPS_MAIN	= main_tpips.o

######################################################################### WPIPS

WPIPS_CPPFLAGS 	= -I$(OPENWINHOME)/include -I$(X11_ROOT)/include -Iicons
WPIPS_LDFLAGS 	= -L$(OPENWINHOME)/lib -L$(X11_ROOT)/lib
WPIPS_LIBS 	= -lxview -lolgx -lX11
WPIPS_MAIN 	= main_wpips.o

######################################################################### FPIPS

FPIPS_LDFLAGS	= $(WPIPS_ADDED_LDFLAGS)
FPIPS_LIBS	= -lpips -ltpips -lwpips $(TPIPS_LIBS) $(WPIPS_LIBS)
FPIPS_MAIN	= main_fpips.o
