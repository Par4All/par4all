# 
# $Id$
#

#debug_output := $(shell echo define_library.mk  > /dev/tty)

######################################################################## NEWGEN

newgen.libs	= genC

######################################################################## LINEAR

linear.libs	= matrice union polyedre sparse_sc sc contrainte sg sommet \
		  ray_dte polynome matrix vecteur arithmetique

######################################################################## OTHERS

other.libs	= m

##################################################################### EXTERNALS

# maybe must create a link to libpolylib64.a
extern.libs	= polylib

################################################################### PIPS COMMON

# old stuff: 
# prgm_mapping scheduling reindexing array_dfg paf-util static_controlize pip

pipslibs.libs	= \
	top-level pipsmake wp65 hpfc hyperplane \
	instrumentation statistics expressions transformations \
	movements bootstrap callgraph icfg chains complexity \
	conversion prettyprint atomizer syntax c_syntax \
	effects-simple effects-convex effects-generic alias-classes \
	comp_sections semantics control continuation rice ricedg \
	pipsdbm transformer preprocessor ri-util properties \
	text-util misc properties reductions flint sac phrase newgen

pips.libs	= \
	$(pipslibs.libs) $(newgen.libs) $(linear.libs) \
	$(extern.libs) $(other.libs)

########################################################################## PIPS

PIPS_MAIN	= main_pips.o

######################################################################### TPIPS

tpips_add.libs	= readline termcap
tpips.libs	= $(pips.libs) $(tpips_add.libs)

TPIPS_MAIN	= main_tpips.o

######################################################################### WPIPS

# The following locations should be parameterized somewhere else
# or à la autocon
X11_ROOT=/usr/X11R6
OPENWINHOME=$(X11_ROOT)

WPIPS_CPPFLAGS 	= -I$(OPENWINHOME)/include -I$(X11_ROOT)/include -Iicons
WPIPS_LDFLAGS 	= -L$(OPENWINHOME)/lib -L$(X11_ROOT)/lib

wpips_add.libs	= xview olgx X11
wpips.libs	= $(pips.libs) $(wpips_add.libs)
WPIPS_MAIN 	= main_wpips.o

######################################################################### FPIPS

ifndef PIPS_NO_WPIPS
	FPIPS_LDFLAGS	= $(WPIPS_LDFLAGS)

	# By default, compile with wpips:
	fpips_add.libs	= wpips $(wpips_add.libs)
else
	fpips_add.libs	=
endif

fpips.libs	= pips tpips $(pips.libs) $(fpips_add.libs) $(tpips_add.libs)
FPIPS_MAIN	= main_fpips.o
