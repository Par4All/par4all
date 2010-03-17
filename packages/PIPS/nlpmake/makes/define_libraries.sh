#
# $Id$
#
# this file define newgen, linear and pips libraries.
# it is to be included by some shell-scripts and makefiles (after a sed).

#
# NEWGEN

NEWGEN_DOCS='doc'
NEWGEN_ORDERED_LIBS='scripts genC'

NEWGEN_LIBS='-lgenC'

#
# C3/LINEAR

LINEAR_DOCS=''

# plint
LINEAR_ORDERED_LIBS='arithmetique vecteur contrainte sc matrice matrix ray_dte sommet sg polynome polyedre sparse_sc union'

# removed because not used by pips: plint
# removed because made external: polylib
LINEAR_LIBS='-lmatrice -lunion -lpolyedre -lsparse_sc -lsc -lcontrainte -lsg -lsommet -lray_dte -lpolynome -lmatrix -lvecteur -larithmetique'

EXTERN_LIBS='-lpolylib'

#
# PIPS

# static ordering for bootstraping PIPS...

# many doc dirs are skipped (would be long?)
PIPS_DOCS='Documentation constants help manl newgen'

PIPS_ORDERED_LIBS='xpomp text-util properties misc ri-util newgen conversion movements pipsdbm comp_sections transformer bootstrap control hyperplane flint syntax c_syntax prettyprint static_controlize effects effects-generic effects-simple semantics complexity continuation reductions regions effects-convex alias-classes callgraph icfg paf-util pip ricedg array_dfg prgm_mapping scheduling reindexing chains rice hyperplane transformations expressions statistics instrumentation hpfc atomizer sac phrase wp65 preprocessor pipsmake top-level pips tpips wpips fpips'

# old.
paf_libs='-lprgm_mapping -lscheduling -lreindexing -larray_dfg -lpaf-util -lstatic_controlize -lpip'

# all libraires for pips
PIPS_LIBS='-ltop-level -lpipsmake -lwp65 -lhpfc -lhyperplane -linstrumentation -lstatistics -lexpressions -ltransformations -lmovements -lbootstrap -lcallgraph -licfg -lchains -lcomplexity -lconversion -lprettyprint -latomizer -lsyntax -lc_syntax -leffects-simple -leffects-convex -leffects-generic -lalias-classes -lcomp_sections -lsemantics -lcontrol -lcontinuation -lrice -lricedg -lpipsdbm -ltransformer -lpreprocessor -lri-util -lproperties -ltext-util -lmisc -lproperties -lreductions -lflint -lsac -lphrase -lnewgen $(NEWGEN_LIBS) $(LINEAR_LIBS) $(EXTERN_LIBS) -lm -lrx'


#
# X11

PIPS_X11_ADDED_CPPFLAGS='-I$(X11_ROOT)/include'
PIPS_X11_ADDED_LDFLAGS='-L$(X11_ROOT)/lib'
PIPS_X11_ADDED_LIBS='-lX11'


#
# PIPS

PIPS_MAIN='main_pips.o'


#
# TPIPS

# is this so portable?

TPIPS_ADDED_LIBS='-lreadline -ltermcap'

TPIPS_MAIN='main_tpips.o'


#
# WPIPS

WPIPS_ADDED_CPPFLAGS='-I$(OPENWINHOME)/include -I$(X11_ROOT)/include -Iicons'
WPIPS_ADDED_LDFLAGS='-L$(OPENWINHOME)/lib -L$(X11_ROOT)/lib'
WPIPS_ADDED_LIBS='-lxview -lolgx -lX11'

WPIPS_MAIN='main_wpips.o'


#
# FPIPS

FPIPS_ADDED_CPPFLAGS='$(WPIPS_ADDED_CPPFLAGS)'
FPIPS_ADDED_LDFLAGS='$(WPIPS_ADDED_LDFLAGS)'
FPIPS_ADDED_LIBS='-lpips -ltpips -lwpips $(TPIPS_ADDED_LIBS) $(WPIPS_ADDED_LIBS)'

FPIPS_MAIN='main_fpips.o'

# end of it!
#
