#
# $Id$
#
# $Log: define_libraries.sh,v $
# Revision 1.11  1997/03/27 13:51:20  coelho
# *_DOCS added.
#
# Revision 1.10  1997/03/27 13:39:00  coelho
# fiew more comments.
#
#
# this file define newgen, linear and pips libraries.
# it is to be included by some shell-scripts and makefiles (after a sed).

#
# NEWGEN

NEWGEN_DOCS='doc'
NEWGEN_ORDERED_LIBS='doc scripts genC'

NEWGEN_LIBS='-lgenC'

#
# C3/LINEAR

LINEAR_DOCS=''
LINEAR_ORDERED_LIBS='arithmetique vecteur contrainte sc matrice matrix ray_dte sommet sg polynome polyedre plint sparse_sc union'

# removed because not used by pips: plint

LINEAR_LIBS='-lmatrice -lunion -lpolyedre -lsparse_sc -lsc -lcontrainte -lsg -lsommet -lray_dte -lpolynome -lmatrix -lvecteur -larithmetique'

#
# PIPS

# static ordering for bootstraping PIPS...

# many doc dirs are skipped (would be long?)
PIPS_DOCS='Documentation constants help manl newgen'

PIPS_ORDERED_LIBS='xpomp text-util properties misc ri-util conversion movements pipsdbm comp_sections transformer bootstrap control hyperplane flint syntax prettyprint static_controlize effects semantics complexity continuation reductions regions callgraph icfg paf-util pip ricedg array_dfg prgm_mapping scheduling reindexing chains rice transformations hpfc atomizer wp65 pipsmake top-level pips tpips wpips'

# all libraires for pips
PIPS_LIBS='-ltop-level -lpipsmake -lwp65 -lhpfc -ltransformations -lmovements -lbootstrap -lcallgraph -licfg -lchains -lcomplexity -lconversion -lprettyprint -latomizer -lprgm_mapping -lscheduling -lreindexing -larray_dfg -lpaf-util -lstatic_controlize -lsyntax -lregions -lcomp_sections -lcontrol -lsemantics -lcontinuation -lrice -lricedg -leffects -lpipsdbm -ltransformer -lpip -lri-util -lproperties -ltext-util -lmisc -lproperties -lreductions -lflint $(NEWGEN_LIBS) $(LINEAR_LIBS) -lm -lrx'

#
# X11

PIPS_X11_ADDED_CPPFLAGS='-I$(X11_ROOT)/include'
PIPS_X11_ADDED_LDFLAGS='-L$(X11_ROOT)/lib'
PIPS_X11_ADDED_LIBS='-lX11'

#
# WPIPS

WPIPS_ADDED_CPPFLAGS='-I$(OPENWINHOME)/include -Iicons'
WPIPS_ADDED_LIBS='-lxview -lolgx -lX11'
WPIPS_ADDED_LDFLAGS='-L$(OPENWINHOME)/lib'

#
# TPIPS

# is this so portable?

TPIPS_ADDED_LIBS='-lreadline -ltermcap'

# end of it!
#
