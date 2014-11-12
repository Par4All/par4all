# $Id$

#  arithmetique:
#  vecteur[arithmetique]: vectors
#  contrainte[vecteur]: constraints
#  sc[contrainte,vecteur]: systems (equalities and inequalities)
#  matrice: rational dense matrices
#  ray_dte:
#  sommet[ray_dte,vecteur]: for constraints, systems and functions (?)
#  plint[sommet,matrice,sc]: integer linear programming
#  sparse_sc[matrice,sc]: sparse systems
#  sg[sommet,ray_dte]: generating systems
#  polyedre[sg,sc]: polyhedrons

FWD_DIRS = \
	arithmetique \
	vecteur \
	contrainte \
	sc \
	matrice \
	matrix \
	ray_dte \
	sommet \
	sparse_sc \
	sg \
	polynome \
	union \
	polyedre \
	doxygen \
	linearlibs
#	plint
#	Tests

FWD_PARALLEL	= 1

ifeq ($(FWD_TARGET),phase0)
USE_DEPS = 1
else ifeq ($(FWD_TARGET),phase2)
USE_DEPS = 1
endif

# (re)build inter library header dependencies
deps.mk:
	{ \
	  echo 'ifdef USE_DEPS'; \
	  inc2deps.sh $(FWD_DIRS) | sed -e 's/:/:fwd-/;s/^/fwd-/'; \
	  echo 'endif # USE_DEPS'; \
	} > $@
