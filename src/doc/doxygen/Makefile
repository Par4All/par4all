# Push the make to 2 subdirectories, one for the NewGen program itself,
# and one for the NewGen libC that is interesting for programs using
# NewGen infrastructure.

TARGETS = newC libgenC
NEWGEN_SOURCE_DIR = ../../../genC
export NEWGEN_SOURCE_DIR

la_cible_par_defaut_si_aucune_n_est_precisee_sur_la_ligne_de_commande: doxygen

# Forward any command to the specified directories:
.DEFAULT:
	for d in $(TARGETS) ; do $(MAKE) -C $$d $(FWD_MKFLAGS) $@; done

#include ../../../makes/doxygen.mk
