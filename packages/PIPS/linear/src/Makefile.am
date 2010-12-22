SUBDIRS= \
	gnulib \
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
	polyedre \
	union

lib_LTLIBRARIES=liblinearlibs.la
liblinearlibs_la_SOURCES=
liblinearlibs_la_LDFLAGS=$(POLYLIB64_LIBS) $(LDFLAGS_NO_UNDEFINED) -lm
liblinearlibs_la_LIBADD=\
	gnulib/libgnu.la\
	arithmetique/libarithmetique.la\
	contrainte/libcontrainte.la\
  	matrice/libmatrice.la\
  	matrix/libmatrix.la\
  	polyedre/libpolyedre.la\
  	polynome/libpolynome.la\
  	ray_dte/libray_dte.la\
  	sc/libsc.la\
	sg/libsg.la\
  	sommet/libsommet.la\
  	sparse_sc/libsparse_sc.la\
  	union/libunion.la\
	vecteur/libvecteur.la

