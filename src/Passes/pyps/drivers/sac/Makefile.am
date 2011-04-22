nodist_pkgpython_PYTHON = sac.py
dist_noinst_DATA=gen_sac.py _sac.py

sac.py: $(srcdir)/_sac.py 
	cp $(srcdir)/_sac.py sac.py

SUBDIRS = impl

CLEANFILES=sac.py
