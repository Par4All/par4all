dist_pkgpython_PYTHON = sac.py
dist_noinst_DATA=_sac.py
dist_noinst_DATA=gen_sac.py

sac_gen_cmd= $(srcdir)/gen_sac.py $(srcdir)/_sac.py
sac.py:$(sac_gen_cmd) _sac.py
	$(PYTHON) $(sac_gen_cmd) >sac.py

