nodist_pkgpython_PYTHON = sac.py
dist_noinst_DATA=gen_sac.py _sac.py \
				 impl/3dnow.h  impl/avx.h  impl/neon.h  impl/sse.h \
				 impl/patterns.c  impl/SIMD.c  impl/SIMD_types.h  

sac_gen_cmd= $(srcdir)/gen_sac.py $(srcdir)/_sac.py
sac.py:$(sac_gen_cmd) $(srcdir)/_sac.py $(srcdir)/impl/SIMD.c $(srcdir)/impl/SIMD_types.h $(srcdir)/impl/patterns.c $(srcdir)/impl/sse.h $(srcdir)/impl/neon.h $(srcdir)/impl/avx.h $(srcdir)/impl/3dnow.h
	$(PYTHON) $(sac_gen_cmd) > sac.py

CLEANFILES=sac.py
