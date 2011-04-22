nodist_pkgpython_PYTHON = sac.py
dist_noinst_DATA=gen_sac.py _sac.py \
				 impl/3dnow.h  impl/avx.h  impl/neon.h  impl/sse.h \
				 impl/patterns.c  impl/SIMD.c  impl/SIMD_types.h  

sac.py: $(srcdir)/_sac.py 
	cp $(srcdir)/_sac.py sac.py

SUBDIRS = impl

CLEANFILES=sac.py
