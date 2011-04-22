sacruntimedir = $(pypsruntimedir)/sac
generated_files = patterns.h 
dist_sacruntime_DATA = $(generated) 3dnow.h  SIMD.c  SIMD_types.h  avx.h  neon.h  patterns.c  sse.h  patterns.h  
.c.h:
	$(CPROTO) $< > $@

CLEANFILES = $(generated)
