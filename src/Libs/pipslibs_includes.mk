INCLUDES=$(NEWGENLIBS_CFLAGS) $(LINEARLIBS_CFLAGS)\
	-I../../Documentation/pipsmake \
	-I../../Documentation/newgen \
	-I$(top_srcdir)/src/Documentation/constants \
	`cat ../pips_includes`
