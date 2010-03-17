INCLUDES=\
	`cat ../pips_includes` \
	-I../../Documentation/pipsmake \
	-I../../Documentation/newgen \
	-I$(top_srcdir)/src/Documentation/constants \
	$(NEWGENLIBS_CFLAGS) $(LINEARLIBS_CFLAGS)
