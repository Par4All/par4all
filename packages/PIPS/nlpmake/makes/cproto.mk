# this rule is quite complicated
# it is in charge of wrapping cproto stuff and be compatible with automake
# the first requirement is that TARGET must be defined
# the second requirement is that there must be a version of $(TARGET).h present 
#   if not just copy the -local version
# finally we call cproto to generate the full new header
# if a change occured, we validate it !
# erros from cproto are just dropped (-O switch)
#
# note that the time stamp is here to prevent too many runs of cproto ...

CPROTO_STAMP_FILE=.cproto.stamp

cproto_bootstrap:$(CPROTO_STAMP_FILE)_init

# this one ensures there is a minimal header
$(CPROTO_STAMP_FILE)_init:$(srcdir)/$(TARGET)-local.h $(srcdir)/Makefile.am
	test -f $(TARGET).h || ( cp $(srcdir)/$(TARGET)-local.h $(TARGET).h && chmod u+w $(TARGET).h && touch -r  $(srcdir)/$(TARGET)-local.h $(TARGET).h )
	touch $(CPROTO_STAMP_FILE)_init

# this one generates the stamp
# we cannot depend on BUILT_SOURCES because it would create a circular dep
# instead we force make of thos targets
$(CPROTO_STAMP_FILE):$(CPROTO_STAMP_FILE)_init  $(SOURCES) $(srcdir)/Makefile.am
	cproto_extra_deps="`echo $(BUILT_SOURCES) | sed -e 's/$(TARGET).h//'`" ;\
	if test "$$cproto_extra_deps" ; then $(MAKE) $$cproto_extra_deps ; fi
	cproto_sources=`for s in $(SOURCES) ; do ( test -f $$s && echo $$s ) || echo $(srcdir)/$$s ; done`; \
	{ \
		cproto_guard="`echo $(TARGET)_header_included | tr - _`";\
      	echo "/* Warning! Do not modify this file that is automatically generated!"; \
      	echo " * Modify $(srcdir)/$(TARGET)-local.h instead, to add your own modifications."; \
      	echo " *"; \
      	echo " * header file built by $(CPROTO)"; \
      	echo " */"; \
      	echo "#ifndef $${cproto_guard}";\
      	echo "#define $${cproto_guard}";\
      	cat $(srcdir)/$(TARGET)-local.h ;\
		for cproto_source in $$cproto_sources ; do \
			case $$cproto_source in \
				*.c)\
					$(CPROTO) -evcf2 -O /dev/null -E "$(CPP) $(INCLUDES) $(DEFAULT_INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) -DCPROTO_IS_PROTOTYPING" $$cproto_source ;;\
				*)\
					cproto_proxy=`basename $${cproto_source}`.c ;\
					sed -n -e '/^%{/,/%}/ p' -e '1,/^%%/ d' -e '/^%%/,$$ p' $$cproto_source | sed -e '/^%/ d'  > $$cproto_proxy ; \
					$(CPROTO) -evcf2 -O /dev/null -E "$(CPP) $(INCLUDES) $(DEFAULT_INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) -DCPROTO_IS_PROTOTYPING" $$cproto_proxy |\
						sed -e '/ yy/ d';\
					rm -f $$cproto_proxy ;;\
			esac ;\
		done ; \
      	echo "#endif /* $${guard} */"; \
	} > $(CPROTO_STAMP_FILE)

# here is the trick: if our stamp file is similar to the header, we do not touch the header
# and so do not trigger rebuild of all files including the header
$(TARGET).h:$(CPROTO_STAMP_FILE)
	cmp -s $(TARGET).h $(CPROTO_STAMP_FILE) || cp $(CPROTO_STAMP_FILE) $(TARGET).h

clean-local:
	rm -f $(TARGET).h $(CPROTO_STAMP_FILE) $(CPROTO_STAMP_FILE)_init

EXTRA_DIST=$(TARGET)-local.h
