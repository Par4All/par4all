# this rule is quite complicated
# it is in charge of wrapping cproto stuff and be compatible with automake
# the first requirement is that TARGET must be defined
# the second requirement is that there must be a version of $(TARGET).h present 
#   if not just copy the -local version
# finally we call cproto to generate the full new header
# if a change occured, we validate it !
# erros from cproto are just dropped (-O switch)
#
# not that the time stamp is here to prevent too many runs of cproto ...

CPROTO_STAMP_FILE=.cproto.stamp
$(CPROTO_STAMP_FILE):$(srcdir)/$(TARGET)-local.h $(SOURCES)
	test -f $(TARGET).h || ( cp $(srcdir)/$(TARGET)-local.h $(TARGET).h && chmod u+w $(TARGET).h )
	SOURCES=`for s in $(SOURCES) ; do ( test -f $$s && echo $$s ) || echo $(srcdir)/$$s ; done`; \
	{ \
		guard=`echo $(TARGET)_header_included | tr - _`;\
      	echo "/* Warning! Do not modify this file that is automatically generated! */"; \
      	echo "/* Modify src/Libs/$(TARGET)/$(TARGET)-local.h instead, to add your own modifications. */"; \
      	echo ""; \
      	echo "/* header file built by $(CPROTO) */"; \
      	echo ""; \
      	echo "#ifndef $${guard}";\
      	echo "#define $${guard}";\
      	cat `( test -f $(TARGET)-local.h && echo $(TARGET)-local.h ) || echo $(srcdir)/$(TARGET)-local.h ` ;\
		for s in $$SOURCES ; do \
			$(CPROTO) -evcf2 -O /dev/null -E "$(CPP) $(INCLUDES) $(DEFAULT_INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) -DCPROTO_IS_PROTOTYPING" $$s ;\
		done ; \
      	echo "#endif /* $${guard} */"; \
	} | sed -e '/ yy/ d' > $(CPROTO_STAMP_FILE)

# here is the trick: if our stamp file is similar to the header, we do not touch the header
# and so do not trigger rebuild of all files including the header
$(TARGET).h:$(CPROTO_STAMP_FILE)
	cmp -s $(TARGET).h $(CPROTO_STAMP_FILE) || cp $(CPROTO_STAMP_FILE) $(TARGET).h

$(TARGET)-local.h:
	touch $@

clean-local:
	rm -f $(TARGET).h $(CPROTO_STAMP_FILE)

EXTRA_DIST=$(TARGET)-local.h
