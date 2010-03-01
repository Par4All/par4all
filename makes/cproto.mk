# this rule is quite complicated
# it is in charge of wrapping cproto stuff and be compatible with automake
# the first requirement is that TARGET must be defined
# the second requirement is that there must be a version of $(TARGET).h present 
#   if not just copy the -local version
# finally we call cproto to generate the full new header
# if a change occured, we validate it !
# erros from cproto are just dropped (-O switch)
#
$(TARGET).h:$(srcdir)/$(TARGET)-local.h $(SOURCES)
	if ! test -f $(TARGET).h ; then \
		cp $(srcdir)/$(TARGET)-local.h $(TARGET).h ;\
	fi
	SOURCES=`for s in $(TARGET)-local.h $(SOURCES) ; do ( test -f $$s && echo $$s ) || echo $(srcdir)/$$s ; done`; \
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
	} | sed -e '/ yy/ d' > $(TARGET).h-tmp ; \
	if cmp -s $(TARGET).h $(TARGET).h-tmp ; then \
		echo "file is unchanged, updating timestamp only" ; \
		rm $(TARGET).h-tmp ;\
		for s in $$SOURCES ; do test $$s -ot $(TARGET).h || touch -r $$s $(TARGET).h ; done ;\
	 else \
	 	echo "udpating file"; \
		rm $(TARGET).h ; \
	    mv $(TARGET).h-tmp $(TARGET).h ;\
	 fi

$(TARGET)-local.h:
	touch $@

clean-local:
	rm -f $(TARGET).h
EXTRA_DIST=$(TARGET)-local.h
