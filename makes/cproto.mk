$(TARGET).h:$(TARGET)-local.h $(SOURCES)
	cat $< > $(TARGET).h
	{ \
		SOURCES=`for s in $^ ; do case $$s in *.[ch]) echo $$s ;; esac ; done`; \
		guard=`echo $(TARGET)_header_included | tr - _`;\
      	echo "/* Warning! Do not modify this file that is automatically generated! */"; \
      	echo "/* Modify src/Libs/$(TARGET)/$(TARGET)-local.h instead, to add your own modifications. */"; \
      	echo ""; \
      	echo "/* header file built by $(CPROTO) */"; \
      	echo ""; \
      	echo "#ifndef $${guard}";\
      	echo "#define $${guard}";\
      	cat $< ;\
		$(CPROTO) -evcf2 -E "$(CPP) $(INCLUDES) $(DEFAULT_INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) -DCPROTO_IS_PROTOTYPING" $$SOURCES ;\
      	echo "#endif /* $${guard} */"; \
	} > $(TARGET).h-tmp
	rm $(TARGET).h
	mv $(TARGET).h-tmp $(TARGET).h

$(TARGET)-local.h:
	touch $@

clean-local:
	rm -f $(TARGET).h
EXTRA_DIST=$(TARGET)-local.h
