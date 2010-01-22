$(TARGET).h:$(TARGET)-local.h $(SOURCES)
	cat $< > $(TARGET).h
	{ \
      	echo "/* Warning! Do not modify this file that is automatically generated! */"; \
      	echo "/* Modify src/Libs/$(TARGET)/$(TARGET)-local.h instead, to add your own modifications. */"; \
      	echo ""; \
      	echo "/* header file built by $(CPROTO) */"; \
      	echo ""; \
      	echo "#ifndef $(TARGET)_header_included";\
      	echo "#define $(TARGET)_header_included";\
		echo "#ifdef HAVE_CONFIG_H";\
		echo "#include \"$(CONFIG_HEADER)\"";\
		echo "#endif";\
      	cat $< ;\
		$(CPROTO) -evcf2 -E "$(CPP) $(INCLUDES) $(DEFAULT_INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS) -DCPROTO_IS_PROTOTYPING" $^ ;\
      	echo "#endif /* $(name)_header_included */"; \
	} > $(TARGET).h-tmp
	rm $(TARGET).h
	mv $(TARGET).h-tmp $(TARGET).h

CLEANFILES=$(TARGET).h
