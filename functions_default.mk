library:	$(STATIC_LIBRARY)

shared_lib:	$(SHARED_LIBRARY)

tags:
	etags $(SRCS) $(HDRS) 

$(STATIC_LIBRARY): $(OBJS)  
	@echo "Loading $(STATIC_LIBRARY) ..." 
	if [ -f $(ROOT)/Libraries/$(SHARED_LIBRARY) ] ; then rm -f $(ROOT)/Libraries/$(SHARED_LIBRARY) ; fi
	ar crv $(STATIC_LIBRARY) $(OBJS) 
	ranlib $(STATIC_LIBRARY)
	cd $(ROOT)/Libraries; \
	rm $(STATIC_LIBRARY); \
	$(LN_S) $(ROOT)/$(RPATH)/$(STATIC_LIBRARY) .
	cd $(ROOT)/$(RPATH);
	@echo "done"

$(SHARED_LIBRARY): $(OBJS) 
	@echo "Loading Shared $(SHARED_LIBRARY) ..."
	if [ -f $(ROOT)/Libraries/$(STATIC_LIBRARY) ] ; then rm -f $(ROOT)/Libraries/$(STATIC_LIBRARY) ; fi
	$(LD) $(CXXSHAREDLD) $(CXXSHAREDLDFLAGS) -o $(SHARED_LIBRARY) $(ROOT)/Kernel/Sample.o $(OBJS) 
	cd $(ROOT)/Libraries; \
	rm $(SHARED_LIBRARY); \
	$(LN_S) $(ROOT)/$(RPATH)/$(SHARED_LIBRARY) .
	cd $(ROOT)/$(RPATH);
	@echo "done"

closure: $(OBJS)
	echo "Making Closer for $(LIBRARY) ..."
	make closure.o
	- $(CXX) -frepo closure.o $(OBJS) $(INCL) $(LIBS) -o closure -lpamela $(LIBRARY_CLOSURE)
	make library

copy:
	@cp $(STATIC_LIBRARY) $(ROOT)/Libraries

patch:
	@rcsdiff -u $(SRCS) $(HDRS) Makefile >> patch

clean:
	rm -rf $(OBJS) 
	rm -rf $(DEPD) 
	rm -rf $(OBJS:.o=.rpo) 

realclean:
	rm -rf $(OBJS) 
	rm -rf $(DEPD) 
	rm -rf $(SHARED_LIBRARY) 
	rm -rf $(STATIC_LIBRARY) 
	rm -rf $(ROOT)/Libraries/$(SHARED_LIBRARY) 
	rm -rf $(ROOT)/Libraries/$(STATIC_LIBRARY) 
	rm -rf *.rpo *.o *.*~ core closure *.tab.* lex.*

rcs:	 
	ci $(SRCS) $(HDRS) Makefile

update:
	co -l $(HDRS) $(SRCS) Makefile

