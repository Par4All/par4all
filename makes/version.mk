# $Id$

VERSION	= $(MAKE.d)/version.sh

.revisions_h:
	$(RM) revisions.h
	{ \
	  echo '#define NEWGEN_REV "$(shell $(VERSION) $(NEWGEN_ROOT))"'; \
	  echo '#define LINEAR_REV "$(shell $(VERSION) $(LINEAR_ROOT))"'; \
	  echo '#define PIPS_REV "$(shell $(VERSION) $(PIPS_ROOT))"'; \
	  echo '#define NLPMAKE_REV "$(shell $(VERSION) $(PIPS_ROOT)/makes)"'; \
	} > revisions.h


$(ARCH)/revisions.o: CPPFLAGS += -DUTC_DATE='$(UTC_DATE)'

revisions.h: .revisions_h

clean: version-clean
version-clean:
	$(RM) revisions.h
