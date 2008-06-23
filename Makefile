# $Id$

FIND	= find . -name '.svn' -type d -prune -o

clean:
	$(FIND) -name '*~' -type f -print0 \
	     -o -name 'core' -type f -print0 \
	     -o -name 'a.out' -type f -print0 \
	     -o -name '*.o' -type f -print0 | xargs -0 $(RM)
	$(FIND) -name '*.database' -type d -print0 \
	     -o -name 'validation_results.*' -type d -print0 | \
		xargs -0 $(RM) -r
	$(RM) */*.result/out properties.rc
	$(RM) -r RESULTS

# subdirectories to consider
TARGET	= $(shell grep '^[a-zA-Z]' defaults)
VOPT	= -v

.check_validate:
	@if [ -d RESULTS ] ; then \
	  echo -e \
	    "\ncannot run validation with prior RESULTS still available." \
	    "\nbefore trying to run the validation, do a cleanup with:\n" \
	    "\n\t\tshell> make clean\n" ; \
	  exit 1; \
	fi

validate: .check_validate
	PIPS_MORE=cat pips_validate $(VOPT) -V $(PWD) -O RESULTS $(TARGET)

accept:
	manual_accept $(TARGET)
