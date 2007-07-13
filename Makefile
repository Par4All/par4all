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

validate:
	pips_validate $(VOPT) -V $(PWD) -O RESULTS $(TARGET)

accept:
	manual_accept $(TARGET)
