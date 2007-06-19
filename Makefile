# $Id$

FIND	= find . -name '.svn' -type d -prune -o

clean:
	$(FIND) -name '*~' -type f -print0 | xargs -0 $(RM)
	$(FIND) -name '*.database' -type d -print0 \
	     -o -name 'validation_results.*' -type d -print0 | \
		xargs -0 $(RM) -r
	$(RM) */*.result/out
	$(RM) -r RESULTS

# subdirectories to consider
TARGET	= $(shell grep '^[a-zA-Z]' defaults)

validate:
	PIPS_VALIDDIR=$(PWD) pips_validate -v -O RESULTS $(TARGET)

accept:
	manual_accept $(TARGET)
