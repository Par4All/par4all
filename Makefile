# $Id$

FIND	= find . -name '.svn' -type d -prune -o

clean:
	$(FIND) -name '*~' -print0 | xargs -0 $(RM)
	$(RM) */*.result/out
	$(RM) -r validation_results.*

# subdirectories to consider
TARGET	=

validate:
	PIPS_VALIDDIR=$(PWD) pips_validate -v $(TARGET)

accept:
	manual_accept $(TARGET)
