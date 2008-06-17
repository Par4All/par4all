#
# Validation/Hpfc/Makefile
#
# Fabien COELHO
# 
# $RCSfile: Makefile,v $ version $Revision: 1.16 $,
# ($Date: 1997/05/03 19:40:49 $,) 

#
# gmake should be invoked...
FFILES = $(wildcard *.f)
UFFILES = $(wildcard *.F)

# sun make
# FFILES = *.f

#
# standard

TESTFILES = $(FFILES:%.f=%.test) $(UFFILES:.F=.test)
RESULTDIRS = $(FFILES:%.f=%.result) $(UFFILES:.F=.result)

all: $(RESULTDIRS)

.SUFFIXES: .test .f .result .F

%.result: %.f
	test -d $@ || mkdir $@ 
	test -f $@/test || touch $@/test
	touch $@

%.result: %.F
	test -d $@ || mkdir $@ 
	test -f $@/test || touch $@/test
	touch $@

%.test: %.f
	sed 's,tested_file,$*,g' < default_test > $@
	chmod a+x $@

%.test: %.F
	sed 's,tested_file,$*,g' < default_test > $@
	chmod a+x $@

accept:
	for d in $(RESULTDIRS) ; \
	do \
	  [ -f $$d/test.out ] && mv $$d/test.out $$d/test ; \
	done

count:
	# also the number of lines
	@wc -l $(FFILES) $(UFFILES)
	# computes the number of files in the validation
	@echo $(FFILES) $(UFFILES) | wc -w

#
# that's all for $RCSfile: Makefile,v $
#
