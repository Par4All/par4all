
# don't call this makefile directly, the upper directory makefile
# does that for you.

# OBJ_DIR and EXEC_EXTRA_SUFFIX must be set correctly

# These may take a long time to compute...
LONG_EHRHART_TESTS = g1 isnm2 c4 alex1

# These do not pass in 32 bits mode without GMP: c3 e10 e11 e15 isnm
# uncomment the following line if you are not running with GMP.
#NOGMP = c3 e10 e11 e15 isnm

EHRHART_TESTS = $(filter-out $(LONG_EHRHART_TESTS) $(NOGMP),\
	$(patsubst ehrhart/%.in,%,$(wildcard ehrhart/*.in)))

PP_TESTS = c4 c5
GENERAL_TESTS = convex1 convex2 simpl1 simpl2 \
		simpl3 simpl4 simpl5 simpl6 simpl7 gauss1 test1 herve

ZPOLYTESTS =  LatHNF1 LatHNF2 LatHNF3 LatInter1 LatInter2 \
	LatInc1 LatInc2 ZEmpty1 ZEmpty2 ZInter1 ZInter2  ZInter3  ZUnion1 \
	ZUnion2 ZDiff1 ZDiff2 ZImage1 ZPre1 ZPre2  ZPre3 ZImPre1 


all_tests : general_tests pp$(EXEC_EXTRA_SUFFIX)_tests \
	    ehrhart$(EXEC_EXTRA_SUFFIX)_tests zpoly_tests
	-@\rm -f xyz
	@echo "----------------------------------------------------"
	@echo "all tests successful"
	@echo "----------------------------------------------------"

#=====================================================================
# these tests are much slower...
long_tests::
	@echo "----------------------------------------------------"
	@echo "Begin long tests"
	@echo "----------------------------------------------------"
long_tests:: $(LONG_EHRHART_TESTS:%=ehrhart/%) \
		verif_ehrhart$(EXEC_EXTRA_SUFFIX)_tests
	-@\rm -f xyz
	@echo "----------------------------------------------------"
	@echo "all long tests successful"
	@echo "----------------------------------------------------"

#=====================================================================
general_tests ::
	@echo "----------------------------------------------------"
	@echo "Begin general tests"
	@echo "----------------------------------------------------"
general_tests :: libtest $(GENERAL_TESTS)
	@echo "General tests successful"

libtest:
	$(OBJ_DIR)/testlib$(EXEC_EXTRA_SUFFIX) <general/test.in >xyz
	diff -w xyz general/test.out

$(GENERAL_TESTS):
	$(OBJ_DIR)/polytest$(EXEC_EXTRA_SUFFIX) <general/$@.in >xyz
	diff -w xyz general/$@.out

#=====================================================================
ehrhart$(EXEC_EXTRA_SUFFIX)_tests ::
	@echo "----------------------------------------------------"
	@echo "Begin 'ehrhart$(EXEC_EXTRA_SUFFIX)' tests"
	@echo "----------------------------------------------------"
ehrhart$(EXEC_EXTRA_SUFFIX)_tests :: $(EHRHART_TESTS:%=ehrhart/%)
	@echo "'ehrhart$(EXEC_EXTRA_SUFFIX)' tests successful"

ehrhart/%:
	$(OBJ_DIR)/ehrhart$(EXEC_EXTRA_SUFFIX) <$@.in >xyz
	diff -w xyz $@.out

#=====================================================================
verif_ehrhart$(EXEC_EXTRA_SUFFIX)_tests ::
	@echo "----------------------------------------------------"
	@echo "Begin 'verif_ehrhart$(EXEC_EXTRA_SUFFIX)' tests"
	@echo "----------------------------------------------------"
verif_ehrhart$(EXEC_EXTRA_SUFFIX)_tests :: $(EHRHART_TESTS:%=verif_ehrhart/%) \
		$(LONG_EHRHART_TESTS:%=verif_ehrhart/%)
	@echo "'verif_ehrhart$(EXEC_EXTRA_SUFFIX)' tests successful"

verif_ehrhart/%:
	$(OBJ_DIR)/verif_ehrhart$(EXEC_EXTRA_SUFFIX) <$(@:verif_ehrhart/%=ehrhart/%).in

#=====================================================================
pp$(EXEC_EXTRA_SUFFIX)_tests ::
	@echo "----------------------------------------------------"
	@echo "Begin 'pp$(EXEC_EXTRA_SUFFIX)' tests"
	@echo "----------------------------------------------------"
pp$(EXEC_EXTRA_SUFFIX)_tests :: $(PP_TESTS:%=pp/%)
	@echo "'pp$(EXEC_EXTRA_SUFFIX)' tests successful"

pp/%:
	$(OBJ_DIR)/pp$(EXEC_EXTRA_SUFFIX) <$@.in >xyz
	diff -w xyz $@.out

#=====================================================================
zpoly_tests ::
	@echo "----------------------------------------------------"
	@echo "Begin 'Zpolyhedron' tests"
	@echo "----------------------------------------------------"
zpoly_tests :: $(ZPOLYTESTS:%=Zpolytest/%)
	@echo "'Zpolyhedron' tests successful"

Zpolytest/%:
	$(OBJ_DIR)/Zpolytest$(EXEC_EXTRA_SUFFIX) <$@.in >xyz
	diff -w xyz $@.out

#=====================================================================

