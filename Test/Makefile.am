EXTRA_DIST = cases old others
SUBDIRS = Zpolytest ehrhart ehrhart_union general pp

long_tests:
	(cd ehrhart; $(MAKE) longtest)

dist-hook:
	rm -rf `find $(distdir) -name CVS`
