# this script can be used to create release of pips
# it patches each svn to fix some #!§% of the build system, so that we get independant tarballs
# and generate auser-friendly config.mk
#

# feel free to override this
SVNUSER=guelton

all:distclean linear.tar.gz newgen.tar.gz pips.tar.gz

dist:linear.tar.gz newgen.tar.gz pips.tar.gz 
	cp *-*.tar.gz /var/www/pips/
	chmod a+x /var/www/pips/*.tar.gz
	rm $^

%.tar.gz:%-svn
	svn up $<
	test ! -f $*.patch || ./$*.patch $<
	make unbuild -C $<
	VERSION="`grep -o '^VERSION=.*$$' $</libraries.make | cut -d '=' -f 2| tail -n 1`" ;\
	TARGET="$*-$$VERSION.tar.gz" &&\
	tar cvzf $$TARGET --exclude debian --exclude '*.svn*' $< &&\
	ln -s $$TARGET $@

SVNCRI	= https://svn.cri.ensmp.fr/svn

# hmmm... why not bundles/prod which is the same?
pips-svn:
	svn checkout --username=$(SVNUSER) $(SVNCRI)/pips/trunk $@
linear-svn:
	svn checkout --username=$(SVNUSER) $(SVNCRI)/linear/trunk $@
newgen-svn:
	svn checkout --username=$(SVNUSER) $(SVNCRI)/newgen/trunk $@

clean:
	rm -rf *.tar.gz

distclean:clean
	rm -rf pips-svn linear-svn newgen-svn
	
