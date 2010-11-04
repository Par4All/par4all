EXTRA_DIST=step
install-data-local:
	@test -z $(datadir)/step || $(MKDIR_P) $(datadir)/step
	cd $(srcdir) ; tar --exclude='.svn' -c -f - step/* | (cd $(datadir) ; tar xfp -)

uninstall-local:
	chmod -R u+w $(datadir)/step
	rm -rf $(datadir)/step

