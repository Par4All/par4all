EXTRA_DIST=step
install-data-local:
	@test -z $(datadir)/step || $(MKDIR_P) $(datadir)/step
	tar --exclude='.svn' -c -f - $(srcdir)/step/* | (cd $(datadir)/step ; tar xfp -)

uninstall-local:
	chmod -R u+w $(datadir)/step
	rm -rf $(datadir)/step

