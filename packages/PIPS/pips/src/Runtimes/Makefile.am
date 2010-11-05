EXTRA_DIST=step
install-data-local:
	@test -z $(DESTDIR)$(datadir)/step || $(MKDIR_P) $(DESTDIR)$(datadir)/step
	cd $(srcdir) ; tar --exclude='.svn' -c -f - step/* | (cd $(DESTDIR)$(datadir) ; tar xfp -)

uninstall-local:
	chmod -R u+w $(DESTDIR)$(datadir)/step
	rm -rf $(DESTDIR)$(datadir)/step

