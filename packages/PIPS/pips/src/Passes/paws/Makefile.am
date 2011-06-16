
dist_bin_SCRIPTS = paws

pawsdir = $(pypsruntimedir)/paws



EXTRA_DIST=pawsapp
dist-hook:
	rm -rf `find $(distdir)/pawsapp -name .svn`
install-data-hook:
	test -d $(DESTDIR)$(pawsdir) || mkdir $(DESTDIR)$(pawsdir)
	test -d $(DESTDIR)$(pawsdir)/pawsapp || mkdir $(DESTDIR)$(pawsdir)/pawsapp
	chmod -R u+rw $(DESTDIR)$(pawsdir)/pawsapp
	cp -r $(srcdir)/pawsapp $(DESTDIR)$(pawsdir)/

uninstall-hook:
	chmod -R u+rw $(DESTDIR)$(pawsdir)/pawsapp
	rm -rf $(DESTDIR)$(pawsdir)/pawsapp

