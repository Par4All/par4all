dist_pkgpython_PYTHON = pocc.py


install-data-hook:
	test -d $(DESTDIR)$(pypsruntimedir)/pocc || mkdir $(DESTDIR)$(pypsruntimedir)/pocc
	chmod -R u+rw $(DESTDIR)$(pypsruntimedir)/pocc

uninstall-hook:
	chmod -R u+rw $(DESTDIR)$(pypsruntimedir)/pocc
	rm -rf $(DESTDIR)$(pypsruntimedir)/pocc
