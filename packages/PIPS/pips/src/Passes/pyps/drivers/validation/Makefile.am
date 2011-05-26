dist_pkgpython_PYTHON = validation.py


install-data-hook:
	test -d $(DESTDIR)$(pypsruntimedir)/validation || mkdir $(DESTDIR)$(pypsruntimedir)/validation
	chmod -R u+rw $(DESTDIR)$(pypsruntimedir)/validation

uninstall-hook:
	chmod -R u+rw $(DESTDIR)$(pypsruntimedir)/validation
	rm -rf $(DESTDIR)$(pypsruntimedir)/validation
