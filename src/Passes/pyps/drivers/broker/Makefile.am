dist_pkgpython_PYTHON = broker.py simpleExampleBroker.py simpleStubBroker.py

EXTRA_DIST=impl
dist-hook:
	rm -rf `find $(distdir)/complexity_cost_tables -name .svn`

install-data-hook:
	test -d $(DESTDIR)$(pypsruntimedir)/broker || mkdir $(DESTDIR)$(pypsruntimedir)/broker
	chmod -R u+rw $(DESTDIR)$(pypsruntimedir)/broker
	cp -r $(srcdir)/impl/* $(DESTDIR)$(pypsruntimedir)/broker/

uninstall-hook:
	chmod -R u+rw $(DESTDIR)$(pypsruntimedir)/broker
	rm -rf $(DESTDIR)$(pypsruntimedir)/broker
