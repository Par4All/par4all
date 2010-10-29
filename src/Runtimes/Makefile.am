EXTRA_DIST=step
install-data-local:
	@test -z $(datadir)/step || $(MKDIR_P) $(datadir)/step
	cp -r $(srcdir)/step/* $(datadir)/step

uninstall-local:
	chmod -R u+w $(datadir)/step
	rm -rf $(datadir)/step

