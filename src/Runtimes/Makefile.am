EXTRA_DIST=step
install-data-local:
	@test -z $(datadir)/step || $(MKDIR_P) $(datadir)/step
	cp -r $(srcdir)/step/* $(datadir)/step

uninstall-local:
	rm -rf $(datadir)/step

