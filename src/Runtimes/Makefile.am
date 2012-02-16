EXTRA_DIST=step

CLEANFILES=STEP.h
install-data-local:STEP.h
	@test -z $(DESTDIR)$(datadir)/step || $(MKDIR_P) $(DESTDIR)$(datadir)/step
	cd $(srcdir) ; tar -h --exclude='.svn' -c -f - step/* | (cd $(DESTDIR)$(datadir) ; tar xfp - ; chmod u+rw -R step )
	cp STEP.h $(DESTDIR)$(datadir)/step/c

STEP.h: $(srcdir)/step/c/step_common.h
	echo "!\n!   Genereted file from $^\n!" > $@ && \
	sed -n "/^#define[\t ]\+\([^ \t]\+\)[ \t]\+\([0-9]\+\)/p" $^ | \
	grep -v "^#define[\t ]*STEP_BARRIER" | \
	sed "s,^#define[\t ]\+\([^ \t]\+\)[ \t]\+\([0-9]\+\),\n       INTEGER \1\n       PARAMETER (\1 = \2)," >>$@

uninstall-local:
	chmod -R u+w $(DESTDIR)$(datadir)/step
	rm -rf $(DESTDIR)$(datadir)/step

