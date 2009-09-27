# $Id$

optimality.count:
	grep 'freia_status .*_helper_' *.result/test | \
	cut -d: -f1 | \
	uniq -c | sort -n > $@

clean: counts-clean
counts-clean:
	$(RM) optimality.count
