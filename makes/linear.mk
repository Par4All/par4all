# $Id$

phase3: .build_inc_second_pass

.build_inc_second_pass:
	$(MAKE) build-header-file
	$(RM) .build_inc ; $(MAKE) .build_inc
	touch $@

clean: linear-phase3-clean
linear-phase3-clean:
	$(RM) .build_inc_second_pass
