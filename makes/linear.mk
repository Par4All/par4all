# $Id$

phase3: .build_inc_second_pass

.build_inc_second_pass: 
	$(MAKE) build-header-file .build_inc
	touch $@

clean: phase3-clean

phase3-clean:
	$(RM) .build_inc_second_pass
