TOP := $(dir $(lastword $(MAKEFILE_LIST)))


%:
	for target in $(TARGETS) ; do \
		make -C $$target $@ ; \
	done

