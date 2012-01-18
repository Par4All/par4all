TOP := $(dir $(lastword $(MAKEFILE_LIST)))


%:
	for target in $(LOCAL_TARGETS) ; do \
		make -C $$target $@ ; \
	done

