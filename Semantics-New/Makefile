# $Id$

PARALLEL_VALIDATION = 1
include ../validate.mk

# ignore empty-line changes on mpeg2enc
mpeg2enc.result/test: FLT = sed -e 's,$(here),$$VDIR,g;/^$$/d'
