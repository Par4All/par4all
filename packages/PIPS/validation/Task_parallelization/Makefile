# $Id: Makefile 5188 2011-12-31 09:49:52Z coelho $

PARALLEL_VALIDATION = 1
include ../validate.mk

# ignore empty-line changes on mpeg2enc
mpeg2enc.result/test: FLT = sed -e 's,$(here),$$VDIR,g;/^ *$$/d'
mpeg2enc.validate: FLT = sed -e 's,$(here),$$VDIR,g;/^ *$$/d'
