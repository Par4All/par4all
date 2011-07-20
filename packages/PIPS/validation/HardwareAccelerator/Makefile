# $Id$

# run validation in parallel
PARALLEL_VALIDATION	= 1

# explicit list of recursion directories
override D.sub	= $(wildcard *.sub) oop anr999

include ../validate.mk

# add-on to check freia-spoc optimality
include counts.mk
