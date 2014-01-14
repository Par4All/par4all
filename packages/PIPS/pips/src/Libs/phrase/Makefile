# $Id$
#
# Copyright 1989-2014 MINES ParisTech
#
# This file is part of PIPS.
#
# PIPS is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
#

TARGET = phrase

LIB_CFILES =    comEngine_optimization_function.c \
                comEngine_HRE_distribute.c \
                comEngine_generate_code.c \
                phrase_remove_dependences.c \
                comEngine_generate_HRECode.c \
                comEngine_generate_procCode.c \
                comEngine_feasability.c \
                comEngine_distribute.c \
                phrase_distributor.c \
		phrase_distributor_control_code.c \
		phrase_distributor_communication.c \
		distribution_context.c \
		fsm_generation.c \
		print_code_smalltalk.c \
		fsm_merge_states.c \
		fsm_split_state.c \
		full_fsm_generation.c \
		loop_spaghettify.c \
		whileloop_spaghettify.c \
		forloop_spaghettify.c \
		test_spaghettify.c \
		spaghettify.c \
		full_spaghettify.c \
		fsm_tools.c \
		phrase_tools.c

INC_TARGET =    $(TARGET).h

LIB_TARGET =    lib$(TARGET).a

# common stuff
ROOT    = ../../..
PROJECT = pips
include $(ROOT)/makes/main.mk
