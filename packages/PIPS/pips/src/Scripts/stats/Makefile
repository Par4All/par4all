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

SCRIPTS = 	print-dg-statistics \
		print-regions-op-statistics \
		print-regions-translation-statistics \
		parallelization_statistics \
		cumulate_statistics.pl

F.awk =		dg-statistics.awk \
		print-dependence.awk \
		region-binary-op.awk \
		region-proj-param-op.awk \
		region-proj-var-op.awk \
		region-umust-op.awk \
		region_trans_stat_common_dim.awk \
		region_trans_stat_phi_elim.awk \
		region_trans_stat_delta_elim.awk \
		region_trans_stat_pred_trans.awk \
		region_trans_stat_dim.awk \
		region_trans_stat_remaining_dims.awk \
		region_trans_stat_linearization.awk \
		region_trans_stat_size.awk \
		region_trans_stat_offset.awk \
		region_trans_stat_type.awk 

INSTALL_UTL = 	$(SCRIPTS) $(F.awk)

# common stuff
ROOT    = ../../..
#PROJECT = pips
include $(ROOT)/makes/main.mk
