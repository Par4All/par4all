# $Id$

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
ifdef PIPS_ROOT
ROOT    = $(PIPS_ROOT)
else
ROOT    = ../../..
endif

#PROJECT = pips
include $(ROOT)/makes/main.mk
