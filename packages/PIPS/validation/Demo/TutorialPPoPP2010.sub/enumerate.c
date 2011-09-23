FOREACH(REGION,reg,regions) {
    Ppolynome reg_footprint= region_enumerate(reg);
    // may be we should use the rectangular hull ?
    polynome_add(&transfer_time,reg_footprint);
    polynome_rm(&reg_footprint);
}

