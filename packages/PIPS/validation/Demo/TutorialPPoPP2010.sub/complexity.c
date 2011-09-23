Ppolynome instruction_time = polynome_dup(complexity_polynome(comp));
polynome_scalar_mult(&instruction_time,1.f/p->frequency);
...
polynome_negate(&transfer_time);
polynome_add(&instruction_time,transfer_time);
int max_degree = polynome_max_degree(instruction_time);
