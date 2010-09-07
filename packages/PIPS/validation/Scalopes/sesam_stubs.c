void sesam_reserve_data(int size){
}

void sesam_data_assignation(int id, int size, int mode){
}


int* sesam_map_data(int id){
  return &id;
}

int sesam_get_page_size(){
  return 1;
}

void sesam_wait_dispo(int id, int page, int mode){

}
void sesam_send_dispo(int id, int page, int mode){

}

void sesam_unmap_data(int* id){

}

void sesam_chown_data(int id_data, int id_task){
}
