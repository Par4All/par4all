module p4a_runtime_interface
  interface
     subroutine P4A_copy_from_accel(element_size, host_address, accel_address)&
          bind (C, name = "P4A_copy_from_accel")
       use iso_c_binding
       integer (c_size_t), value :: element_size
       type (c_ptr), value :: host_address, accel_address
     end subroutine P4A_copy_from_accel

     subroutine P4A_copy_to_accel(element_size, host_address, accel_address)&
          bind (C, name = "P4A_copy_to_accel")
       use iso_c_binding
       integer (c_size_t), value :: element_size
       type (c_ptr), value :: host_address, accel_address
     end subroutine P4A_copy_to_accel

     subroutine P4A_copy_from_accel_1d(element_size, d1_size, d1_block_size,&
          d1_offset, host_address, accel_address)&
          bind (C, name = "P4A_copy_from_accel_1d")
       use iso_c_binding
       integer (c_size_t), value :: element_size, d1_size
       integer (c_size_t), value :: d1_block_size, d1_offset
       type (c_ptr), value :: host_address, accel_address
     end subroutine P4A_copy_from_accel_1d

     subroutine P4A_copy_to_accel_1d(element_size, d1_size, d1_block_size,&
          d1_offset, host_address, accel_address)&
          bind (C, name = "P4A_copy_to_accel_1d")
       use iso_c_binding
       integer (c_size_t), value :: element_size, d1_size
       integer (c_size_t), value :: d1_block_size, d1_offset
       type (c_ptr), value :: host_address, accel_address
     end subroutine P4A_copy_to_accel_1d

     subroutine P4A_copy_to_accel_2d(element_size, d1_size, d2_size,&
          d1_block_size, d2_block_size, d1_offset, d2_offset,&
          host_address, accel_address)&
          bind (C, name = "P4A_copy_to_accel_2d")
       use iso_c_binding
       integer (c_size_t), value :: element_size
       integer (c_size_t), value :: d1_size, d2_size
       integer (c_size_t), value :: d1_block_size, d2_block_size
       integer (c_size_t), value :: d1_offset, d2_offset
       type (c_ptr), value :: host_address, accel_address
     end subroutine P4A_copy_to_accel_2d

     subroutine P4A_copy_from_accel_2d(element_size, d1_size, d2_size,&
          d1_block_size, d2_block_size, d1_offset, d2_offset,&
          host_address, accel_address)&
          bind (C, name = "P4A_copy_from_accel_2d")
       use iso_c_binding
       integer (c_size_t), value :: element_size
       integer (c_size_t), value :: d1_size, d2_size
       integer (c_size_t), value :: d1_block_size, d2_block_size
       integer (c_size_t), value :: d1_offset, d2_offset
       type (c_ptr), value :: host_address, accel_address
     end subroutine P4A_copy_from_accel_2d

     subroutine P4A_copy_from_accel_3d(element_size, d1_size, d2_size, d3_size,&
          d1_block_size, d2_block_size, d3_block_size, d1_offset, d2_offset,&
          d3_offset, host_address, accel_address)&
          bind (C, name = "P4A_copy_from_accel_3d")
       use iso_c_binding
       integer (c_size_t), value :: element_size
       integer (c_size_t), value :: d1_size, d2_size, d3_size
       integer (c_size_t), value :: d1_block_size, d2_block_size, d3_block_size
       integer (c_size_t), value :: d1_offset, d2_offset, d3_offset
       type (c_ptr), value :: host_address, accel_address
     end subroutine P4A_copy_from_accel_3d

     subroutine P4A_copy_to_accel_3d (element_size, d1_size, d2_size, d3_size,&
          d1_block_size, d2_block_size, d3_block_size, d1_offset, d2_offset,&
          d3_offset, host_address, accel_address)&
          bind (C, name = "P4A_copy_to_accel_3d")
       use iso_c_binding
       integer (c_size_t), value :: element_size
       integer (c_size_t), value :: d1_size, d2_size, d3_size
       integer (c_size_t), value :: d1_block_size, d2_block_size, d3_block_size
       integer (c_size_t), value :: d1_offset, d2_offset, d3_offset
       type (c_ptr), value :: host_address, accel_address
     end subroutine P4A_copy_to_accel_3d

     subroutine P4A_copy_from_accel_4d (element_size, d1_size, d2_size, d3_size,&
          d4_size, d1_block_size, d2_block_size, d3_block_size, d4_block_size,&
          d1_offset, d2_offset, d3_offset, d4_offset, host_address, accel_address)&
          bind (C, name = "P4A_copy_from_accel_4d")
       use iso_c_binding
       integer (c_size_t), value :: element_size
       integer (c_size_t), value :: d1_size, d2_size, d3_size, d4_size
       integer (c_size_t), value :: d1_block_size, d2_block_size, d3_block_size, d4_block_size
       integer (c_size_t), value :: d1_offset, d2_offset, d3_offset, d4_offset
       type (c_ptr), value :: host_address, accel_address
     end subroutine P4A_copy_from_accel_4d

     subroutine P4A_copy_to_accel_4d (element_size, d1_size, d2_size, d3_size,&
          d4_size, d1_block_size, d2_block_size, d3_block_size, d4_block_size,&
          d1_offset, d2_offset, d3_offset, d4_offset, host_address, accel_address)&
          bind (C, name = "P4A_copy_to_accel_4d")
       use iso_c_binding
       integer (c_size_t), value :: element_size
       integer (c_size_t), value :: d1_size, d2_size, d3_size, d4_size
       integer (c_size_t), value :: d1_block_size, d2_block_size, d3_block_size, d4_block_size
       integer (c_size_t), value :: d1_offset, d2_offset, d3_offset, d4_offset
       type (c_ptr), value :: host_address, accel_address
     end subroutine P4A_copy_to_accel_4d

     subroutine P4A_accel_malloc (ptr, size) bind(C, name = "P4A_accel_malloc")
       use iso_c_binding
       type (c_ptr) :: ptr
       integer (c_size_t), value :: size
     end subroutine P4A_accel_malloc

     subroutine P4A_accel_free (ptr) bind(C, name = "P4A_accel_free")
       use iso_c_binding
       type (c_ptr), value :: ptr
     end subroutine P4A_accel_free
  end interface
end module p4a_runtime_interface
