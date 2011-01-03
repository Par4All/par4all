module p4a_runtime_interface
  interface
     subroutine P4A_accel_malloc (ptr, size) bind(C)
       use iso_c_binding
       type (c_ptr) :: ptr
       integer (c_size_t) , value :: size
     end subroutine P4A_accel_malloc

     subroutine P4A_accel_free (ptr) bind(C)
       use iso_c_binding
       type (c_ptr), value :: ptr
     end subroutine P4A_accel_free
  end interface
end module p4a_runtime_interface
