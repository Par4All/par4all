!> @addtogroup p4a_accel_stubs

!> @{

!> @defgroup p4a_accel_Fortran_stubs Equivalent stubs in Fortran of Par4All
!> runtime to have PIPS analyzes happy

!> @{

!>  Stub for copying memory from the host to the hardware accelerator.
!>
!>  Since it is a stub so that PIPS can understand it, use simple
!>  implementation with standard memory copy operations
!>
!>  Do not change the place of the pointers in the API. The host address
!>  is always in the first position...
!>
!>  @param[in] host_address is the address of a source zone in the host memory
!>
!>  @param[out] accel_address is the address of a destination zone in the
!>  accelerator memory
!>
!>  @param[in] size is the size in bytes of the memory zone to copy
      subroutine P4A_COPY_TO_ACCEL(host_address, accel_address, size)
      integer size
      character host_address(size)
      character accel_address(size)
      integer i

      do i = 1,size
        accel_address(i) = host_address(i)
      end do
      end

!>  Stub for copying memory from the hardware accelerator to the host.
!>
!>  Do not change the place of the pointers in the API. The host address
!>  is always in the first position...
!>
!>  @param[out] host_address is the address of a destination zone in the
!>  host memory
!>
!>  @param[in] accel_address is the address of a source zone in the
!>  accelerator memory
!>
!>  @param[in] size is the size in bytes of the memory zone to copy
      subroutine P4A_COPY_FROM_ACCEL(host_address, accel_address,
     &size)
      integer size
      character host_address(size)
      character accel_address(size)
      integer i

      do i = 1,size
        host_address(i) = accel_address(i)
      end do
      end


!> Stub for copying memory from the host to the hardware accelerator.
!>
!> Since it is a stub so that PIPS can understand it, use simple
!> implementation with standard memory copy operations
!>
!>  Do not change the place of the pointers in the API. The host address
!>  is always in the first position...
!>
!>  This function could be quite simpler but is designed by symmetry with
!>  other functions.
!>
!>  @param[in] elem_size is the size of one element of the array in
!>  byte
!>
!>  @param[in] d1_size is the number of elements in the array.
!>
!>  @param[in] d1_size is the number of elements in the array. It is not
!>  used but here for symmetry with functions of higher dimensionality
!>
!>  @param[in] d1_block_size is the number of element to transfer
!>
!>  @param[in] d1_offset is the element order to start the transfer to
!>
!>  @param[out] host_address point to the array on the host to write into
!>
!>  @param[in] accel_address refer to the compact memory area to read
!>  data. In the general case, accel_address may be seen as a unique idea (FIFO)
!>  and not some address in some memory space.

      subroutine P4A_COPY_FROM_ACCEL_1D (elem_size, d1_size,
     &d1_block_size, d1_offset, host_address, accel_address)

      integer d1_size
      integer elem_size
      integer d1_block_size
      integer d1_offset
      character host_address(d1_size * elem_size)
      character accel_address(d1_block_size * elem_size)
      integer i
      integer size
      integer offset

      offset  = d1_offset*elem_size
      size = d1_size * elem_size
      do i = 1,size
        host_address (offset + i) = accel_address (i)
      end do
      end

!>  Stub for copying a 1D memory zone from the hardware accelerator to the
!>  host.
!>
!>  Do not change the place of the pointers in the API. The host address
!>  is always in the first position...
!>
!>  This function could be quite simpler but is designed by symmetry with
!>  other functions.
!>
!>  @param[in] elem_size is the size of one element of the array in
!>  byte
!>
!>  @param[in] d1_size is the number of elements in the array. It is not
!>  used but here for symmetry with functions of higher dimensionality
!>
!>  @param[in] d1_size is the number of elements in the array. It is not
!>  used but here for symmetry with functions of higher dimensionality
!>
!>  @param[in] d1_block_size is the number of element to transfer
!>
!>  @param[in] d1_offset is element order to start the transfer from
!>
!>  @param[in] host_address point to the array on the host to read
!>
!>  @param[out] accel_address refer to the compact memory area to write
!>  data. In the general case, accel_address may be seen as a unique idea
!>  (FIFO) and not some address in some memory space.

      subroutine P4A_COPY_TO_ACCEL_1D(elem_size, d1_size,
     &d1_block_size, d1_offset, host_address, accel_address)

      integer d1_size
      integer elem_size
      integer d1_block_size
      integer d1_offset
      character host_address(d1_size * elem_size)
      character accel_address(d1_block_size * elem_size)
      integer i
      integer size
      integer offset

      offset  = d1_offset*elem_size
      size = d1_size * elem_size
      do i = 1,size
        accel_address (i) = host_address (offset + i)
      end do
      end

!>  Stub for copying a 2D memory zone from the host to a compact memory
!>  zone in the hardware accelerator.

      subroutine P4A_COPY_TO_ACCEL_2D (elem_size, d1_size, d2_size,
     &d1_block_size, d2_block_size, d1_offset, d2_offset, host_address,
     &accel_address)

      integer d1_size
      integer d2_size
      integer elem_size
      integer d1_block_size
      integer d2_block_size
      integer d1_offset
      integer d2_offset
      character host_address(d1_size * elem_size, d2_size * elem_size)
      character accel_address(d1_block_size * elem_size,
     &                        d2_block_size * elem_size)
      integer i_1
      integer i_2
      integer offset_1
      integer offset_2

      offset_1  = d1_offset*elem_size
      offset_2  = d2_offset*elem_size
      do i_2 = 1, d2_block_size * elem_size
        do i_1 = 1, d1_block_size * elem_size
          accel_address (i_1, i_2) = host_address (offset_1 + i_1,
     &                                             offset_2 + i_2)
         end do
      end do
      end

!>  Stub for copying memory from the hardware accelerator to a 2D array in
!>  the host.

      subroutine P4A_COPY_FROM_ACCEL_2D (elem_size, d1_size, d2_size,
     &d1_block_size, d2_block_size, d1_offset, d2_offset, host_address,
     &accel_address)

      integer d1_size
      integer d2_size
      integer elem_size
      integer d1_block_size
      integer d2_block_size
      integer d1_offset
      integer d2_offset
      character host_address(d1_size * elem_size, d2_size * elem_size)
      character accel_address(d1_block_size * elem_size,
     &                        d2_block_size * elem_size)
      integer i_1
      integer i_2
      integer offset_1
      integer offset_2

      offset_1  = d1_offset*elem_size
      offset_2  = d2_offset*elem_size
      do i_2 = 1, d2_block_size * elem_size
        do i_1 = 1, d1_block_size * elem_size
          host_address (offset_1 + i_1, offset_2 + i_2) =
     &    accel_address (i_1, i_2)
        end do
      end do
      end

!>  Stub for copying a 3D memory zone from the host to a compact memory
!>  zone in the hardware accelerator.

      subroutine P4A_COPY_TO_ACCEL_3D (elem_size, d1_size, d2_size,
     &d3_size,
     &d1_block_size, d2_block_size, d3_block_size,
     &d1_offset, d2_offset, d3_offset, host_address,
     &accel_address)

      integer d1_size
      integer d2_size
      integer d3_size
      integer elem_size
      integer d1_block_size
      integer d2_block_size
      integer d3_block_size
      integer d1_offset
      integer d2_offset
      integer d3_offset
      character host_address(d1_size * elem_size,
     &                       d2_size * elem_size,
     &                       d3_size * elem_size)
      character accel_address(d1_block_size * elem_size,
     &                        d2_block_size * elem_size,
     &                        d3_block_size * elem_size)
      integer i_1
      integer i_2
      integer i_3
      integer offset_1
      integer offset_2
      integer offset_3

      offset_1  = d1_offset*elem_size
      offset_2  = d2_offset*elem_size
      offset_3  = d3_offset*elem_size
      do i_3 = 1, d3_block_size * elem_size
         do i_2 = 1, d2_block_size * elem_size
            do i_1 = 1, d1_block_size * elem_size
               accel_address (i_1, i_2, i_3)
     &         =
     &         host_address  (offset_1 + i_1,
     &                        offset_2 + i_2,
     &                        offset_3 + i_3)
            end do
         end do
      end do
      end

!>  Stub for copying memory from the hardware accelerator to a 3D array in
!>  the host.

      subroutine P4A_COPY_FROM_ACCEL_3D (elem_size, d1_size, d2_size,
     &d3_size,
     &d1_block_size, d2_block_size, d3_block_size,
     &d1_offset, d2_offset, d3_offset, host_address,
     &accel_address)

      integer d1_size
      integer d2_size
      integer d3_size
      integer elem_size
      integer d1_block_size
      integer d2_block_size
      integer d3_block_size
      integer d1_offset
      integer d2_offset
      integer d3_offset
      character host_address(d1_size * elem_size,
     &                       d2_size * elem_size,
     &                       d3_size * elem_size)
      character accel_address(d1_block_size * elem_size,
     &                        d2_block_size * elem_size,
     &                        d3_block_size * elem_size)
      integer i_1
      integer i_2
      integer i_3
      integer offset_1
      integer offset_2
      integer offset_3

      offset_1  = d1_offset*elem_size
      offset_2  = d2_offset*elem_size
      offset_3  = d3_offset*elem_size
      do i_3 = 1, d3_block_size * elem_size
         do i_2 = 1, d2_block_size * elem_size
            do i_1 = 1, d1_block_size * elem_size
               host_address (offset_1 + i_1,
     &                       offset_2 + i_2,
     &                       offset_3 + i_3)
     &         =
     &         accel_address (i_1, i_2, i_3)
            end do
         end do
      end do
      end

!>  Stub for copying a 4D memory zone from the host to a compact memory
!>  zone in the hardware accelerator.

      subroutine P4A_COPY_TO_ACCEL_4D (elem_size, d1_size, d2_size,
     &d3_size, d4_size,
     &d1_block_size, d2_block_size, d3_block_size, d4_block_size,
     &d1_offset, d2_offset, d3_offset, d4_offset, host_address,
     &accel_address)

      integer d1_size
      integer d2_size
      integer d3_size
      integer d4_size
      integer elem_size
      integer d1_block_size
      integer d2_block_size
      integer d3_block_size
      integer d4_block_size
      integer d1_offset
      integer d2_offset
      integer d3_offset
      integer d4_offset
      character host_address(d1_size * elem_size,
     &                       d2_size * elem_size,
     &                       d3_size * elem_size,
     &                       d4_size * elem_size)
      character accel_address(d1_block_size * elem_size,
     &                        d2_block_size * elem_size,
     &                        d3_block_size * elem_size,
     &                        d4_block_size * elem_size)
      integer i_1
      integer i_2
      integer i_3
      integer i_4
      integer offset_1
      integer offset_2
      integer offset_3
      integer offset_4

      offset_1  = d1_offset*elem_size
      offset_2  = d2_offset*elem_size
      offset_3  = d3_offset*elem_size
      offset_4  = d4_offset*elem_size
      do i_4 = 1, d4_block_size * elem_size
         do i_3 = 1, d3_block_size * elem_size
            do i_2 = 1, d2_block_size * elem_size
               do i_1 = 1, d1_block_size * elem_size
                  accel_address (i_1, i_2, i_3, i_4)
     &            =
     &            host_address  (offset_1 + i_1,
     &                           offset_2 + i_2,
     &                           offset_3 + i_3,
     &                           offset_4 + i_4)
                  end do
            end do
         end do
      end do
      end

!>  Stub for copying memory from the hardware accelerator to a 4D array in
!>  the host.

      subroutine P4A_COPY_FROM_ACCEL_4D (elem_size, d1_size, d2_size,
     &d3_size, d4_size,
     &d1_block_size, d2_block_size, d3_block_size, d4_block_size,
     &d1_offset, d2_offset, d3_offset, d4_offset, host_address,
     &accel_address)

      integer d1_size
      integer d2_size
      integer d3_size
      integer d4_size
      integer elem_size
      integer d1_block_size
      integer d2_block_size
      integer d3_block_size
      integer d4_block_size
      integer d1_offset
      integer d2_offset
      integer d3_offset
      integer d4_offset
      character host_address(d1_size * elem_size,
     &                       d2_size * elem_size,
     &                       d3_size * elem_size,
     &                       d4_size * elem_size)
      character accel_address(d1_block_size * elem_size,
     &                        d2_block_size * elem_size,
     &                        d3_block_size * elem_size,
     &                        d4_block_size * elem_size)
      integer i_1
      integer i_2
      integer i_3
      integer i_4
      integer offset_1
      integer offset_2
      integer offset_3
      integer offset_4

      offset_1  = d1_offset*elem_size
      offset_2  = d2_offset*elem_size
      offset_3  = d3_offset*elem_size
      offset_4  = d4_offset*elem_size
      do i_4 = 1, d4_block_size * elem_size
         do i_3 = 1, d3_block_size * elem_size
            do i_2 = 1, d2_block_size * elem_size
               do i_1 = 1, d1_block_size * elem_size
                  host_address (offset_1 + i_1,
     &                          offset_2 + i_2,
     &                          offset_3 + i_3,
     &                          offset_4 + i_4)
     &         =
     &         accel_address (i_1, i_2, i_3, i_4)
               end do
            end do
         end do
      end do
      end

!>  Stub for copying a 5D memory zone from the host to a compact memory
!>  zone in the hardware accelerator.

      subroutine P4A_COPY_TO_ACCEL_5D (elem_size, d1_size, d2_size,
     &d3_size, d4_size, d5_size,
     &d1_block_size, d2_block_size, d3_block_size, d4_block_size,
     &d5_block_size,
     &d1_offset, d2_offset, d3_offset, d4_offset, d5_offset,
     & host_address, accel_address)

      integer d1_size
      integer d2_size
      integer d3_size
      integer d4_size
      integer d5_size
      integer elem_size
      integer d1_block_size
      integer d2_block_size
      integer d3_block_size
      integer d4_block_size
      integer d5_block_size
      integer d1_offset
      integer d2_offset
      integer d3_offset
      integer d4_offset
      integer d5_offset
      character host_address(d1_size * elem_size,
     &                       d2_size * elem_size,
     &                       d3_size * elem_size,
     &                       d4_size * elem_size,
     &                       d5_size * elem_size)
      character accel_address(d1_block_size * elem_size,
     &                        d2_block_size * elem_size,
     &                        d3_block_size * elem_size,
     &                        d4_block_size * elem_size,
     &                        d5_block_size * elem_size)
      integer i_1
      integer i_2
      integer i_3
      integer i_4
      integer i_5
      integer offset_1
      integer offset_2
      integer offset_3
      integer offset_4
      integer offset_5

      offset_1  = d1_offset*elem_size
      offset_2  = d2_offset*elem_size
      offset_3  = d3_offset*elem_size
      offset_4  = d4_offset*elem_size
      offset_5  = d5_offset*elem_size
      do i_5 = 1, d5_block_size * elem_size
         do i_4 = 1, d4_block_size * elem_size
            do i_3 = 1, d3_block_size * elem_size
               do i_2 = 1, d2_block_size * elem_size
                  do i_1 = 1, d1_block_size * elem_size
                     accel_address (i_1, i_2, i_3, i_4, i_5)
     &               =
     &               host_address  (offset_1 + i_1,
     &                              offset_2 + i_2,
     &                              offset_3 + i_3,
     &                              offset_4 + i_4,
     &                              offset_5 + i_5)
                  end do
               end do
            end do
         end do
      end do
      end

!>  Stub for copying memory from the hardware accelerator to a 5D array in
!>  the host.

      subroutine P4A_COPY_FROM_ACCEL_5D (elem_size, d1_size, d2_size,
     &d3_size, d4_size, d5_size,
     &d1_block_size, d2_block_size, d3_block_size, d4_block_size,
     &d5_block_size,
     &d1_offset, d2_offset, d3_offset, d4_offset, d5_offset,
     &host_address, accel_address)

      integer d1_size
      integer d2_size
      integer d3_size
      integer d4_size
      integer d5_size
      integer elem_size
      integer d1_block_size
      integer d2_block_size
      integer d3_block_size
      integer d4_block_size
      integer d5_block_size
      integer d1_offset
      integer d2_offset
      integer d3_offset
      integer d4_offset
      integer d5_offset
      character host_address(d1_size * elem_size,
     &                       d2_size * elem_size,
     &                       d3_size * elem_size,
     &                       d4_size * elem_size,
     &                       d5_size * elem_size)
      character accel_address(d1_block_size * elem_size,
     &                        d2_block_size * elem_size,
     &                        d3_block_size * elem_size,
     &                        d4_block_size * elem_size,
     &                        d5_block_size * elem_size)
      integer i_1
      integer i_2
      integer i_3
      integer i_4
      integer i_5
      integer offset_1
      integer offset_2
      integer offset_3
      integer offset_4
      integer offset_5

      offset_1  = d1_offset*elem_size
      offset_2  = d2_offset*elem_size
      offset_3  = d3_offset*elem_size
      offset_4  = d4_offset*elem_size
      offset_5  = d5_offset*elem_size
      do i_5 = 1, d5_block_size * elem_size
         do i_4 = 1, d4_block_size * elem_size
            do i_3 = 1, d3_block_size * elem_size
               do i_2 = 1, d2_block_size * elem_size
                  do i_1 = 1, d1_block_size * elem_size
                     host_address (offset_1 + i_1,
     &                             offset_2 + i_2,
     &                             offset_3 + i_3,
     &                             offset_4 + i_4,
     &                             offset_5 + i_5)
     &               =
     &               accel_address (i_1, i_2, i_3, i_4, i_5)
                  end do
               end do
            end do
         end do
      end do
      end

!>  Stub for copying a 6D memory zone from the host to a compact memory
!>  zone in the hardware accelerator.

      subroutine P4A_COPY_TO_ACCEL_6D (elem_size, d1_size, d2_size,
     &d3_size, d4_size, d5_size, d6_size,
     &d1_block_size, d2_block_size, d3_block_size, d4_block_size,
     &d5_block_size, d6_block_size,
     &d1_offset, d2_offset, d3_offset, d4_offset, d5_offset, d6_offset,
     &host_address, accel_address)

      integer d1_size
      integer d2_size
      integer d3_size
      integer d4_size
      integer d5_size
      integer d6_size
      integer elem_size
      integer d1_block_size
      integer d2_block_size
      integer d3_block_size
      integer d4_block_size
      integer d5_block_size
      integer d6_block_size
      integer d1_offset
      integer d2_offset
      integer d3_offset
      integer d4_offset
      integer d5_offset
      integer d6_offset
      character host_address(d1_size * elem_size,
     &                       d2_size * elem_size,
     &                       d3_size * elem_size,
     &                       d4_size * elem_size,
     &                       d5_size * elem_size,
     &                       d6_size * elem_size)
      character accel_address(d1_block_size * elem_size,
     &                        d2_block_size * elem_size,
     &                        d3_block_size * elem_size,
     &                        d4_block_size * elem_size,
     &                        d5_block_size * elem_size,
     &                        d6_block_size * elem_size)
      integer i_1
      integer i_2
      integer i_3
      integer i_4
      integer i_5
      integer i_6
      integer offset_1
      integer offset_2
      integer offset_3
      integer offset_4
      integer offset_5
      integer offset_6

      offset_1  = d1_offset*elem_size
      offset_2  = d2_offset*elem_size
      offset_3  = d3_offset*elem_size
      offset_4  = d4_offset*elem_size
      offset_5  = d5_offset*elem_size
      offset_6  = d6_offset*elem_size
      do i_6 = 1, d6_block_size * elem_size
         do i_5 = 1, d5_block_size * elem_size
            do i_4 = 1, d4_block_size * elem_size
               do i_3 = 1, d3_block_size * elem_size
                  do i_2 = 1, d2_block_size * elem_size
                     do i_1 = 1, d1_block_size * elem_size
                        accel_address (i_1, i_2, i_3, i_4, i_5, i_6)
     &                  =
     &                  host_address  (offset_1 + i_1,
     &                                 offset_2 + i_2,
     &                                 offset_3 + i_3,
     &                                 offset_4 + i_4,
     &                                 offset_5 + i_5,
     &                                 offset_6 + i_6)
                     end do
                  end do
               end do
            end do
         end do
      end do
      end

!>  Stub for copying memory from the hardware accelerator to a 6D array in
!>  the host.

      subroutine P4A_COPY_FROM_ACCEL_6D (elem_size, d1_size, d2_size,
     &d3_size, d4_size, d5_size, d6_size,
     &d1_block_size, d2_block_size, d3_block_size, d4_block_size,
     &d5_block_size, d6_block_size,
     &d1_offset, d2_offset, d3_offset, d4_offset, d5_offset, d6_offset,
     &host_address, accel_address)

      integer d1_size
      integer d2_size
      integer d3_size
      integer d4_size
      integer d5_size
      integer d6_size
      integer elem_size
      integer d1_block_size
      integer d2_block_size
      integer d3_block_size
      integer d4_block_size
      integer d5_block_size
      integer d6_block_size
      integer d1_offset
      integer d2_offset
      integer d3_offset
      integer d4_offset
      integer d5_offset
      integer d6_offset
      character host_address(d1_size * elem_size,
     &                       d2_size * elem_size,
     &                       d3_size * elem_size,
     &                       d4_size * elem_size,
     &                       d5_size * elem_size,
     &                       d6_size * elem_size)
      character accel_address(d1_block_size * elem_size,
     &                        d2_block_size * elem_size,
     &                        d3_block_size * elem_size,
     &                        d4_block_size * elem_size,
     &                        d5_block_size * elem_size,
     &                        d6_block_size * elem_size)
      integer i_1
      integer i_2
      integer i_3
      integer i_4
      integer i_5
      integer i_6
      integer offset_1
      integer offset_2
      integer offset_3
      integer offset_4
      integer offset_5
      integer offset_6

      offset_1  = d1_offset*elem_size
      offset_2  = d2_offset*elem_size
      offset_3  = d3_offset*elem_size
      offset_4  = d4_offset*elem_size
      offset_5  = d5_offset*elem_size
      offset_6  = d6_offset*elem_size
      do i_6 = 1, d6_block_size * elem_size
         do i_5 = 1, d5_block_size * elem_size
            do i_4 = 1, d4_block_size * elem_size
               do i_3 = 1, d3_block_size * elem_size
                  do i_2 = 1, d2_block_size * elem_size
                     do i_1 = 1, d1_block_size * elem_size
                        host_address (offset_1 + i_1,
     &                                offset_2 + i_2,
     &                                offset_3 + i_3,
     &                                offset_4 + i_4,
     &                                offset_5 + i_5,
     &                                offset_6 + i_6)
     &                  =
     &                  accel_address (i_1, i_2, i_3, i_4, i_5, i_6)
                     end do
                  end do
               end do
            end do
         end do
      end do
      end

!>  Stub for allocating memory on the hardware accelerator.
!>
!>  @param[out] address is the address of a variable that is updated by
!>  this macro to contains the address of the allocated memory block
!>  @param[in] size is the size to allocate in bytes
      subroutine P4A_ACCEL_MALLOC(address, size)
      integer size
      integer address
! Do nothing since it is not representable in Fortran 77...
      end


!>  Stub for freeing memory on the hardware accelerator.
!>
!>  @param[in] address is the address of a previously allocated memory
!>  zone on the hardware accelerator
      subroutine P4A_ACCEL_FREE(address)
      integer address
! Do nothing since it is not representable in Fortran 77...
      end
!> @}
!> @}
