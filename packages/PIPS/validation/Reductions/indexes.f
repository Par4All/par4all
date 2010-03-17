      program indexes
      integer i,j
      integer s(10)
      do i=1, 10
         do j=1, 10
            s(j)=s(j)+i
         enddo
         s(i)=s(i)+2
      enddo
      do i=1, 10
         s(3)=s(3)+i+j
         s(5)=i+s(5)+j
         s(7)=i+j+s(7)
      enddo
      end
!tps$ activate PRINT_CODE_PROPER_REDUCTIONS
!tps$ display PRINTED_FILE
!tps$ activate PRINT_CODE_CUMULATED_REDUCTIONS
!tps$ display PRINTED_FILE
