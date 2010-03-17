      program intercom
      common /sum/ s
      integer s, i, j
      s = 5
      call zero
      do i=1, 10
         do j=1, i
            s = s + i
         enddo
      enddo
      do i=1, 10
         do j=1, i
            s = s + i
            call inc
         enddo
      enddo
      do i=1, 10
         do j=1, i
            s = s + i
            call zero
         enddo
      enddo
      print *, s
      end
      subroutine inc
      common /sum/ s
      s = s + 1
      end
      subroutine zero
      common /sum/ s
      s = 0
      end
!tps$ echo PROPER REDUCTIONS
!tps$ activate PRINT_CODE_PROPER_REDUCTIONS
!tps$ display PRINTED_FILE($ALL)
!tps$ echo CUMULATED REDUCTIONS
!tps$ activate PRINT_CODE_CUMULATED_REDUCTIONS
!tps$ display PRINTED_FILE($ALL)
