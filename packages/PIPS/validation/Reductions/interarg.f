      program interarg
      integer s, i
      s = 0
      do i=1, 10
         call addto(s, i)
      enddo
      print *, s
      end
      subroutine addto(x, j)
      integer x, j
      x = x + j
      end
!tps$ echo PROPER REDUCTIONS
!tps$ activate PRINT_CODE_PROPER_REDUCTIONS
!tps$ display PRINTED_FILE($ALL)
!tps$ echo CUMULATED REDUCTIONS
!tps$ activate PRINT_CODE_CUMULATED_REDUCTIONS
!tps$ display PRINTED_FILE($ALL)
