      program reduction
      integer i, n
      parameter (n=10)
      real s, a(n), p
      s = 1.0
      s = s - 2.0
      s = s + s
      s = 3.0 + s
      do i=1, n
         a(i) = real(i)
      enddo

      do i=1, n
         s = s + a(i)
         s = s - a(i)
      enddo
      print *, 'sum is ', s

      p = 1.0
      do i=1, n
         if (MOD(i,2).eq.0) then
            p = a(i) * p
         else
            p = p / a(i)
         endif
      enddo
      print *, 'p is ', p

      end
!tps$ echo REDUCTION validation script      
!tps$ activate PRINT_CODE_PROPER_REDUCTIONS
!tps$ display PRINTED_FILE
!tps$ activate PRINT_CODE_CUMULATED_REDUCTIONS
!tps$ display PRINTED_FILE
