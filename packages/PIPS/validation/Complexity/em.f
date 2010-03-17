      program em
      n1 = 10
      n2 = 30
      call add(n1, n2, n)
      print *,n
      do 100 i = n1,n
         u = 1.
 100  continue
      end
c
      subroutine add(m1, m2, m)
      m = m1 + m2
      return
      end
