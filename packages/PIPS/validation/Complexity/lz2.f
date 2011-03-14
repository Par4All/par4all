      program LZ2
      call sub2(7)
      end
c
      subroutine sub2(m)
      integer m, ii, i, j, jj
      do 10 i = 1, m
         ii = i + 1
         do 20 j = ii, m + 2
            jj = i + j - 2
            do 30 k = jj + 10, 100
               t = t + 1.0
               u = u + 1.0
 30         continue
 20      continue
 10   continue
      return
      end
