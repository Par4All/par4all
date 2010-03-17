c unstructured
      program hpftest61
      integer n, m, i, j
      parameter (n=10)
      integer a(n,n)
chpf$ template t(n,n)
chpf$ processors p(2,2)
chpf$ distribute t(block, block) onto p
chpf$ align a(i,j) with t(i,j)
      print *, 'hpftest61 running'
 10   print *, 'please enter a value'
      read *, m
      if (m.gt.n.or.m.lt.1) goto 10
      print *, 'loop nests'
chpf$ independent(i,j)
      do i=1, n
         do j=1, n
            a(j,i) = m+i+j
         enddo
      enddo
      print *, 'result:'
      do i=1, n
         print *, (a(j,i),j=1,n)
      enddo
      end
