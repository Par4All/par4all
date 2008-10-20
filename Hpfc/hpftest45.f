      program hpftest45
c
c version avec 8 processeurs
c
      parameter (n=500)
      parameter (time=100)
      real tc(n,n), ts(n,n), north(n)
CHPF$ TEMPLATE t(n,n)
CHPF$ ALIGN tc(i,j), ts(i,j) with t(i,j)
CHPF$ ALIGN north(i) with t(1,i)
CHPF$ PROCESSORS p(2,4)
CHPF$ DISTRIBUTE t(block,block) ONTO p
      print *, 'hpftest45 running'
      print *, 'thermo'
c
c initialization
      print *, 'initializing'
chpf$ independent(i)
      do i=1,n
         north(i) = 100.0
      enddo
chpf$ independent(i)
      do i=1,n
         tc(1,i) = north(i)
         ts(1,i) = north(i)
      enddo
chpf$ independent(j,i)
      do j=1,n
         do i=2,n
            tc(i,j) = 10.0
            ts(i,j) = 10.0
         enddo
      enddo
c
c iterations... should be a test, but reductions are not accepted
      print *, 'running'
      do k=1,time
c
c computation and copy back (dataparallel semantic)
chpf$ independent(j,i)         
         do j=2,n-1
            do i=2,n-1
               ts(i,j) = 0.25*
     $              (tc(i-1,j) + tc(i+1,j) + tc(i,j-1) + tc(i,j+1))
            enddo
         enddo
chpf$ independent(j,i)         
         do j=2,n-1
            do i=2,n-1
               tc(i,j) = ts(i,j)
            enddo
         enddo
      enddo
c
c print results
      print *, 'results:'
 10   format(F8.2, F8.2, F8.2, F8.2, F8.2)
      do i=2, 10, 2
         write (6,10) tc(i,2), tc(i,4), tc(i,6), tc(i,8), tc(i,10)
      enddo
      print *, 'hpftest45 ended'
      end
