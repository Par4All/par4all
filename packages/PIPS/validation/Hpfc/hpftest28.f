      program hpftest28
      real tc(10,10), ts(10,10), north(10)
CHPF$ TEMPLATE t(10,10)
CHPF$ ALIGN tc(i,j), ts(i,j) with t(i,j)
CHPF$ ALIGN north(i) with t(1,i)
CHPF$ PROCESSORS p(2,2)
CHPF$ DISTRIBUTE t(block,block) ONTO p
      print *, 'hpftest28 running'
      print *, 'thermo'
      print *, 'please enter vector north(1:10)'
      do i=1,10
         read *, north(i)
      enddo
c
c initialization
      print *, 'initializing'
chpf$ independent(i)
      do i=1,10
         tc(1,i) = north(i)
         ts(1,i) = north(i)
      enddo
chpf$ independent(i,j)
      do i=2,10
         do j=1,10
            tc(i,j) = 0.0
            ts(i,j) = 0.0
         enddo
      enddo
c
c iterations... should be a test, but reductions are not accepted
      do k=1,50
c
c computation and copy back (dataparallel semantic)
chpf$ independent(i,j)         
         do i=2,9
            do j=2,9
               ts(i,j) = 0.25*
     $              (tc(i-1,j) + tc(i+1,j) + tc(i,j-1) + tc(i,j+1))
            enddo
         enddo
chpf$ independent(i,j)         
         do i=2,9
            do j=2,9
               tc(i,j) = ts(i,j)
            enddo
         enddo
      enddo
c
c print results
      print *, 'results:'
 10   format(F8.2, F8.2, F8.2, F8.2, F8.2, F8.2, F8.2, F8.2)
      do i=2,9
         write (6,10) tc(i,2), tc(i,3), tc(i,4), tc(i,5), tc(i,6), 
     $        tc(i,7), tc(i,8), tc(i,9)
      enddo
      print *, 'hpftest28 ended'
      end
