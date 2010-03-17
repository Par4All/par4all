      program hpftest39
      real tc(500,500), ts(500,500), north(500)
CHPF$ TEMPLATE t(500,500)
CHPF$ ALIGN tc(i,j), ts(i,j) with t(i,j)
CHPF$ ALIGN north(i) with t(1,i)
CHPF$ PROCESSORS p(2,2)
CHPF$ DISTRIBUTE t(block,block) ONTO p
      print *, 'hpftest39 running'
      print *, 'thermo'
c
c initialization
      print *, 'initializing'
chpf$ independent(i)
      do i=1,500
         north(i) = 250.0
      enddo
chpf$ independent(i)
      do i=1,500
         tc(1,i) = north(i)
         ts(1,i) = north(i)
      enddo
chpf$ independent(j,i)
      do j=1,500
         do i=2,500
            tc(i,j) = 10.0
            ts(i,j) = 10.0
         enddo
      enddo
c
c iterations... should be a test, but reductions are not accepted
      print *, 'running'
      do k=1,100
c
c computation and copy back (dataparallel semantic)
chpf$ independent(j,i)         
         do j=2,499
            do i=2,499
               ts(i,j) = 0.25*
     $              (tc(i-1,j) + tc(i+1,j) + tc(i,j-1) + tc(i,j+1))
            enddo
         enddo
chpf$ independent(j,i)         
         do j=2,499
            do i=2,499
               tc(i,j) = ts(i,j)
            enddo
         enddo
      enddo
c
c print results
      print *, 'results:'
 10   format(F8.2, F8.2, F8.2, F8.2, F8.2, F8.2, F8.2, F8.2)
      do i=1, 30, 3
         write (6,10) tc(i,40), tc(i,70), tc(i,100), tc(i,130), 
     $        tc(i,160), tc(i,190), tc(i,220), tc(i,250)
      enddo
      print *, 'hpftest39 ended'
      end
