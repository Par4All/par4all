      program hpftest37
      real tc(250,250), ts(250,250), north(250)
CHPF$ TEMPLATE t(250,250)
CHPF$ ALIGN tc(i,j), ts(i,j) with t(i,j)
CHPF$ ALIGN north(i) with t(1,i)
CHPF$ PROCESSORS p(4)
CHPF$ DISTRIBUTE t(block, *) ONTO p
      print *, 'hpftest37 running'
      print *, 'thermo'
c
c initialization
      print *, 'initializing'
chpf$ independent(i)
      do i=1,250
         north(i) = 250.0
      enddo
chpf$ independent(i)
      do i=1,250
         tc(1,i) = north(i)
         ts(1,i) = north(i)
      enddo
chpf$ independent(i,j)
      do i=2,250
         do j=1,250
            tc(i,j) = 10.0
            ts(i,j) = 10.0
         enddo
      enddo
c
c iterations... should be a test, but reductions are not accepted
      print *, 'running'
      do k=1,500
c
c computation and copy back (dataparallel semantic)
chpf$ independent(i,j)         
         do i=2,249
            do j=2,249
               ts(i,j) = 0.25*
     $              (tc(i-1,j) + tc(i+1,j) + tc(i,j-1) + tc(i,j+1))
            enddo
         enddo
chpf$ independent(i,j)         
         do i=2,249
            do j=2,249
               tc(i,j) = ts(i,j)
            enddo
         enddo
      enddo
c
c print results
      print *, 'results:'
 10   format(F8.2, F8.2, F8.2, F8.2, F8.2, F8.2, F8.2, F8.2)
      do i=25,200,25
         write (6,10) tc(i,50), tc(i,75), tc(i,100), tc(i,125), 
     $        tc(i,150), tc(i,175), tc(i,200), tc(i,225)
      enddo
      print *, 'hpftest37 ended'
      end
