      program hpftest21
      integer j(10)
chpf$ template t(10)
chpf$ processors p(2)
chpf$ align j(i) with t(i)
chpf$ distribute t(block) onto p
      print *, 'hpftest21 running'
chpf$ independent(i)
      do i=1, 10
         j(i) = i
      enddo
      do i=1, 10
         print *, 'j(',i,') = ', j(i)
      enddo
      do i=1, 1000000
         a = 1.0
      enddo
      print *, 'hpftest21 end'
      end
