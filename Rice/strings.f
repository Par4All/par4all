      program strings

C     Parallelization of string manipulations

      character*80 tab(10)

      do i = 1, 10
         tab(i) = 'Hello! '
      enddo

      do i = 2, 10
         tab(i) = tab(i)(1:7) // tab(i-1)(1:7*i-7)
      enddo

      do i = 1, 10
         print *, tab(i)
      enddo

      end
