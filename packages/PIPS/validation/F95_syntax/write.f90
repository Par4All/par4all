program main
     integer i, j, a(1:10), b(100:200), c(1:10, 1:10)


! Print can be converted to write, but the format must be created
    print*,"hello world"
    write (1,fmt=*) "hello world"

! Try to handle also write with more than one data param
    i = 0
    write(*, *) i, i

! Try to handle also write with implied-do loop inside (this one was tricky)
    write (*,*) (a(i), i=1, 10), (b(i), i=100, 110)

! Try to handle also write with mutliple implied-do loop inside (this one was even more tricky)
    write (*,*) ((c(i, j), i=1, 10), j=1, 10)


end program main


