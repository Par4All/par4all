c
      Subroutine CHAR_(ch,ierr0,ierr1)
      implicit real*4 (A-H,O-Z)
      character*16 string0
      character*75 string1
      character*1 ch
      data string0/'+,.eE-1234567890'/ 
      data string1(1:25)/'abcdfghijklmnopqrstuvwxyz'/
      data string1(26:50)/'ABCDFGHIJKLMNOPQRSTUVWXYZ'/
      data string1(51:75)/'!@#$%^&*()_=[]{}:;"|\\~`/?'/
      save string0,string1
      ierr0=1
      ierr1=0
      do 1 k=1,16
       if(ch.eq.string0(k:k)) ierr0=0
 1    continue
      if(ierr0.eq.1) then
       do 2 k=1,75
        if(ch.eq.string1(k:k)) then
         ierr1=1
        endif
 2     continue
      endif
      return
      end
