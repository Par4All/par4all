      program ioerrend

      read(*, 100, ERR=1000, END=2000) i
 100  format(i5)

      stop

 1000 continue
      print *, 'end of file'
      stop

 2000 continue
      print *, 'read error'

      end
