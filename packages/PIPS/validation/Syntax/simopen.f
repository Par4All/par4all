      program simopen

C     Bug dans simula (sidolo package)

      open(isca,file=fichs,recl=lonrec*(nscal+maxvobs+nscal2),
     .     access='direct')

      end
