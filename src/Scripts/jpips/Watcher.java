
/** $Id$
  * $Log: Watcher.java,v $
  * Revision 1.1  1998/06/30 15:03:42  didry
  * Initial revision
  *
  */


package JPips;


import java.lang.*;
import java.util.*;
import java.io.*;


/** A class that s the tpips output stream.
  * @author Francois Didry
  */  
public class Watcher implements Runnable
{


  public	Process		tpips;		//tpips instance
  
  
  public Watcher(Process tpips)
    {
      this.tpips = tpips;
    }


  public void run()
    {
      try
        {
	  System.out.println("Tpips watcher running");
          tpips.waitFor();
          System.out.println("tpips down");
	}
      catch(InterruptedException e)
        {
	  System.out.println(e);
	}
    }


}
