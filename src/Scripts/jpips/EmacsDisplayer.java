/*
  $Id$

  Send displays to emacs.

  $Log: EmacsDisplayer.java,v $
  Revision 1.1  1998/11/12 17:20:51  coelho
  Initial revision

*/

package JPips;

import java.io.*;

public class EmacsDisplayer 
  extends Displayer
{
  static public final String BEGIN = "\200";
  static public final String END = "\201";

  PrintStream to_emacs;

  public EmacsDisplayer(PrintStream out)
  {
    this.to_emacs = out;
  }
  
  boolean display(File file, boolean locked, boolean writable)
  {
    to_emacs.print(BEGIN + "WINDOW_NUMBER:" + 4 + END +
		   BEGIN + "Sequential View:" + file + END);
    to_emacs.flush();
    return true;
  }
  
  void display(String name, String string, boolean locked, boolean writable)
  {
    display(new File(name), locked, writable);
  }
}
