/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
/*
 Send displays to emacs.
*/

package fr.ensmp.cri.jpips;

import java.io.*;

public class EmacsDisplayer 
  extends Displayer
{
  static public final String BEGIN = "\200";
  static public final String END = "\201";
  public Process g2davinci; 
  public Process davinci; 
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
  
  boolean graphdisplay(File file, boolean locked, boolean writable)
  {
    String filename = file.getPath();
    try
    {
      String ps = "pips_graph2daVinci -launch_daVinci " + filename;
      System.err.println("running: " + ps);
      g2davinci = Runtime.getRuntime().exec(ps);
      System.err.println("done");
    }
    catch (Exception e)
    {
      System.err.println("EmacsDisplayer.graphdisplay Exception :" + e);
    }
    
    /*
     int index = filename.lastIndexOf('-');
     String ft = filename.substring(1,index);
     String davinciname = ft + "-daVinci";
     String pstring1 =  "daVinci " + davinciname;
     try
     {
     davinci = Runtime.getRuntime().exec(pstring1);
     }
     catch (Exception e)
     {
     System.err.println("EmacsDisplayer.graphdisplay Exception :" + e);
     }
     */      
    
    //to_emacs.print(BEGIN + "VIEW_DAVINCI_GRAPH:" + filename + END);
    //to_emacs.flush(); 
    
    return true;
  }
}
