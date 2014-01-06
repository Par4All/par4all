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
package fr.ensmp.cri.jpips;

import java.applet.*;
import java.net.*;
import java.util.*;
import java.io.*;

import fr.ensmp.cri.jpips.Pawt.*;

import java.awt.event.*;
import java.awt.*;


/** A  Graph  Displayer  
 * It creates a process starting daVinci 
 * and displays the graph file that is  in daVinci format
 * @author Francois Didry
 */
public class GraphDisplayer extends Displayer
{
  static public Vector infos; //contains the davinvi extensions
  public PFrame frame;  //frame of jpips
  public Process davinci; //davinci process
  
  public final String  DAVINCI = "daVinci",
    TGRAPH="-graph",
    TDAVINCI="-daVinci";
  
  
  /** Sets the number of displayed windows to 0.
   * Creates the vector of windows.
   */
  public GraphDisplayer(PFrame frame)
  {
    super();
    this.frame = frame;
  }
  
  /** Sets the extension info table.
   */
  static 
  {
    infos = new Vector();
    infos.addElement(new Extension("daVinci","daVinci call graph view"));
    infos.addElement(new Extension("-daVinci","daVinci control graph view"));
  }
  
  /** Calls Davinci onto the  text file.
   @param file the file to display.
   @return whether a new text frame was created to display the file.
   */
  public boolean display(File file, boolean locked, boolean writable)
  {
    String dav = DAVINCI + " " + file.getPath();
    if (dav.endsWith(TGRAPH))
    { 
      int index = dav.lastIndexOf('-');
      String dav1 = dav.substring(1,index);
      dav = dav1 + TDAVINCI;
    }
    try
    {
      davinci = Runtime.getRuntime().exec(dav);
    }
    catch (Exception e)
    {
      System.err.println("GraphDisplayer.display Exception :" + e);
    }
    
    return false;
  } 
  
  public void display(String title, String text,
                      boolean locked, boolean writable)
  {
    ;
  }
  
  /** 
   *@param file a file 
   *@return true whether the specified file has daVinci extension 
   */
  public boolean davinciExtension(File file)
  { 
    String name = file.getName();
    int index = name.indexOf(".");
    boolean result=false;
    if(index == -1) 
      return result;
    String extension = name.substring(index+1);  
    
    for(int i=0; i<infos.size(); i++)
    {
      Extension e = (Extension)infos.elementAt(i); 
      if(extension.endsWith(e.getExtension()))
        result = true;
    } 
    return result;
  }
  /** 
   *@param file a file 
   *@return true whether the specified file has daVinci extension 
   */
  public boolean controlgraphExtension(File file)
  { 
    String name = file.getName();
    int index = name.indexOf(".");
    boolean result=false;
    if(index == -1) 
      return result;
    String extension = name.substring(index+1);  
    if(extension.endsWith("-graph"))
      result = true;
    
    return result;
  }
  
}
