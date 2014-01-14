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

import java.lang.*;

import java.util.*;
import java.io.*;
import java.applet.*;

import javax.swing.*;

import fr.ensmp.cri.jpips.Pawt.*;

/** A class that creates a tpips process.
 * It defines the methods to interact with the process.  
 * 
 * @author Francois Didry
 */
public class TPips 
  implements Requestable
{
  public Resetable jpips;
  public Process  tpips;  //tpips process
  public Thread  listener, //listener instance
    watcher; //watcher instance
  public PList  list;  //contains the modules
  
  public PrintWriter out;  //output stream to tpips
  public BufferedReader in,  //input stream from tpips
    inErr;  //input stream from tpips
  
  public Vector  optionVector; //the options of jpips
  
  public TextDisplayer textDisplayer; //textdisplayer instance
  public GraphDisplayer graphDisplayer; //graphdisplayer instance
  public        EmacsDisplayer  emacsDisplayer;
  
  public PFrame  frame;  //frame of jpips
  public File  directory; //current directory
  
  
  //the followings are tags used for the parsing of tpips messages
  public final String  DONE = "done",
    JPIPS = "#jpips:",
    BEGIN_REQUEST = "begin_user_request",
    BEGIN_ERROR = "begin_user_error",
    END_REQUEST = "end_user_request",
    END_ERROR = "end_user_error",
    MODULE_TAG = "$MODULES",
    TAG = "$TAG",
    INFO = "info",
    DISPLAY = "display",
    MODULES = "modules",
    SHOW = "show",
    EOL = "\n",
    SPACE = " ",
    DIR = ".",
    WORKSPACE = "workspace",
    DIRECTORY = "directory",
    NONE = "<none>",
    RESULT = "result";
  
  
  public final String  execute = "tpips -jw",
    TO_TPIPS = "to tpips   : ",
    FROM_TPIPS = "from tpips : ",
    moduleMessage = "No modules selected!",
    error = "Error",
    GRAPH="-graph";
  
  
  /** Sets the necessary instances.
   */  
  public TPips(Resetable jpips)
  {
    this.jpips = jpips;
  }
  
  /** properties with emacs mode...
   */
  static final String 
    PROP_EMACS = "setproperty PRETTYPRINT_ADD_EMACS_PROPERTIES TRUE",
    PROP_DECLS = "setproperty PRETTYPRINT_ALL_DECLARATIONS TRUE",
    PROP_COMMS = "setproperty PRETTYPRINT_HEADER_COMMENTS TRUE";
  
  /** Starts the tpips process and sets the streams.
   */  
  public void start()
  {
    try
    {
      System.out.println("starting tpips...");
      tpips = Runtime.getRuntime().exec(execute);
      out = new PrintWriter(tpips.getOutputStream());
      in = new BufferedReader
        (new InputStreamReader(tpips.getInputStream()));
      inErr = new BufferedReader
        (new InputStreamReader(tpips.getErrorStream()));
      
      // listen to tpips error
      Thread listener = new Thread(new Listener(inErr,jpips));
      listener.start();
      
      // warns if tpips dies.
      //   Thread watcher = new Thread(new Watcher(tpips));
      //   watcher.start();
      
      if (emacsDisplayer!=null) 
      {
        System.err.println("setting emacs properties...");
        this.sendCommand(PROP_EMACS);
        this.sendCommand(PROP_DECLS);
        this.sendCommand(PROP_COMMS);
      }
    }  
    catch(Exception e)
    {
      System.out.println("Cannot start tpips!");
    }
  }
  
  
  /** Stops the tpips process.
   */  
  public void stop()
  {
    tpips.destroy();
  }
  
  
  /** Sends a quit message to tpips.
   */  
  public void quit()
  {
    out.println(replaceCommand("quit"));
    out.flush();
  }
  
  
  /** Sends an exit message to tpips.
   */  
  public void exit()
  {
    out.println(replaceCommand("exit"));
    out.flush();
  }
  
  
  /** Sends a message to tpips.
   * @param command the message
   * @return the tpips response
   */  
  synchronized public String sendCommand(String command)
  {
    String s = replaceCommand(command);
    if(s == null) return null;
    System.out.println(TO_TPIPS + s);
    out.println(s);
    out.flush();
    return getResponse();
  }
  
  
  /** Parses the messages of tpips.
   * @return the tpips response
   */  
  public String getResponse()
  {
    try
    {
      String result = null;
      String s = in.readLine();
      while(s!=null)
      {
        System.out.println(FROM_TPIPS + s);
        StringTokenizer t = new StringTokenizer(s,SPACE,false);
        if(t.hasMoreTokens())
        {
          String response = t.nextToken();
          if(response.equals(JPIPS))
          {
            response = t.nextToken();
            if(response.equals(DONE)) 
            {
              return result;
            }
            else if(response.equals(DIRECTORY))
            {
              result = t.nextToken(EOL).substring(1);
            }
            else if(response.equals(WORKSPACE))
            {
              result = t.nextToken(EOL).substring(1);
              if(result.equals(NONE)) result = null;
            }
            else if(response.equals(MODULES))
            {
              result = t.nextToken(EOL).substring(1);
            }
            else if(response.equals(RESULT))
            {
              result = t.nextToken(EOL).substring(1);
            }
            else if(response.equals(SHOW))
            {
              String path = t.nextToken(EOL);
              if(path.substring(1,2).equals(DIR))
                path = path.substring(2);
              File f= new File(directory.getAbsolutePath()+path);
              System.out.println(f.getAbsolutePath());
              
              if (graphDisplayer.davinciExtension(f))
                graphDisplayer.display(f,true,true);
              else 
                
                if (graphDisplayer.controlgraphExtension(f))
                emacsDisplayer.graphdisplay(f, true, true);
              else
              {
                if (emacsDisplayer!=null)
                  emacsDisplayer.display(f, true, true);
                else 
                  textDisplayer.display(f,true,true);
              }
            }
            else if(response.equals(BEGIN_ERROR))
            {
              JOptionPane.showMessageDialog(frame,
                                            getText(END_ERROR),error,
                                            JOptionPane.ERROR_MESSAGE);
            }
            else if(response.equals(BEGIN_REQUEST))
            {
              String text = getText(END_REQUEST);
              //System.err.println("question: "+ text);
              String answer = JOptionPane.showInputDialog
                (frame, text, "TPIPS Input", JOptionPane.QUESTION_MESSAGE);
              out.println(answer);
              out.flush();
            }
          }
        }
        s = in.readLine();
      }
    }
    catch(IOException e)
    {
      System.out.println(e);
    }
    return null;            
  }
  
  
  /** Parses the messages of tpips until the tag is reached.
   The tag is expected after a leading JPIPS...
   @param expected the expected tag for the response
   @return the tpips response
   */  
  public String getText(String expected)
  {
    expected = JPIPS + " " + expected;
    try
    {
      String text = "";
      while(true)
      {
        String s = in.readLine();
        if(s != null && !s.equals(""))
        {
          if(s.equals(expected))
            return text;
          else
            text += s;
        }
      }
    }
    catch(IOException e)
    {
      System.out.println(e);
    }
    return null;            
  }
  
  
  /** Modifies a message by replacing a tag by modules names.
   * Modifies a message by replacing a marker by its true value.
   * @param message the message to modify
   * @return the new message
   */  
  public String replaceCommand(String command)
  {
    String realCommand;
    int index;
    index = command.indexOf(MODULE_TAG);
    if(index != -1)
    {
      realCommand = command.substring(0,index);
      Object tab[] = list.getSelectedValues();
      if(tab.length > 0)
      {
        for(int i=0; i<tab.length; i++)
        {
          realCommand = realCommand + (String)tab[i] + SPACE;
        }      
        command
          = realCommand
          + command.substring(index+MODULE_TAG.length());
      }
      else
      {
        JOptionPane.showMessageDialog(frame,moduleMessage,
                                      "Error",JOptionPane.ERROR_MESSAGE);
        return null;
      }
    }
    index = command.indexOf(TAG);
    if(index != -1)
    {
      index = command.indexOf(SPACE);
      String marker = command.substring(TAG.length(),index);
      realCommand = command.substring(index);
      command = replaceTAG(marker) + realCommand;
    }
    return command;
  }
  
  
  /** Parses the messages of tpips until the tag is reached.
   * @param expected the expected tag for the response
   * @return the tpips response
   */  
  public String replaceTAG(String marker)
  {
    for(int i=0; i<optionVector.size(); i++)
    {
      Stateable op = (Stateable) optionVector.elementAt(i);
      Vector state = op.getState();
      if (state != null)
      {
        for(int j=0; j<state.size(); j++)
        {
          PComboBox cob = (PComboBox) state.elementAt(j);
          if(cob.marker.equals(marker))
          {
            int index = cob.getSelectedIndex();
            return (String)cob.vCommand.elementAt(index);
          }
        }
      }
    }
    return null;            
  }
}
