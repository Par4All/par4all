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

import java.awt.*;
import java.util.*;

import javax.swing.*;

import fr.ensmp.cri.jpips.Pawt.*;

/** A component for JPips which associates a frame to a menu.
 * 
 * @author Francois Didry
 */
public class Option 
  implements JPipsComponent, Stateable
{
  public TPips  tpips;  //tpips instance
  public String  title;  //title of the option 
  public PFrame  frame;  //frame
  public PMenu  menu;  //menu
  public Vector  vector,  //contains the component of the frame
    state,
    v1,  //contains components
    v2;  //contains components
  public String  TRUE = "true",
    FALSE = "false",
    GETPROPERTY = "getproperty";
  
  /** Defines the title, the menu, the frame and the vector of the option.
   * @param title the title of the option
   * @param menu the menu of the option
   * @param frame the frame of the option
   * @param vector the vector of the components of the frame
   */
  public Option(String title, PMenu menu, PFrame frame,
                Vector vector, Vector state)
  {
    this.title = title;
    this.menu = menu;
    this.frame = frame;
    this.vector = vector;
    this.state = state;
  }
  
  
  /** Defines the title, the menu, the frame and the vector of the option.
   * Defines the component to disable/enable in the vectors v1 and v2.
   * @param v1 contains the component to enable/disable
   * @param v2 contains the component to disable/enable
   */
  public Option(String title, PMenu menu, PFrame frame, Vector vector,
                Vector state, Vector v1, Vector v2)
  {
    this.title = title;
    this.menu = menu;
    this.frame = frame;
    this.vector = vector;
    this.state = state;
    this.v1 = v1;
    this.v2 = v2;
  }
  
  /** @return the frame associated to the Option
   */
  public Component getComponent()
  {
    return (Component)frame;
  }
  
  
  /** @return the menu associated to the Option
   */
  public PMenu getMenu()
  {
    return menu;
  }
  
  
  /** Enable or disable components of the menu.
   * @param yes true means enable
   */
  public void setState(boolean yes)
  {
    for(int i=0; i<v1.size(); i++)
      ((PMenuItem)v1.elementAt(i)).setEnabled(yes);
    for(int i=0; i<v2.size(); i++)
      ((PMenuItem)v2.elementAt(i)).setEnabled(!yes);
  }
  
  /** @return the state vector.
   */
  public Vector getState()
  {
    return state;
  }
  
  /** Enables or disables the frame and the menu.
   * @param yes true means enable
   */
  public void setActivated(boolean yes)
  {
    if(frame != null)
    {
      for(int i=0; i<vector.size(); i++)
        ((Component)vector.elementAt(i)).setEnabled(yes);
    }
    if(menu != null) menu.setEnabled(yes);
  }
  
  public void reset()
  {
  }
}
