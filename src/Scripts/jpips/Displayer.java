

/** $Id$
  * $Log: Displayer.java,v $
  * Revision 1.1  1998/06/09 07:28:28  didry
  * Initial revision
  *
  */


package JPips;


import java.io.*;
import java.awt.*;
import JPips.Pawt.*;
import java.awt.swing.border.*;


/** A window manager.
  * It manages the displaying of windows on the screen.
  * @author Francois Didry
  */
abstract class Displayer
{


  private int		noWindows;	//number of displayed windows
  private PPanel	panel;		//panel of the displayer
  private PLabel	labelNo;	//


  /** Sets the number of displayed windows to 0.
    * Creates the displayer panel for JPips.
    */
  public Displayer()
    {
      noWindows = 0;
      panel = new PPanel(new BorderLayout());
      panel.setPreferredSize(new Dimension(80,40));
      panel.setBorder(new TitledBorder("Windows"));
      labelNo = new PLabel();
      updatePanel();
      panel.add(labelNo, BorderLayout.CENTER);
    }
  
  
  /** Adds a window to the count and updates the textfield.
    */
  public void incNoWindows()
    {
      noWindows++;
      updatePanel();
    }


  /** Removes a window from the count and updates the textfield.
    */
  public void decNoWindows()
    {
      noWindows--;
      updatePanel();
    }


  /** @return the current number of windows
    */
  public int getNoWindows()
    {
      return noWindows;
    }


  /** @return the panel for JPips
    */
  public PPanel getPanel()
    {
      return panel;
    }


  /** Updates the number of displayed windows in the panel.
    */
  public void updatePanel()
    {
      labelNo.setText(noWindows+"");
    }


  abstract void display(File file, boolean locked, boolean writable);

  abstract void display(String name, String string,
                        boolean locked, boolean writable);


}
