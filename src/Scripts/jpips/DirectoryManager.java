/*
  $Id$
 
  $Log: DirectoryManager.java,v $
  Revision 1.4  1998/10/16 17:16:20  coelho
  nope.

  Revision 1.3  1998/10/16 14:55:44  coelho
  chooser fixed to 1.2b4.
 
  Revision 1.2  1998/07/01 07:02:57  coelho
  cleaner (wrt my standards).
 
  Revision 1.1  1998/06/30 17:35:33  coelho
  Initial revision
*/

package JPips;

import java.io.*;

import java.awt.*;
import java.awt.event.*;

import com.sun.java.swing.*;
//import com.sun.java.swing.preview.*;
import com.sun.java.swing.border.*;

import JPips.Pawt.*;

/** A manager for the current directory of JPips 
  * 
  * @author Francois Didry
  */  
public class DirectoryManager implements JPipsComponent
{
  public	TPips		tpips;		//tpips instance
  public	File		directory;	//current directory
  public	PPanel		panel;		//jpips directory panel
  public	PTextField	tf;		//textfield in the panel
  public	PButton		browse;		//button in the panel
  public	PFrame		frame;		//frame of jpips

  /** Creates and opens the root directory file.
    * Builds the panel for JPips.
    */  
  public DirectoryManager(PFrame frame, TPips tpips)
  {
    this.frame = frame;
    this.tpips = tpips;
    //this.directory = new File(".");
    buildPanel();
    check();
  }

  /** @return the current directory.
    */  
  public String getDirectory()
  {
    return directory.getAbsolutePath();
  }
  
  /** Creates the directory panel for jpips.
    */  
  public void buildPanel()
  {
    panel = new PPanel(new GridBagLayout());
    GridBagConstraints c = new GridBagConstraints();
    c.insets = new Insets(0,0,0,0);
    c.gridwidth = 1; c.gridheight = 1;
    c.ipadx = 1; c.ipady = 1;
    c.anchor = GridBagConstraints.WEST;
    panel.setBorder(new TitledBorder("Current directory"));
    tf = new PTextField();
    tf.setFont(new Font("Monospaced", Font.PLAIN, 12));
    ActionListener a = new ActionListener() {
      public void actionPerformed(ActionEvent e)
      {
	PTextField tf = (PTextField)e.getSource();
	tf.setFont(new Font("Monospaced", Font.PLAIN, 12));
	File f = new File(tf.getText());
	if(f.isDirectory())
	{
	  directory = f;
	  open(directory);
	}
	else
	{
	  JOptionPane.showMessageDialog
	  (frame, "Invalid directory","Error",JOptionPane.ERROR_MESSAGE);
	  tf.setText(directory.getAbsolutePath());
	}
      }
    };
    tf.addActionListener(a);
    tf.setPreferredSize(new Dimension(500,20));
    c.gridx = 0; c.gridy = 0;
    c.weightx = 1.0; c.weighty = 0.0;
    c.fill = GridBagConstraints.HORIZONTAL;
    panel.add(tf,c);
    browse = new PButton("cd ...");
    a = new ActionListener()
    {
      public void actionPerformed(ActionEvent e) { choose(); }
    };
    browse.addActionListener(a);
    c.gridx = 1; c.gridy = 0;
    c.weightx = 0.0; c.weightx = 0.0;
    c.fill = GridBagConstraints.NONE;
    panel.add(browse,c);
  }

  /** @return the directory panel for JPips
    */  
  public Component getComponent()
  {
    return (Component)panel;
  }
  
  /** Opens the specified directory in tpips.
    * Updates the textfield.
    * @param f the directory file
    */  
  public void open(File f)
  {
    directory = f;
    tpips.sendCommand("cd " + directory.getAbsolutePath());
    check();
  }

  /** Checks if the tpips and the jpips directory are similar.
    */
  public void check()
  {
    String s = tpips.sendCommand("info directory");
    directory = new File(s);
    tf.setText(directory.getAbsolutePath());
    tpips.directory = directory;
  }
  
  /** Displays a file chooser to select a directory.
    */
  public void choose()
  {
    JFileChooser chooser = new JFileChooser(directory.getAbsolutePath());
    if(chooser.showOpenDialog(frame) == 0)
    {
      File f = chooser.getSelectedFile();
      if(f == null || !f.isDirectory())
	    f = chooser.getCurrentDirectory();
      open(f);
    }
    frame.pack();
  }
  
  public PMenu getMenu()
  {
    return null;
  }
  
  public void setActivated(boolean yes)
  {
    tf.setEnabled(yes);
    browse.setEnabled(yes);      
  }

  public void reset()
  {
    check();
  }
}
