/*
 * $Id$
 * 
 * $Log: Pawt.java,v $
 * Revision 1.3  1998/07/03 08:13:33  coelho
 * simpler constructors. tips added to labels and buttons.
 *
 * Revision 1.2  1998/06/30 17:35:33  coelho
 * abstarct wt for jpips.
 *
 * Revision 1.1  1998/06/30 16:40:14  coelho
 * Initial revision
 *
 */

package JPips;

import java.util.*;
import java.awt.*;
import java.awt.swing.*;
import java.awt.event.*;
import java.awt.swing.event.*;

/** A graphical package for JPips
  * All components extends from swing components.
  * Some components contain informations to manage their Events.
  * 
  * @author Francois Didry
  */
interface Pawt
{
  class PComponent extends java.awt.swing.JComponent
  { 
    public String checking; 
    PComponent() { super(); } 
  }

  class PFrame extends java.awt.swing.JFrame
    { 
      public Vector optionVector;
      PFrame(String name){ super(name); }
      
      /** Disables or enables the Option objects of JPips.
       * @param yes true means disable
       */
      public void lock(boolean yes)
        {
          for(int i=1; i<optionVector.size(); i++)
	      ((Activatable) optionVector.elementAt(i)).setActivated(!yes);
        }
    }

  class PDialog extends java.awt.swing.JDialog
    { PDialog(Frame f, String s, boolean b){ super(f, s, b); } }

  class PPanel extends java.awt.swing.JPanel
    {
      PPanel(){ super(); }
      PPanel(LayoutManager l){ super(l); }
    }

  class PMenuBar extends java.awt.swing.JMenuBar
    { PMenuBar(){ super(); } }

  class PMenu extends java.awt.swing.JMenu
    { PMenu(){ super(); }
      PMenu(String name){ super(name); }
    }

  class PMenuItem extends java.awt.swing.JMenuItem
    {
      public String command;
      public PTextField tf;
      PMenuItem(String name){ super(name); }
      PMenuItem(String name,String command)
        { super(name); this.command = command; }
      PMenuItem(String name,String command,PTextField tf)
        { super(name); this.command = command; this.tf = tf; }
    }

  class PCheckBox 
    extends java.awt.swing.JCheckBox
    {
      public String command,checking;
      public PCheckBoxMenuItem cbmi;
      public PTextFrame frame;
      
      PCheckBox(String name, String command, String checking, String tip, 
		PCheckBoxMenuItem cbmi, PTextFrame frame)
      { 
	super(name); 
	if (tip!=null) setToolTipText(tip);
	this.command = command; 
	this.checking = checking;
	this.cbmi = cbmi; 
	this.frame = frame;
      }	

      PCheckBox(String name)
      { this(name, null, null, null, null, null); }
      
      PCheckBox(String name, String command, String checking, String tip)
      { this(name, command, checking, tip, null, null); }

      PCheckBox(String name, PTextFrame frame)
      { this(name, null, null, null, null, frame); }
    }

  class PCheckBoxMenuItem extends java.awt.swing.JCheckBoxMenuItem
    {
      public String command,checking;
      public PCheckBox cb;

      PCheckBoxMenuItem(String name){ super(name); }

      PCheckBoxMenuItem(String name,String command,String checking)
      { 
	super(name);
	this.command = command;
	this.checking = checking; 
      }

      PCheckBoxMenuItem(String name, String command, 
			String checking, PCheckBox cb)
      {
	super(name); 
	this.command = command; 
	this.checking = checking;
	this.cb = cb; 
      }
    }

  class PRadioButtonMenuItem extends java.awt.swing.JRadioButtonMenuItem
    {
      public String command,
                    checking;
      public PLabel label;
      public PComboBox cob;
      public Object o;
      public PRadioButtonMenuItem rbmi;

      PRadioButtonMenuItem(String name){ super(name); }
      PRadioButtonMenuItem(String name, String command)
        { super(name); this.command = command; }
      PRadioButtonMenuItem(String name, String command, PLabel label,
                           PRadioButtonMenuItem rbmi)
        { super(name); this.command = command; this.label = label;
	  this.rbmi = rbmi; }
      PRadioButtonMenuItem(String name, String command, PLabel label,
                           PRadioButtonMenuItem rbmi, String checking)
        { super(name); this.command = command; this.label = label;
	  this.rbmi = rbmi; this.checking = checking; }
      PRadioButtonMenuItem(String name, String command,
                           PComboBox cob, Object o)
        { super(name); this.command = command; this.cob = cob; this.o = o; }
      PRadioButtonMenuItem(String name, String command,
                           PComboBox cob, Object o, String checking)
        { super(name); this.command = command; this.cob = cob; this.o = o;
	  this.checking = checking; }
    }


  class PSeparator 
    extends java.awt.swing.JSeparator
  { 
    PSeparator(){ super(); } 
  }

  class PButton 
    extends java.awt.swing.JButton
  {
    public String 	command;
    public PTextField 	tf;
    public PTextFrame 	frame;

    PButton(String name, String command, String tip, 
	    PTextField tf, PTextFrame frame)
      { 
	super(name); 
	setMargin(new Insets(0,0,0,0));
	if (tip!=null) setToolTipText(tip);
	this.command = command;
	this.tf = tf; 
	this.frame = frame;
      }

    PButton(String name) 
      { this(name, null, null, null, null); }
    PButton(String name, PTextFrame frame) 
      { this(name, null, null, null, frame); }
    PButton(String name, String command) 
      { this(name, command, null, null, null); }
    PButton(String name, String command, String tip)
      {	this(name, command, tip, null, null); }
  }

  class PLabel 
    extends java.awt.swing.JLabel
  {
    PLabel(String s, String tip)
      { 
	super(s); 
	if (tip!=null) setToolTipText(tip);
      }

    PLabel(String s) { this(s, null); }
    PLabel() { super(); }
  }
    
  class PTextField extends java.awt.swing.JTextField
    {
      PTextField(){ super(); }
      PTextField(String s){ super(s); }
    }
    
  class PList extends java.awt.swing.JList
    {
      PList(){ super(); }
      PList(Vector v){ super(v); }
      PList(DefaultListModel l){ super(l); }
    }

    
  class PComboBox extends java.awt.swing.JComboBox
    {
      public String checking,
      		    marker;
      public Vector vCommand = new Vector(),
                    vRbmi = new Vector(),
		    vChecking = new Vector();
      PComboBox(){ super(); }
      PComboBox(String ch){ super(); checking = ch; }
      PComboBox(String ch, String ma){ super(); checking = ch; marker = ma; }
    }


  class PScrollPanel extends java.awt.swing.JScrollPane
    {
      PScrollPanel(){ super(); } 
      PScrollPanel(Component c){ super(c); } 
    }
        
        
  class PButtonGroup extends java.awt.swing.ButtonGroup
    {
      String checking;
      PButtonGroup(String ch){ super(); checking = ch; }
    }
        
        
  class PTextArea extends java.awt.TextArea
    {
      PTextArea(){ super(); }
      PTextArea(String s){ super(s); }
    }
        
        
  class PTextFrame extends java.awt.swing.JFrame
    {
      boolean locked;
      boolean writable;
      PTextArea ta;
      PButton panelButton;
            
      PTextFrame(String name, String text, boolean locked, boolean writable)
        {
	  //window
	  super(name);
	  this.locked = locked;
	  this.writable = writable;
          PButton b;
	  getContentPane().setLayout(new BorderLayout());
	  ta = new PTextArea(text);
          ta.setFont(new Font("Monospaced", Font.PLAIN, 12));
          ta.setEnabled(writable);
	  PScrollPanel s = new PScrollPanel((Component)ta);
	  getContentPane().add(s, BorderLayout.CENTER);
	  
	  PPanel p = new PPanel(new GridLayout(1,3));
	  PCheckBox cb = new PCheckBox("Locked",this);
	  cb.setSelected(locked);
	  cb.addActionListener(new ActionListener()
            {
	      public void actionPerformed(ActionEvent e)
	        {
		  PCheckBox check = (PCheckBox)e.getSource();
		  check.frame.locked = check.isSelected();
		}
	    });
	  p.add(cb);
	  b = new PButton("Hide",this);
	  b.addActionListener(new ActionListener()
            {
	      public void actionPerformed(ActionEvent e)
	        {
		  PButton button = (PButton)e.getSource();
		  button.frame.setVisible(false);
		}
	    });
	  p.add(b);
	  b = new PButton("Close",this);
	  b.addActionListener(new ActionListener()
            {
	      public void actionPerformed(ActionEvent e)
	        {
		  PButton button = (PButton)e.getSource();
		  button.frame.dispose();
		}
	    });
	  p.add(b);
	  getContentPane().add(p, BorderLayout.SOUTH);
	  
	  //jpips button
	  panelButton = new PButton(name,this);
	  panelButton.addActionListener(new ActionListener()
            {
	      public void actionPerformed(ActionEvent e)
	        {
		  PButton button = (PButton)e.getSource();
		  button.frame.setVisible(true);
		  button.frame.toFront();
		}
	    });

	  pack();
	  setSize(new Dimension(600,400));
	}
    }
}
