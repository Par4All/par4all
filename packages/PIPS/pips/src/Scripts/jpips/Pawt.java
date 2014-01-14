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

import java.util.*;
import java.io.*;

import java.awt.*;
import java.awt.event.*;

import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;

/**
 A graphical package for JPips
 All components extends from swing components.
 Some components contain informations to manage their Events.
 
 @author Francois Didry
 */
public abstract class Pawt
{
  /** Default display colors.
   */
  static public Color text_bg = Color.white;
  static public Color text_fg = Color.black;
  static public Color comp_bg = new Color(200,200,255);
  static public Color comp_fg = Color.blue;
  static public Color comp_fgr = Color.red;
  static public Color comp_fgg = Color.black;
  
  static class PComponent extends javax.swing.JComponent
  { 
    public String checking; 
    PComponent() 
    { 
      super(); 
      
      setBackground(comp_bg);
      setForeground(comp_fg);
    } 
  }
  
  static class PFrame extends javax.swing.JFrame
  { 
    public Vector optionVector;
    PFrame(String name)
    {
      super(name); 
      
      setBackground(comp_bg);
      setForeground(comp_fg);
      
      Container pane = getContentPane();
      pane.setBackground(comp_bg);
      pane.setForeground(comp_fg);
    }
    
    /** Disables or enables the Option objects of JPips.
     @param yes true means disable
     */
    public void lock(boolean yes)
    {
      for(int i=1; i<optionVector.size(); i++)
        ((Activatable) optionVector.elementAt(i)).setActivated(!yes);
    }
  }
  
  static class PDialog extends javax.swing.JDialog
  { 
    PDialog(Frame f, String s, boolean b)
    { 
      super(f, s, b); 
      
      setBackground(comp_bg);
      setForeground(comp_fg);
      
      Container pane = getContentPane();
      pane.setBackground(comp_bg);
      pane.setForeground(comp_fg);
    }
  }
  
  static class PPanel extends javax.swing.JPanel
  {
    PPanel(LayoutManager l)
    {
      super(l); 
      
      setBackground(comp_bg);
      setForeground(comp_fg);
    }
    
    PPanel()
    { 
      this(null); 
    }
  }
  
  static class PMenuBar extends javax.swing.JMenuBar
  { 
    PMenuBar()
    { 
      super(); 
      
      setBackground(comp_bg);
      //    setForeground(comp_fg);
    } 
  }
  
  static class PMenu extends javax.swing.JMenu
  { 
    PMenu(String name)
    { 
      super(name); 
      
      setBackground(comp_bg);
      setForeground(comp_fg);
    }
    
    PMenu(){ this(null); }
  }
  
  static  class PMenuItem extends javax.swing.JMenuItem
  {
    public String command;
    public PTextField tf;
    
    PMenuItem(String name,String command,PTextField tf)
    { 
      super(name); 
      
      this.command = command; 
      this.tf = tf; 
      
      setBackground(comp_bg);
      setForeground(comp_fg);
    }
    
    PMenuItem(String name)
    {
      this(name, null, null);
    }
    
    PMenuItem(String name,String command)
    { 
      this(name, command, null);
    }
  }
  
  static class PCheckBox 
    extends javax.swing.JCheckBox
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
      
      setBackground(comp_bg);
      setForeground(comp_fg);
    } 
    
    PCheckBox(String name)
    { this(name, null, null, null, null, null); }
    
    PCheckBox(String name, String command, String checking, String tip)
    { this(name, command, checking, tip, null, null); }
    
    PCheckBox(String name, PTextFrame frame)
    { this(name, null, null, null, null, frame); }
  }
  
  static class PCheckBoxMenuItem extends javax.swing.JCheckBoxMenuItem
  {
    public String command, checking;
    public PCheckBox cb;
    
    /** main constructor. */
    PCheckBoxMenuItem(String name, String command, 
                      String checking, PCheckBox cb)
    {
      super(name); 
      
      this.command = command; 
      this.checking = checking;
      this.cb = cb; 
      
      setBackground(comp_bg);
      setForeground(comp_fg);
    }
    
    PCheckBoxMenuItem(String name)
    {
      this(name, null, null, null);
    }
    
    PCheckBoxMenuItem(String name, String command, String checking)
    { 
      this(name, command, checking, null);
    }
  }
  
  static class PRadioButtonMenuItem 
    extends javax.swing.JRadioButtonMenuItem
  {
    public String command, checking;
    public PLabel label;
    public PComboBox cob;
    public Object o;
    public PRadioButtonMenuItem rbmi;
    
    /** main constructor. */
    PRadioButtonMenuItem(String name, 
                         String command, PLabel label,
                         PComboBox cob, PRadioButtonMenuItem rbmi,
                         Object o, String checking)
    { 
      super(name); 
      
      this.command = command; 
      this.label = label;
      this.cob = cob;
      this.rbmi = rbmi;
      this.o = o;
      this.checking = checking;
      
      setBackground(comp_bg);
      //    setForeground(comp_fg);
    }
    
    PRadioButtonMenuItem(String name)
    { 
      this(name, null, null, null, null, null, null); 
    }
    
    PRadioButtonMenuItem(String name, String command)
    {
      this(name, command, null, null, null, null, null);
    }
    
    PRadioButtonMenuItem(String name, 
                         String command, PLabel label,
                         PRadioButtonMenuItem rbmi)
    {
      this(name, command, label, null, rbmi, null, null);
    }
    
    PRadioButtonMenuItem(String name, String command, PLabel label,
                         PRadioButtonMenuItem rbmi, String checking)
    { 
      this(name, command, label, null, rbmi, null, checking);
    }
    
    PRadioButtonMenuItem(String name, String command,
                         PComboBox cob, Object o)
    { 
      this(name, command, null, cob, null, o, null);
    }
    
    PRadioButtonMenuItem(String name, String command,
                         PComboBox cob, Object o, String checking)
    { 
      this(name, command, null, cob, null, o, checking);
    }
  }
  
  /** create a border with a title.
   @param name the border title.
   @return the titled border.
   */
  static public TitledBorder createTitledBorder(String name)
  {
    TitledBorder t = BorderFactory.createTitledBorder(name);
    t.setTitleColor(comp_fgg);
    return t;
  }
  
  static class PSeparator 
    extends javax.swing.JSeparator
  { 
    PSeparator()
    {
      super(); 
      
      setBackground(comp_bg);
      //    setForeground(comp_fg);
    } 
  }
  
  static class PButton 
    extends javax.swing.JButton
  {
    public String  command;
    public PTextField  tf;
    public PTextFrame  frame;
    
    PButton(String name, String command, String tip, 
            PTextField tf, PTextFrame frame)
    { 
      super(name); 
      
      setMargin(new Insets(0,0,0,0));
      if (tip!=null) setToolTipText(tip);
      
      this.command = command;
      this.tf = tf; 
      this.frame = frame;
      
      setBackground(comp_bg);
      setForeground(comp_fg);
    }
    
    PButton(String name) 
    { this(name, null, null, null, null); }
    PButton(String name, PTextFrame frame) 
    { this(name, null, null, null, frame); }
    PButton(String name, String command) 
    { this(name, command, null, null, null); }
    PButton(String name, String command, String tip)
    { this(name, command, tip, null, null); }
  }
  
  static class PLabel 
    extends javax.swing.JLabel
  {
    PLabel(String s, String tip)
    { 
      super(s); 
      if (tip!=null) setToolTipText(tip);
      
      setBackground(comp_bg);
      //     setForeground(comp_fg);
    }
    
    PLabel(String s) { this(s, null); }
    PLabel() { this(null, null); }
  }
  
  static class PTextField extends javax.swing.JTextField
  {
    PTextField(String s)
    {
      super(s); 
      
      setBackground(text_bg); 
      setForeground(text_fg);
    }
    
    PTextField()
    { 
      this(""); 
    }
  }
  
  static class PList extends javax.swing.JList
  {
    PList() { super(); }
    PList(Vector v) { super(v); }
    PList(DefaultListModel l) { super(l); }
  }
  
  /** A PComboBox can send directly its command on selections,
   * or wait for some specific action to do the job (e.g. a button pressed).
   */
  static class PComboBox extends javax.swing.JComboBox
  {
    public String checking, marker;
    public boolean direct;
    public Vector vCommand, vRbmi, vChecking;
    
    PComboBox(String ch, String ma, boolean direct)
    { 
      super(); 
      
      checking = ch; 
      marker = ma; 
      this.direct = direct;
      
      vCommand = new Vector();
      vRbmi = new Vector();
      vChecking = new Vector();
      
      setBackground(comp_bg);
      //   setForeground(comp_fg);
    }
    
    PComboBox()
    { 
      this(null, null, true); 
    }
    
    PComboBox(String ch)
    {
      this(ch, null, true); 
    }
    
    PComboBox(String ch, String ma) 
    { 
      this(ch, ma, true); 
    }
  }
  
  static class PScrollPanel extends javax.swing.JScrollPane
  {
    PScrollPanel(Component c)
    {
      super(c); 
      
      setBackground(comp_bg);
      setForeground(comp_fg);      
    } 
    
    PScrollPanel()
    {
      this(null);
    } 
  }
  
  static class PButtonGroup extends javax.swing.ButtonGroup
  {
    public String checking;
    
    PButtonGroup(String ch)
    { 
      super(); 
      checking = ch;     
    }
    
    PButtonGroup()
    {
      this(null);
    }
  }
  
  static class PTextArea extends java.awt.TextArea
  {
    PTextArea(String s)
    { 
      super(s);  
      
      setBackground(text_bg);
      setForeground(text_fg);
    }
    
    PTextArea()
    { 
      this("");
    }
  }
  
  static class PTextFrame extends javax.swing.JFrame
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
      cb.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent e) {
          PCheckBox check = (PCheckBox)e.getSource();
          check.frame.locked = check.isSelected();
        }
      });
      p.add(cb);
      b = new PButton("Hide",this);
      b.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent e) {
          PButton button = (PButton)e.getSource();
          button.frame.setVisible(false);
        }
      });
      p.add(b);
      b = new PButton("Close",this);
      b.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent e) {
          PButton button = (PButton)e.getSource();
          button.frame.dispose();
        }
      });
      p.add(b);
      getContentPane().add(p, BorderLayout.SOUTH);
      
      //jpips button
      panelButton = new PButton(name,this);
      panelButton.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent e) {
          PButton button = (PButton)e.getSource();
          button.frame.setVisible(true);
          button.frame.toFront();
        }
      });
      
      pack();
      setSize(new Dimension(600,400));
    }
  }
  
  static public JFileChooser createFileChooser(File dir)
  {
    JFileChooser chooser = new JFileChooser(dir);
    chooser.setBackground(comp_bg);
    chooser.setForeground(comp_fg);
    return chooser;
  }
  
  static public JFileChooser createFileChooser(String dir)
  {
    return createFileChooser(new File(dir));
  }
}
