"""Non-intrusive Tk tooltips. No blocking dialogs, delayed display and auto-hide."""
from __future__ import annotations
import tkinter as tk

class ToolTip:
    def __init__(self, widget, text:str, delay:int=650):
        self.widget=widget
        self.text=text
        self.delay=delay
        self.pending=None
        self.window=None
        widget.bind('<Enter>',self._schedule,add='+')
        widget.bind('<Leave>',self._hide,add='+')
        widget.bind('<ButtonPress>',self._hide,add='+')

    def _schedule(self,_event=None):
        self._hide()
        try:self.pending=self.widget.after(self.delay,self._show)
        except tk.TclError:pass

    def _show(self):
        self.pending=None
        try:
            x=self.widget.winfo_pointerx()+14
            y=self.widget.winfo_pointery()+18
            window=tk.Toplevel(self.widget)
            window.wm_overrideredirect(True)
            window.wm_geometry(f'+{x}+{y}')
            tk.Label(window,text=self.text,wraplength=310,justify='left',
                     bg='#173954',fg='white',font=('Segoe UI',9),
                     relief='solid',borderwidth=1,padx=10,pady=7).pack()
            self.window=window
        except tk.TclError:pass

    def _hide(self,_event=None):
        if self.pending is not None:
            try:self.widget.after_cancel(self.pending)
            except tk.TclError:pass
            self.pending=None
        if self.window:
            try:self.window.destroy()
            except tk.TclError:pass
            self.window=None
