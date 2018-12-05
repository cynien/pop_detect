import numpy as np
import tkinter as tk
import audio_in
import view
import scipy


class Ctrl:
    """pop detector"""
    def __init__(self):
        self.root = tk.Tk()
        self.gui = view.Gui(self.root)
        self.process = audio_in.AudioInProcess()
        self.process.stream.start_stream()
        self.define_update_waveform()
        self.define_update_gui()

    def define_update_waveform(self):
        def ctrl_update_waveform():
            self.gui.line_l.set_ydata(self.process.data[0:2*self.gui.plot_len:2])
            self.gui.line_r.set_ydata(self.process.data[1:2*self.gui.plot_len:2])

            self.gui.canvas.draw()
        self.process.update_waveform = ctrl_update_waveform

    def define_update_gui(self):
        def ctrl_update_gui():
            self.gui.entry_text_freq_l.set(int(self.process.freq_avgs[0]))
            self.gui.entry_text_freq_r.set(int(self.process.freq_avgs[1]))
            self.gui.entry_no_ng_time.set('{:6.1f}'.format(self.process.no_ng_time))
            self.gui.entry_no_ng_time_last.set('{:6.1f}'.format(self.process.no_ng_time_last))
            self.set_pop_alarm_bg()
            self.root.update()

        self.process.update_gui = ctrl_update_gui

    def set_pop_alarm_bg(self):
        if self.process.pop_test_pass:
            self.gui.but_alarm_l.config(bg=self.gui.pop_alarm_color_pass)
            self.gui.but_alarm_r.config(bg=self.gui.pop_alarm_color_pass)
        else:
            self.gui.but_alarm_l.config(bg=self.gui.pop_alarm_color_false)
            self.gui.but_alarm_r.config(bg=self.gui.pop_alarm_color_false)


if __name__ == "__main__":
    app = Ctrl()
    app.root.mainloop()

