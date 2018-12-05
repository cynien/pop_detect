import tkinter as tk
import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class Gui:
    def __init__(self, master):
        # wave form plot length
        self.plot_len = 80

        # font
        self.labelfont = ('Arial', 10, 'bold')
        self.entryfont = ('Arial', 12, 'bold')
        master.columnconfigure(0, weight=6)
        master.columnconfigure(1, weight=6)

        self.pop_alarm_color_pass = 'gray94'
        self.pop_alarm_color_false = 'gray30'

        self.create_menu(master)
        self.create_ui_body(master)

    def create_menu(self, master):
        """
        create a menu bar
        :param master: tkinter root
        :return: None
        """

        menu = tk.Menu(master)
        master.config(menu=menu)
        filemenu = tk.Menu(menu)
        channelmenu = tk.Menu(menu)
        filtermenu = tk.Menu(menu)
        plotmenu = tk.Menu(menu)

        menu.add_cascade(label="Device", menu=filemenu)
        menu.add_cascade(label="Channel", menu=channelmenu)
        menu.add_cascade(label="Filter", menu=filtermenu)
        menu.add_cascade(label="Curve plot", menu=plotmenu)

    def create_ui_body(self, master):
        """
        create the ui body, include (1)ok time, (2)last ok time, (3)frequency,
         (4)LR pop alarm (5)wave plot
        :param master: tkinter root
        :return: None
        """
        # ok time
        self.entry_no_ng_time = tk.StringVar()
        self.entry_no_ng_time_last = tk.StringVar()
        lab_t1 = tk.Label(master, text="ok time now", font=self.labelfont)
        lab_t1.grid(row=0, column=0, sticky=tk.W, padx=3)
        self.ent_t1 = tk.Entry(master, textvariable=self.entry_no_ng_time,
                               width=8, font=self.entryfont, justify='center')
        self.ent_t1.grid(row=0, column=1)
        lab_t2 = tk.Label(master, text="ok time last", font=self.labelfont)
        lab_t2.grid(row=1, column=0, sticky=tk.W, padx=3)
        self.ent_t2 = tk.Entry(master, textvariable=self.entry_no_ng_time_last,
                               width=8, font=self.entryfont, justify='center')
        self.ent_t2.grid(row=1, column=1)

        # frequency
        self.entry_text_freq_l = tk.StringVar()
        self.entry_text_freq_r = tk.StringVar()
        lab_freq_l = tk.Label(master, text="Freq.Left", font=self.labelfont)
        lab_freq_l.grid(row=2, column=0, sticky=tk.W, padx=3)
        ent_freq_l = tk.Entry(master, textvariable=self.entry_text_freq_l,
                              width=8, font=self.entryfont, justify='center')
        ent_freq_l.grid(row=2, column=1)

        lab_freq_r = tk.Label(master, text="Freq.Right", font=self.labelfont)
        lab_freq_r.grid(row=3, column=0, sticky=tk.W, padx=3)
        ent_freq_r = tk.Entry(master, textvariable=self.entry_text_freq_r,
                              width=8, font=self.entryfont, justify='center' )
        ent_freq_r.grid(row=3, column=1)


        # pop alarm
        self.but_alarm_l = tk.Button(master, text="L", width=7, font=self.labelfont)
        self.but_alarm_l.grid(row=4, column=0)
        self.but_alarm_r = tk.Button(master, text="R", width=7, font=self.labelfont)
        self.but_alarm_r.grid(row=4, column=1)

        # curve plot
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([0, self.plot_len])
        ax.set_ylim([-35000, 35000])
        ax.yaxis.set_ticks(np.arange(-30000, 40000, 10000))
        scale = 1e3                     # KHz
        ticks = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
        ax.yaxis.set_major_formatter(ticks)
        self.line_l, = ax.plot(range(self.plot_len), 'b')
        self.line_r, = ax.plot(range(self.plot_len), 'm')
        self.canvas = FigureCanvasTkAgg(fig, master=master)
        self.canvas.show()
        self.canvas.get_tk_widget().config(width=360, height=200)
        self.canvas.get_tk_widget().grid(row=0, column=2, rowspan=5, padx=5, pady=10)
        # self.x, self.y = self.line1.get_data()


if __name__ == "__main__":
    root = tk.Tk()
    app = Gui(root)
    root.mainloop()
