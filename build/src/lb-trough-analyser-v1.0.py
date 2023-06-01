# Creative Commons Attribution (CC-BY) 4.0 2023 Domantas Narkevičius. Some rights reserved.
#
# For help or bug reports, please contact me via Github: https://github.com/dnarkevi/
#
# This program can be used or modified in any way you want, but
# it goes under the terms of the Creative Commons Attribution 4.0, 
# which means you must reference the author. See https://creativecommons.org/licenses
# for more information.

import os
import numpy as np
from traceback import format_exc

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.constants import Avogadro

from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)

import PySimpleGUI as sg
from tkinter import (
    Frame as tkFrame,
    filedialog as tkfiledialog
)


# Abbreviation explanation:
# piar - area vs surace pressure (pi)
# cmsp - compression modulus vs surface pressure (pi)
# dgxi - delta Gibbs energy vs greek letter chi
# fg - foreground
# bg - background

# Notes:
# * Color #000001 is sometimes selected instead of pure black due to a bug in
# PySimpleGUI API.


# =============================================================================
# Default global values ant attributes.
# =============================================================================

TEXT_COLOR = 'white'
BG_COLOR = '#52524E'
rcParams.update({'font.size': 14})
COLORMAP = ["#202020",
            "#E03c3c",
            "#3cE03c",
            "#3c3cE0",
            "#3cE0E0",
            "#E03cE0",
            "#E0E03c"]
# matplotlib.use("TkAgg")
sg.theme('DarkGrey4')


BOX_W = 449
BOX_H = BOX_W / 1.618 + 40
FIG_SZ = (BOX_W, BOX_W / 1.618)
FIG_DPI = 65


# =============================================================================
# Functions for changing default attributes and for simplifying widget usage.
# =============================================================================

def Button(*args, s=(7, None), **kwargs):
    return sg.Button(*args, **kwargs, size=s, pad=(5, 5))


def Listbox(*args, **kwargs):
    return sg.Listbox('', *args, **kwargs,
                      select_mode='LISTBOX_SELECT_MODE_SINGLE',
                      size=(14, None), text_color='#000001', pad=(5, 5))


def Input(*args, **kwargs):
    return sg.Input(*args, **kwargs, size=(5, None),
                    text_color='#000001', pad=(5, 5))


def Frame(*args, pad=(5, 5), **kwargs):
    return sg.Frame(*args, **kwargs, pad=pad)


def Column(*args, **kwargs):
    return sg.Column(*args, **kwargs, pad=(5, 5))


def VSeparator():
    return sg.VSeparator(color=BG_COLOR, pad=(2, 2))


def HSeparator():
    return sg.HSeparator(color=BG_COLOR, pad=(2, 2))


def prep_input(key, min_val=-np.inf, max_val=np.inf, integer=False):
    """
    Reads input value from PySimpleGUI widget. Converts to number if
    possible. Bounds limit return value. Input widget is updated by default.

    """
    if integer:
        try:
            val = int(float(values[key]))
        except ValueError:
            val = 0
    else:
        try:
            val = float(values[key])
        except ValueError:
            val = 0
    if val < min_val:
        val = min_val
    if val > max_val:
        val = max_val
    window[key].update(val)
    return val


# =============================================================================
# Helper functions.
# =============================================================================

def unique_sort(arr1, arr2):
    """
    Arrays arr1 and arr2 must be the same size. Removes repeating
    elements in arr1 and values from arr2 by corresponding indices.
    Also sorts arr1 and arr2 by arr1.

    """
    arr1 = np.array(arr1, dtype=float)
    arr2 = np.array(arr2, dtype=float)

    arr1, unique_idx = np.unique(arr1, return_index=True)
    arr2 = arr2[unique_idx]

    # Sorting everything by increasing value.
    sorted_idx = arr1.argsort()
    arr1 = arr1[sorted_idx]
    arr2 = arr2[sorted_idx]
    return arr1, arr2


def sort(lst1, lst2):
    """
    Lists lst1 and lst2 must be the same size.
    Sorts lst1 and lst2 by lst1.

    """
    zipped_lists = zip(lst1, lst2)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    return [list(tpl) for tpl in tuples]


def assign_phase(y):
    """Returns phase transition type based on compression modulus value"""
    if y < 12.5:
        return "Undefined"
    elif y < 50:
        return "Liquid-expanded"
    elif y < 100:
        return "Liquid"
    elif y < 250:
        return "Liquid-condensed"
    elif y < 1000:
        return "Condensed"
    else:
        return "Solid"


def str2normalized(string):
    """Returns first available number from string between 0 and 1."""
    string = string.replace(',', '.')
    delimiterFound = False
    digitsStarted = False
    start = None
    for i, s in enumerate(string):
        if digitsStarted:
            if s.isdigit():
                pass
            elif (not delimiterFound) and s == '.':
                delimiterFound = True
            else:
                break
        elif s.isdigit():
            digitsStarted = True
            start = i

    if s.isdigit():
        i += 1  # To include final integer.

    if start is None:
        return 0.5  # In case there are no numbers in the string.
    else:
        number = float(string[start:i])

    if number > 1:  # Ignoring negative, because loop does not capture minus.
        return 1.0
    else:
        return number


def export_columns(columns, head, path):
    """
    Saves double-nested lists (matrix) to a file. Matrix can be irregular,
    exported data will be empty strings.

    """
    file = open(path, 'w')
    file.write(head + "\n")
    maxlen = max(map(len, columns))
    for i in range(maxlen):
        row = []
        for col in columns:
            if i < len(col):
                try:  # Converting numbers to str if they are numbers.
                    row.append("%.4f" % col[i])
                except TypeError:
                    row.append(col[i])
            else:
                row.append('')
        file.write("\t".join(row) + "\n")
    file.close()


# =============================================================================
# Functions and classes related to plotting and data operations.
# =============================================================================

def set_plot_color(fig, bg_color, fg_color):
    """Sets colors for the plot frame, axes, etc."""
    ax, = fig.axes
    # Facecolor
    fig.patch.set_facecolor(bg_color)
    # Plot area
    ax.patch.set_facecolor('white')
    # Outline
    for spine in ax.spines.values():
        spine.set_edgecolor(fg_color)
    # Title
    ax.set_title(label=None, color=fg_color)
    # Axes labels
    ax.yaxis.label.set_color(fg_color)
    ax.xaxis.label.set_color(fg_color)
    # Axes tick and ticklabel
    ax.tick_params(color=fg_color, labelcolor=fg_color)


class Toolbar(NavigationToolbar2Tk):
    """Modified matplotlib toolbar."""

    def __init__(self, canvas, window, crd_disp=False, toolitems=(0, 4, 5, 8)):
        self.toolitems = [NavigationToolbar2Tk.toolitems[c] for c in toolitems]
        self.crd_disp = crd_disp
        tkFrame.__init__(self)
        super().tk_setPalette(BG_COLOR)
        super().__init__(canvas, window)

    def set_message(self, msg):
        if self.crd_disp:
            super().set_message(msg)
        else:
            pass

    # Re-ordering buttons.
    def _Button(self, text, image_file, toggle, command):
        btn = super()._Button(text, image_file, toggle, command)
        btn.pack(side='right')
        return btn

    def save_figure(self, *args):
        filetypes = (
            ('Portable Network Graphics', '*.png'),
            ('Scalable Vector Graphics', '*.svg'),
            ('Joint Photographic Experts Group', '*.jpeg'),
            ('Tagged Image File Format', '*.tiff')
        )

        fname = tkfiledialog.asksaveasfilename(
            title='Save the figure',
            filetypes=filetypes,
            defaultextension='',
        )

        if fname in ["", ()]:
            return

        set_plot_color(self.canvas.figure, 'white', 'black')

        if values['-DPI300-']:
            dpi = 300
        elif values['-DPI600-']:
            dpi = 600
        elif values['-DPI1200-']:
            dpi = 1200

        try:
            self.canvas.figure.savefig(fname, dpi=dpi)
        except Exception:
            pass
        set_plot_color(self.canvas.figure, BG_COLOR, TEXT_COLOR)


class Fig:
    """Responsible for plot manipulation in window."""

    def __init__(self, size, key, toolbar_key=None,
                 msg=True, **kwargs):
        """Creates matplotlib figure and axes.
        Draws them into canvas widget."""
        self.fig = Figure(
            figsize=[s / FIG_DPI for s in size],
            dpi=FIG_DPI,
            edgecolor='none',
            layout='constrained'
        )
        self.ax = self.fig.subplots()
        set_plot_color(self.fig, BG_COLOR, TEXT_COLOR)
        self.canvas = FigureCanvasTkAgg(self.fig, window[key].TKCanvas)
        if toolbar_key is not None:
            self.tb = Toolbar(self.canvas, window[toolbar_key].TKCanvas, msg)
            self.tb.update()
        self.canvas.get_tk_widget().pack(side='right',
                                         fill='both', expand=1)
        self.canvas.draw()
        self.labels = None

    def shift_pix(self, original, shift):
        """Returns shifted values by pixel domain in data domain."""
        x, y = self.ax.transData.transform(original)
        x += shift[0]
        y += shift[1]
        return self.ax.transData.inverted().transform((x, y))

    def draw(self, rescale=False):
        """Rescales plots."""
        if rescale:
            self.ax.relim()
            self.ax.autoscale(True)
            self.ax.set(ylim=[None, None],
                        xlim=[None, None])

            x_min = +np.inf
            x_max = -np.inf
            y_min = +np.inf
            y_max = -np.inf

            # Finding minima and maxima of all plotted data.
            for child in self.ax.get_children():
                # The reasoning for "weird" isinstance operation to not import
                # whole matplotlib library.
                if str(type(child)) == "<class 'matplotlib.text.Annotation'>":
                    # Shifting annotation, so it could fit.
                    datamin = self.shift_pix(child.xy, (-60, -20))
                    datamax = self.shift_pix(child.xy, (60, 20))
                elif str(type(child)) == "<class 'matplotlib.lines.Line2D'>":
                    data = child.get_data()
                    datamin = np.min(data, axis=1)
                    datamin = self.shift_pix(datamin, (0, -10))
                    datamax = np.max(data, axis=1)
                    datamax = self.shift_pix(datamax, (0, 10))
                else:
                    continue

                x_min = min(datamin[0], x_min)
                x_max = max(datamax[0], x_max)

                y_min = min(datamin[1], y_min)
                y_max = max(datamax[1], y_max)

            if not any(np.isinf((x_min, x_max))):
                self.ax.set(xlim=[x_min, x_max])

            if not any(np.isinf((y_min, y_max))):
                self.ax.set(ylim=[y_min, y_max])

            if hasattr(self, 'tb'):
                self.tb.update()

        # Check if labels are empty.
        if self.ax.get_legend_handles_labels()[1]:
            self.ax.legend(prop={'size': 12})
        else:
            self.ax.legend('').remove()
        self.canvas.draw()

    def get_new_color(self):
        """Returns the unused color in the plot from the COLORMAP."""
        used_colors = [line.get_color() for line in self.ax.get_lines()]
        for color in COLORMAP:
            if not (color in used_colors):
                return color
        return COLORMAP[0]

    def clear_annotations(self):
        for child in self.ax.get_children():
            if str(type(child)) == "<class 'matplotlib.text.Annotation'>":
                child.remove()

    def clear_scatter(self):
        for child in self.ax.get_children():
            if str(type(child)) == (
                    "<class 'matplotlib.collections.PathCollection'>"):
                child.remove()

    def export(self, path):
        """Filters data points by selected limits in plot and exports."""
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

        labels = []
        data = []

        for line in self.ax.get_lines():
            X = np.array(line.get_xdata())
            Y = np.array(line.get_ydata())

            xidx = np.where(np.logical_and(xmin <= X, X <= xmax))
            X = X[xidx]
            Y = Y[xidx]

            yidx = np.where(np.logical_and(ymin <= Y, Y <= ymax))
            X = X[yidx]
            Y = Y[yidx]

            data.append(X)
            data.append(Y)

            labels.append(line.get_label())
            labels.append("\t\t")  # To have the same lengh as data list.
        if data:
            export_columns(data, ''.join(labels), path)


class Lbdata():
    """
    Main data class for storing read data from files. Contains methods for
    plotting, and calculating various values and attributes.

    """

    def __init__(self, path, key):
        try:
            file = open(path)
        except FileNotFoundError:
            return
        self.path = path
        self.molfrac = str2normalized(key)
        self.key = key
        try:
            lines = []
            for line in file.readlines():
                lines.append(line.split())
            lines = list(filter(None, lines))
            lines.append([])
            file.close()

            data_pi = []
            data_ar = []
            data_temp = []
            for i, line in enumerate(lines):
                if line[0].startswith("Name"):
                    self.name = ' '.join(line[2:])
                if line[0].startswith("t[s]"):
                    a_ixd = line.index("Mma[Å²]")  # Searching for area column.
                    pi_ixd = line.index("P1[mN/m]")  # Pressure column.
                    temp_ixd = line.index("T[°C]")  # Temperature column.
                    rows_cnt = len(line)  # Total count of the columns.
                    i += 1
                    break

            for line in lines[i:]:
                if len(line) == rows_cnt:
                    data_ar.append(line[a_ixd].replace(',', '.'))
                    data_pi.append(line[pi_ixd].replace(',', '.'))
                    data_temp.append(float(line[temp_ixd].replace(',', '.')))
                else:
                    break
        except (ValueError, IndexError):
            sg.popup("File format is wrong.", title="Error")
            return

        self.data_ar, self.data_pi = unique_sort(data_ar, data_pi)
        self.data_ar = self.data_ar / 100  # Converting Å to nm.
        self.temp = np.average(data_temp)
        self.doCmspPlot = False
        self.doDgxiPlot = False
        self.lines = {}

    def calc_cm(self):
        if values['-BDIF-']:
            win_len = prep_input('-BDIF_WLN-', 1, self.data_ar.size, True)
            poly_ord = prep_input('-BDIF_ORD-', 0, win_len - 1, True)

            # ar is being evenly spaced, needed for smoothing.
            ar = np.linspace(self.data_ar.min(),
                             self.data_ar.max(),
                             self.data_ar.size)
            model = interp1d(self.data_ar,
                             self.data_pi,
                             kind='cubic')
            pi = model(ar)
            pi = savgol_filter(pi, win_len, poly_ord)
        else:
            ar = self.data_ar
            pi = self.data_pi

        dydx = np.gradient(pi, ar)
        cm = -ar * dydx
        pi, cm = unique_sort(pi, cm)

        if values['-ADIF-']:
            win_len = prep_input('-ADIF_WLN-', 1, pi.size, True)
            poly_ord = prep_input('-ADIF_ORD-', 0, win_len - 1, True)

            # Pi is being evenly spaced, needed for smoothing.
            pi_cmsp = np.linspace(pi.min(), pi.max(), pi.size)
            itp = interp1d(pi, cm, kind='cubic')
            cm = itp(pi_cmsp)
            cm = savgol_filter(cm, win_len, poly_ord)
        else:
            pi_cmsp = pi

        self.data_pi_cmsp = pi_cmsp
        self.data_cm = cm

    def interpolated_piar(self, x):
        """Prepares interpolated data for area vs pi (not pi vs area)."""
        pi, ar = unique_sort(self.data_pi, self.data_ar)
        model = interp1d(pi,
                         ar,
                         kind='cubic',
                         fill_value='extrapolate')
        return model(x)

    def plot(self, fig, key, update=False, dgxi=None):
        """Plots selected data. Updates it if needed."""
        if key == 'piar':
            x = self.data_ar
            y = self.data_pi
        elif key == 'cmsp':
            x = self.data_pi_cmsp
            y = self.data_cm
        elif key == 'dgxi':
            x, y = dgxi

        if not (key in self.lines):
            self.lines[key], = fig.ax.plot(x, y, color=fig.get_new_color())
        elif update:
            self.lines[key].set_xdata(x)
            self.lines[key].set_ydata(y)

    def annotate(self, fig):
        """Annotates cmsp peak."""
        idx = self.data_cm.argmax()
        x = self.data_pi_cmsp[idx]
        y = self.data_cm[idx]
        text = assign_phase(y)
        fig.ax.annotate(text, (x, y * 1.05), ha="center")
        fig.ax.scatter(x, y, c=self.lines['cmsp'].get_color())

    def delplot(self, *keys):
        """Removes line from plot."""
        if not keys:
            keys = ('piar', 'cmsp', 'dgxi')
        for key in keys:
            if key in self.lines:
                # Checking if line is drawn somethere.
                if self.lines[key]._axes is not None:
                    self.lines[key].remove()
                del self.lines[key]


# =============================================================================
# The layout of the program.
# =============================================================================

first_square = [
    [
         sg.Canvas(k='-CANVAS0BAR-'),
         sg.In(enable_events=True, visible=False, k='-piar_export-'),
         sg.SaveAs(
             "Export", s=(7, None), pad=(5, 5),
             file_types=(('Text documents', '*.txt'), ('ALL Files', '*.*')),
             ),
    ],
    [
         sg.Canvas(s=FIG_SZ, k='-CANVAS0-')
    ]
]

second_square = [
    [
         sg.Canvas(k='-CANVAS1BAR-'),
         sg.In(enable_events=True, visible=False, k='-cmsp_export-'),
         sg.SaveAs(
             "Export", s=(7, None), pad=(5, 5),
             file_types=(('Text documents', '*.txt'), ('ALL Files', '*.*')),
             ),
    ],
    [
         sg.Canvas(s=FIG_SZ, k='-CANVAS1-')
    ]
]

third_square = [
    [
         sg.Canvas(k='-CANVAS2BAR-'),
         sg.In(enable_events=True, visible=False, k='-dgxi_export-'),
         sg.SaveAs(
             "Export", s=(7, None), pad=(5, 5),
             file_types=(('Text documents', '*.txt'), ('ALL Files', '*.*')),
             ),
    ],
    [
         sg.Canvas(s=FIG_SZ, k='-CANVAS2-')
    ]
]


first_row = [
    sg.Column(first_square, s=(BOX_W, BOX_H), element_justification="right"),
    VSeparator(),
    sg.Column(second_square, s=(BOX_W, BOX_H), element_justification="right"),
    VSeparator(),
    sg.Column(third_square, s=(BOX_W, BOX_H), element_justification="right")
]


def Selection_panel(key, delkey, deltext, extra=[]):
    """
    The selection panel is the same for all plots, only the key differs.
    For simplicity reasons, this function was created. Additional widgets
    can be added if needed (extra keyword).

    """
    widget = Frame("Selection of data", [[
        Listbox(enable_events=True, k=key + '_sel-'),
        Column([
            [
                sg.In(enable_events=True, visible=False, k=key + '_color-'),
                sg.ColorChooserButton("Color", s=(7, None), pad=(5, 5)),
                Button(deltext, k=delkey)
            ],
            [
                Frame("Line style", [[
                    Button("Solid", s=(4, None), k=key + '_solid-'),
                    Button("Dotted", s=(5, None), k=key + '_dotted-'),
                    Button("Dashed", s=(6, None), k=key + '_dashed-'),
                    Button("Dashdot", s=(7, None), k=key + '_dashdot-')
                    ]])
            ], extra], vertical_alignment='top'),
        ]])
    return widget


fourth_square = [
    [
         Selection_panel('-piar', '-delfile-', 'Delete',
                         extra=[sg.Text('', k='-piar_temp-')])
    ],
    [
         Frame("Compression modulus vs π plot", [[
             Button("Draw", k='-cmsp_draw-'),
             Button("Draw All", k='-cmsp_dall-')
             ]],),
         Frame("Gibbs energy vs χ plot", [[
             Button("Send", k='-dgxi_send-'),
             Button("Send All", k='-dgxi_sall-')
             ]])
    ],
]

savgol_frame = [
    [
         Column([
             [sg.T("Enable", pad=(5, 5))],
             [sg.T("Window length", pad=(5, 5))],
             [sg.T("Polyorder", pad=(5, 5))],
             ], vertical_alignment='bottom'),
         Column([
             [sg.T("Before diff")],
             [sg.Checkbox('', pad=(5, 5), k='-BDIF-')],
             [Input(100, k='-BDIF_WLN-')],
             [Input(3, k='-BDIF_ORD-')],
             ], element_justification='center'),
         Column([
             [sg.T("After diff")],
             [sg.Checkbox('', pad=(5, 5), k='-ADIF-')],
             [Input(100, k='-ADIF_WLN-')],
             [Input(3, k='-ADIF_ORD-')],
             ], element_justification='center')
    ]
]


fifth_square = [
    [
         Selection_panel('-cmsp', '-cmsp_erase-', 'Erase')
    ],
    [
         Frame("Savgol filter", savgol_frame, pad=(0, 0)),
         Column(
             [[sg.Checkbox('Identify maximum', enable_events=True,
                           pad=(5, 10), k='-cmsp_peak-')]],
             vertical_alignment='top'
             )
    ],
    [
         Button("Apply smoothing", s=(15, None), k='-smooth-')
    ],
]

sixth_square = [
    [
         Selection_panel('-dgxi', '-dgxi_erase-', 'Erase')
    ],
    [
         Frame("Plot properties", [[
             Column([
                 [
                     sg.Text("Component A fraction:"),
                     Input(1.0, k='-dgxi_fracread-'),
                     Button("Set", k='-dgxi_fracset-')
                 ],
                 [
                     sg.Text("π integration boundary:"),
                     Input(1.0, k='-dgxi_piset-'),
                     Button("Draw", k='-dgxi_draw-')
                 ]])
             ]])
    ],
    [
         sg.Text('', s=(50, None), k='-dgxi_msg-')
    ]
]

second_row = [
    sg.Column(fourth_square, s=(BOX_W, 400)),
    VSeparator(),
    sg.Column(fifth_square, s=(BOX_W, 400)),
    VSeparator(),
    sg.Column(sixth_square, s=(BOX_W, 400))
]

layout = [
    [
         sg.In(enable_events=True, visible=False, k='-openfile-'),
         sg.FilesBrowse(button_text="Load files", files_delimiter='*',
                        file_types=(('ALL Files', '*.*'),), pad=(5, 5)),
         sg.Push(),
         sg.Text("Legend:"),
         sg.Radio("Filename", "leg",
                  enable_events=True, k='-leg1-'),
         sg.Radio("Name in file", "leg",
                  enable_events=True, default=True, k='-leg2-'),
         sg.Radio("None", "leg",
                  enable_events=True, k='-leg3-'),
         VSeparator(),
         sg.Text("Image saving resolution:"),
         sg.Radio("300 DPI", "dpi", k='-DPI300-'),
         sg.Radio("600 DPI", "dpi", default=True, k='-DPI600-'),
         sg.Radio("1200 DPI", "dpi", k='-DPI1200-')
    ],
    [
         HSeparator()
    ],
    [
         first_row
    ],
    [
         HSeparator()
    ],
    [
         second_row
    ]
]


# Run the Event Loop
def main():
    global window, event, values, event_history
    data = {}
    while True:
        event, values = window.read()
        event_history.append(event)
        if event == sg.WIN_CLOSED:
            break

        # Events related to first first plot.
        if event == '-openfile-':
            paths = values['-openfile-'].split('*')
            paths = filter(None, paths)
            if not paths:  # Check if empty.
                continue
            for path in paths:
                fname = os.path.splitext(path)[0]
                fname = fname.split('/')[-1]
                if fname in data:  # For duplicate names.
                    for i in range(10):
                        if (fname + str(i)) not in data:
                            fname += str(i)
                            break
                lb_class = Lbdata(path, fname)
                if lb_class is not None:
                    data[fname] = lb_class
                    lb_class.plot(fig1, 'piar')
            fig1.draw(rescale=True)

        if event == '-delfile-':
            if values['-piar_sel-']:  # Check if not empty.
                key, = values['-piar_sel-']
                data[key].delplot()
                del data[key]
                fig1.draw(rescale=True)
                fig2.draw(rescale=True)
                fig3.draw(rescale=True)

        if event in {'-openfile-', '-delfile-'}:
            if data:  # Check if empty.
                window['-piar_sel-'].update(tuple(data.keys()), set_to_index=0)
                key = window['-piar_sel-'].get_list_values()[0]
                window['-piar_temp-'].update(('Temperature: %.1f °C' %
                                              data[key].temp))
            else:
                window['-piar_sel-'].update('', set_to_index=0)
                window['-piar_temp-'].update('')

        if event == '-piar_sel-':
            if values['-piar_sel-']:
                key, = values['-piar_sel-']
                window['-piar_temp-'].update(('Temperature: %.1f °C' %
                                              data[key].temp))
            else:
                window['-piar_temp-'].update('')

        # Events related to second first plot.
        if event == '-cmsp_draw-':
            if values['-piar_sel-']:
                key, = values['-piar_sel-']
                data[key].doCmspPlot = True

        if event == '-cmsp_dall-':
            for lb_class in data.values():
                lb_class.doCmspPlot = True

        if event == '-cmsp_erase-':
            if values['-cmsp_sel-']:
                key, = values['-cmsp_sel-']
                data[key].doCmspPlot = False

        if event in {'-delfile-', '-cmsp_draw-', '-cmsp_dall-',
                     '-cmsp_erase-', '-smooth-', '-cmsp_peak-'}:
            cmsp_keys = []
            fig2.clear_annotations()
            fig2.clear_scatter()
            for key, lb_class in data.items():
                if lb_class.doCmspPlot:
                    lb_class.calc_cm()
                    lb_class.plot(fig2, 'cmsp', update=True)
                    if values['-cmsp_peak-']:
                        lb_class.annotate(fig2)
                    cmsp_keys.append(key)
                else:
                    lb_class.delplot('cmsp')
            fig2.draw(rescale=True)
            window['-cmsp_sel-'].update(cmsp_keys, set_to_index=0)

        # Events related to second third plot.
        if event == '-dgxi_send-':
            if values['-piar_sel-']:  # Check if not empty.
                key, = values['-piar_sel-']
                data[key].doDgxiPlot = True

        if event == '-dgxi_sall-':
            for lb_class in data.values():
                lb_class.doDgxiPlot = True

        if event == '-dgxi_erase-':
            if values['-dgxi_sel-']:
                key, = values['-dgxi_sel-']
                data[key].doDgxiPlot = False

        if event in {'-delfile-', '-dgxi_send-', '-dgxi_sall-',
                     '-dgxi_erase-'}:
            dgxi_keys = []
            for key, lb_class in data.items():
                if lb_class.doDgxiPlot:
                    dgxi_keys.append(key)
            window['-dgxi_sel-'].update(dgxi_keys, set_to_index=0)
            _, values = window.read(0)  # Needed for updating listbox.

        if event in {'-dgxi_fracset-'}:
            if values['-dgxi_sel-']:
                key, = values['-dgxi_sel-']
                data[key].molfrac = prep_input('-dgxi_fracread-', 0, 1)

        if event in {'-delfile-', '-dgxi_send-', '-dgxi_sall-',
                     '-dgxi_erase-', '-dgxi_sel-'}:
            if values['-dgxi_sel-']:
                key, = values['-dgxi_sel-']
                window['-dgxi_fracread-'].update(data[key].molfrac)

        if event in {'-dgxi_draw-'}:
            # data_ar_dgxi
            frac = []
            dgxidata = []
            for lb_class in data.values():
                if lb_class.doDgxiPlot:
                    frac.append(lb_class.molfrac)
                    dgxidata.append(lb_class)

            if any((frac.count(x) > 1) for x in frac):
                window['-dgxi_msg-'].update("There are multiple files with \
the same component.\nUse Set to change this.")
                continue

            if any(not (f in frac) for f in (0, 1)):
                window['-dgxi_msg-'].update("You must have files with ideal \
components.\nUse Set to 0 and (or) 1 to specify.")
                continue
            window['-dgxi_msg-'].update('')
            # Sorting fractions
            frac, dgxidata = sort(frac, dgxidata)

            pi_bound = prep_input('-dgxi_piset-', 0)
            # Checking validity of pi bound.
            for lb_class in dgxidata:
                min_pi = lb_class.data_pi.min()
                if min_pi < 0:
                    continue
                elif int(min_pi * 10) != 0:  # 0,099 would be close to zero.
                    window['-dgxi_msg-'].update("There is no π value close to \
zero in one of the files. Extrapolation is applied, be cautious!")

            for lb_class in dgxidata:
                if max(lb_class.data_pi) < pi_bound:
                    window['-dgxi_msg-'].update("The selected π bound is \
higher than measured in one of the files. Extrapolation is applied, be \
cautious!")

            pi = np.linspace(0,
                             pi_bound,
                             dgxidata[0].data_pi.size)
            ar_comp_A = dgxidata[0].interpolated_piar(pi)
            ar_comp_B = dgxidata[-1].interpolated_piar(pi)

            dg_list = []
            for chi_comp_A, lb_class in zip(frac, dgxidata):
                ar_ideal = (chi_comp_A * ar_comp_A +
                            (1 - chi_comp_A) * ar_comp_B)
                ar_real = lb_class.interpolated_piar(pi)
                nm2mN_m_to_j = 1e-21
                dG = (Avogadro *
                      nm2mN_m_to_j *
                      np.trapz((ar_real - ar_ideal), x=pi))
                dg_list.append(dG)
            dgxidata[0].plot(fig3, 'dgxi', update=True, dgxi=(frac, dg_list))
            dgxidata[0].lines['dgxi'].set_label('')

            # Copying lines to other data classes for the ability to change
            # line colour, type when selecting through -dgxi_sel- listbox.
            for lb_class in dgxidata[1:]:
                lb_class.lines['dgxi'] = dgxidata[0].lines['dgxi']
            fig3.draw(rescale=True)

        # Events related to all plots.
        if event in {'-leg1-', '-leg2-', '-leg3-', '-openfile-', '-delfile-',
                     '-cmsp_draw-', '-cmsp_dall-', '-cmsp_erase-', '-smooth-'}:
            for line_key in ('piar', 'cmsp'):
                for lb_class in data.values():
                    if line_key in lb_class.lines:
                        line = lb_class.lines[line_key]
                        if values['-leg1-']:
                            line.set_label(lb_class.key)
                        elif values['-leg2-']:
                            line.set_label(lb_class.name)
                        elif values['-leg3-']:
                            line.set_label('')
                fig_dict[line_key].draw()

        if event[5:] in {'_solid-', '_dotted-', '_dashed-', '_dashdot-'}:
            line_key, linestyle = event.strip('-').split('_')
            listbox_key = ('-%s_sel-' % line_key)
            if values[listbox_key]:  # Check if not empty.
                key, = values[listbox_key]
                data[key].lines[line_key].set_linestyle(linestyle)
                fig_dict[line_key].draw()

        if event[5:] in {'_color-'}:
            line_key = event.strip('-').split('_')[0]
            listbox_key = ('-%s_sel-' % line_key)
            if values[listbox_key]:  # Check if not empty.
                key, = values[listbox_key]
                data[key].lines[line_key].set_color(values[event])
                fig_dict[line_key].draw()

        if event[5:] in {'_export-'}:
            key = event.strip('-').split('_')[0]
            fig_dict[key].export(values[event])


if __name__ == "__main__":
    # Initialization of window.
    window = sg.Window("Langmuir–Blodgett trough data analyser", layout,
                       default_element_size=(5, None), element_padding=(0, 0),
                       location=(-4, 0), margins=(4, 0), finalize=True,
                       size=(1366, 768), icon="icon.ico")

    # Initialization of the figures for plotting.
    fig1 = Fig(FIG_SZ, '-CANVAS0-', '-CANVAS0BAR-')
    fig1.ax.set(xlabel="A (nm²/molecule)", ylabel="π (mN/m)",
                title="Surface pressure vs area")

    fig1.tb._update_buttons_checked()
    fig2 = Fig(FIG_SZ, '-CANVAS1-', '-CANVAS1BAR-')
    fig2.ax.set(xlabel="π (mN/m)", ylabel="Cₛ⁻¹ (mN/m)",
                title="Compression modulus vs surface pressure")

    fig3 = Fig(FIG_SZ, '-CANVAS2-', '-CANVAS2BAR-')
    fig3.ax.set(xlabel="χ, component A", ylabel="ΔG (J)",
                title="Excess Gibbs energy of mixing vs component A")

    # Makes less distraction when deleting file and nothing is plotted.
    fig1.draw(rescale=True)
    fig2.draw(rescale=True)
    fig3.draw(rescale=True)

    fig_dict = {'piar': fig1, 'cmsp': fig2, 'dgxi': fig3}

    # Changing the colour of the ibeam, because themes choose it wrong.
    for key in ('-BDIF_WLN-', '-BDIF_ORD-', '-ADIF_WLN-',
                '-ADIF_ORD-', '-dgxi_fracread-',  '-dgxi_piset-',):
        window[key].set_cursor(cursor_color='#000000')

    event = None
    values = None
    event_history = []
    try:
        main()
    except Exception:
        event_history.append(format_exc())
        event_history = '\n'.join(map(str, event_history))
        with open('crash_log.txt', 'w') as f:
            f.write(event_history)
    window.close()
