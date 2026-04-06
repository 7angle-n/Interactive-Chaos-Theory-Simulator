"""
Chaos Theory Simulator
Demonstrates various chaotic dynamical systems using Tkinter + Matplotlib
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.integrate import odeint
import threading
import time




# ─────────────────────────────────────────────
#  COLOR PALETTE  (dark cyberpunk / sci-fi)
# ─────────────────────────────────────────────
BG_DARK    = "#0a0a0f"
BG_PANEL   = "#0f0f1a"
BG_CARD    = "#13131f"
ACCENT     = "#00d4ff"
ACCENT2    = "#ff006e"
ACCENT3    = "#7fff00"
TEXT_PRI   = "#e8e8ff"
TEXT_SEC   = "#7878aa"
BORDER     = "#1e1e3a"
GLOW       = "#00d4ff"

# ── FONT SIZES (easy to tune) ─────────────────
F_HEADER   = ("Consolas", 20, "bold")   # app title
F_CARD_TTL = ("Consolas", 10, "bold")   # card section headings  ← was 8
F_LIST     = ("Consolas", 11)           # system listbox          ← was 9
F_INFO     = ("Consolas", 10)           # info / desc text        ← was 8/9
F_LABEL    = ("Consolas", 11)           # parameter labels        ← was 9
F_SCALE    = ("Consolas", 9)            # slider value readout    ← was 7/8
F_CTRL     = ("Consolas", 11)           # controls labels         ← was 9
F_BTN      = ("Consolas", 12, "bold")   # render button           ← was 11
F_STATUS   = ("Consolas", 10)           # status bar              ← was 8
F_CMAP     = ("Consolas", 10)           # colormap combo          ← was 9
F_TOOLBAR  = ("Consolas", 9)            # bottom toolbar          ← was 8




# ─────────────────────────────────────────────
#  CHAOTIC SYSTEM DEFINITIONS
# ─────────────────────────────────────────────

def lorenz(state, t, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

def rossler(state, t, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    return [-(y+z), x+a*y, b+z*(x-c)]

def chen(state, t, a=35, b=3, c=28):
    x, y, z = state
    return [a*(y-x), (c-a)*x-x*z+c*y, x*y-b*z]

def aizawa(state, t, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
    x, y, z = state
    return [
        (z-b)*x - d*y,
        d*x + (z-b)*y,
        c + a*z - z**3/3 - (x**2+y**2)*(1+e*z) + f*z*x**3
    ]

def thomas(state, t, b=0.208186):
    x, y, z = state
    return [np.sin(y)-b*x, np.sin(z)-b*y, np.sin(x)-b*z]

def dadras(state, t, a=3, b=2.7, c=1.7, d=2, e=9):
    x, y, z = state
    return [y-a*x+b*y*z, c*y-x*z+z, d*x*y-e*z]

def halvorsen(state, t, a=1.4):
    x, y, z = state
    return [-a*x-4*y-4*z-y**2, -a*y-4*z-4*x-z**2, -a*z-4*x-4*y-x**2]

def three_scroll(state, t, a=32.48, b=45.84, c=1.18, d=0.13, e=0.57, f=14.7):
    x, y, z = state
    return [a*(y-x)+d*x*z, b*x-x*z+f*y, c*z+x*y-e*x**2]

def bouali(state, t, a=0.3, b=1.0, s=4.0):
    x, y, z = state
    return [x*(4-y)+a*z, -y*(1-x**2), -x*(1.5-s*z)-b*z]

def dequan_li(state, t, a=40, c=1.833, d=0.16, e=0.65, k=55, f=20):
    x, y, z = state
    return [a*(y-x)+d*x*z, k*x+f*y-x*z, c*z+x*y-e*x**2]




# ─────────────────────────────────────────────
#  2-D MAPS
# ─────────────────────────────────────────────

def henon_map(n_points=50000, a=1.4, b=0.3):
    x, y = 0.1, 0.1
    xs, ys = [], []
    for _ in range(n_points):
        x_new = 1 - a*x**2 + y
        y_new = b*x
        x, y = x_new, y_new
        xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)

def tinkerbell_map(n_points=50000, a=0.9, b=-0.6013, c=2.0, d=0.5):
    x, y = -0.72, -0.64
    xs, ys = [], []
    for _ in range(n_points):
        x_new = x**2 - y**2 + a*x + b*y
        y_new = 2*x*y + c*x + d*y
        x, y = x_new, y_new
        if abs(x) < 1e6 and abs(y) < 1e6:
            xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)

def ikeda_map(n_points=50000, u=0.9):
    x, y = 0.1, 0.1
    xs, ys = [], []
    for _ in range(n_points):
        t = 0.4 - 6/(1 + x**2 + y**2)
        x_new = 1 + u*(x*np.cos(t) - y*np.sin(t))
        y_new = u*(x*np.sin(t) + y*np.cos(t))
        x, y = x_new, y_new
        if abs(x) < 1e6 and abs(y) < 1e6:
            xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)

def logistic_map(r=3.99, n_points=5000, x0=0.5):
    x = x0
    xs = []
    for _ in range(n_points):
        x = r*x*(1-x)
        xs.append(x)
    return np.arange(len(xs)), np.array(xs)

def clifford_attractor(n_points=200000, a=-1.4, b=1.6, c=1.0, d=0.7):
    x, y = 0.0, 0.0
    xs, ys = np.zeros(n_points), np.zeros(n_points)
    for i in range(n_points):
        x_new = np.sin(a*y) + c*np.cos(a*x)
        y_new = np.sin(b*x) + d*np.cos(b*y)
        x, y = x_new, y_new
        xs[i], ys[i] = x, y
    return xs, ys

def dejong_attractor(n_points=200000, a=1.641, b=1.902, c=0.316, d=1.525):
    x, y = 0.0, 0.0
    xs, ys = np.zeros(n_points), np.zeros(n_points)
    for i in range(n_points):
        x_new = np.sin(a*y) - np.cos(b*x)
        y_new = np.sin(c*x) - np.cos(d*y)
        x, y = x_new, y_new
        xs[i], ys[i] = x, y
    return xs, ys




# ─────────────────────────────────────────────
#  SYSTEM REGISTRY
# ─────────────────────────────────────────────

SYSTEMS = {
    "Lorenz Attractor": {
        "type": "3d_ode", "fn": lorenz,
        "ic": [1.0, 1.0, 1.0], "t_end": 60, "dt": 0.005,
        "color": ACCENT, "color2": ACCENT2,
        "desc": "The iconic butterfly attractor. Discovered by Edward Lorenz in 1963\nwhile modeling atmospheric convection. σ=10, ρ=28, β=8/3",
        "params": {"sigma": (1, 50, 10), "rho": (1, 60, 28), "beta": (0.1, 10, 2.67)},
    },
    "Rössler Attractor": {
        "type": "3d_ode", "fn": rossler,
        "ic": [0.1, 0.1, 0.1], "t_end": 400, "dt": 0.01,
        "color": "#ff6b35", "color2": "#f7c59f",
        "desc": "Proposed by Otto Rössler in 1976. Simpler than Lorenz but\nequally chaotic. a=0.2, b=0.2, c=5.7",
        "params": {"a": (0.1, 1.0, 0.2), "b": (0.1, 1.0, 0.2), "c": (1.0, 15.0, 5.7)},
    },
    "Chen Attractor": {
        "type": "3d_ode", "fn": chen,
        "ic": [-0.1, 0.5, -0.4], "t_end": 60, "dt": 0.002,
        "color": "#b44fff", "color2": "#f0abfc",
        "desc": "Discovered by Guanrong Chen in 1999. Related to Lorenz but\nwith different topology. a=35, b=3, c=28",
        "params": {"a": (20, 50, 35), "b": (1, 10, 3), "c": (10, 50, 28)},
    },
    "Aizawa Attractor": {
        "type": "3d_ode", "fn": aizawa,
        "ic": [0.1, 1.0, 0.01], "t_end": 250, "dt": 0.01,
        "color": "#00ffb3", "color2": "#00d4ff",
        "desc": "A toroidal chaotic attractor with a complex wrapped structure.\nDisplays beautiful winding behavior.",
        "params": {"a": (0.5, 1.5, 0.95), "b": (0.1, 1.5, 0.7), "c": (0.1, 1.5, 0.6)},
    },
    "Thomas Attractor": {
        "type": "3d_ode", "fn": thomas,
        "ic": [0.1, 0.0, 0.0], "t_end": 500, "dt": 0.05,
        "color": "#ff0055", "color2": "#ff6699",
        "desc": "René Thomas's cyclically symmetric attractor. The parameter b\ncontrols dissipation. Near b≈0.208 is chaotic.",
        "params": {"b": (0.1, 0.5, 0.208)},
    },
    "Dadras Attractor": {
        "type": "3d_ode", "fn": dadras,
        "ic": [1.0, 0.0, 0.3], "t_end": 200, "dt": 0.005,
        "color": "#ffcc00", "color2": "#ff8800",
        "desc": "A relatively new chaotic attractor (2009) with a distinctive\nbifurcation structure. a=3, b=2.7, c=1.7, d=2, e=9",
        "params": {"a": (1, 6, 3), "b": (1, 6, 2.7), "c": (0.5, 4, 1.7)},
    },
    "Halvorsen Attractor": {
        "type": "3d_ode", "fn": halvorsen,
        "ic": [-1.48, -1.51, 2.04], "t_end": 200, "dt": 0.005,
        "color": "#00ffff", "color2": "#0080ff",
        "desc": "A cyclically symmetric chaotic system with three saddle points.\nBeautiful figure-8 loop structure.",
        "params": {"a": (0.5, 2.5, 1.4)},
    },
    "Three-Scroll Attractor": {
        "type": "3d_ode", "fn": three_scroll,
        "ic": [0.1, 0.1, 0.1], "t_end": 100, "dt": 0.001,
        "color": "#39ff14", "color2": "#00cc88",
        "desc": "A chaotic system with three scrolls. Exhibits multi-scroll\nchaotic behavior. Complex folding structure.",
        "params": {"a": (20, 50, 32.48), "b": (30, 60, 45.84), "c": (0.5, 3, 1.18)},
    },
    "Bouali Attractor": {
        "type": "3d_ode", "fn": bouali,
        "ic": [1, 1, 0], "t_end": 400, "dt": 0.01,
        "color": "#ff3399", "color2": "#ff99cc",
        "desc": "Fouad Bouali's attractor with a distinctive pretzel-like shape.\nMultiple coexisting attractors possible.",
        "params": {"a": (0.1, 0.9, 0.3), "b": (0.5, 2.0, 1.0), "s": (1.0, 8.0, 4.0)},
    },
    "Héon Map": {
        "type": "2d_map", "fn": henon_map,
        "color": ACCENT, "color2": ACCENT2,
        "desc": "Discovered by Michel Hénon in 1976. A 2D discrete map\nthat exhibits fractal strange attractor. a=1.4, b=0.3",
        "params": {"a": (0.5, 2.0, 1.4), "b": (0.1, 0.9, 0.3)},
    },
    "Tinkerbell Map": {
        "type": "2d_map", "fn": tinkerbell_map,
        "color": "#ff99ff", "color2": "#cc66ff",
        "desc": "Named for its fairy-like shape. A 2D discrete map\nshowing beautiful fractal boundaries.",
        "params": {"a": (0.5, 1.5, 0.9), "b": (-1.0, 0.0, -0.6013), "c": (1.0, 3.0, 2.0)},
    },
    "Ikeda Map": {
        "type": "2d_map", "fn": ikeda_map,
        "color": "#00ffaa", "color2": "#00aaff",
        "desc": "Models light bouncing in a nonlinear optical cavity.\nU parameter controls the level of chaos.",
        "params": {"u": (0.6, 0.99, 0.9)},
    },
    "Clifford Attractor": {
        "type": "2d_map_dense", "fn": clifford_attractor,
        "color": "#ff6600", "color2": "#ffaa00",
        "desc": "Clifford Pickover's attractor. Millions of points form\ndelicate fractal cloud structures.",
        "params": {"a": (-2.5, 2.5, -1.4), "b": (-2.5, 2.5, 1.6), "c": (-2.5, 2.5, 1.0)},
    },
    "De Jong Attractor": {
        "type": "2d_map_dense", "fn": dejong_attractor,
        "color": "#aa00ff", "color2": "#ff00aa",
        "desc": "Peter de Jong's attractor. Sinusoidal maps create\nstunning, symmetrical strange attractors.",
        "params": {"a": (-3, 3, 1.641), "b": (-3, 3, 1.902), "c": (-3, 3, 0.316)},
    },
    "Logistic Map": {
        "type": "bifurcation", "fn": logistic_map,
        "color": "#ffff00", "color2": "#ff8800",
        "desc": "The simplest route to chaos. xₙ₊₁ = rxₙ(1-xₙ)\nThe bifurcation diagram reveals period-doubling to chaos.",
        "params": {"r": (2.5, 4.0, 3.99)},
    },
}




# ─────────────────────────────────────────────
#  MAIN APPLICATION CLASS
# ─────────────────────────────────────────────

class ChaosSimulator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("⚛  Chaos Theory Simulator")
        self.geometry("1400x860")
        self.minsize(1100, 700)
        self.configure(bg=BG_DARK)

        self._anim = None
        self._computing = False
        self._param_vars = {}
        self._current_system = None

        # Mouse-drag rotation state
        self._drag_active = False
        self._drag_x0 = 0
        self._drag_y0 = 0
        self._drag_elev0 = 25.0
        self._drag_azim0 = 0.0
        self._auto_elev = 25.0
        self._auto_azim = 0.0
        self._anim_frame = 0
        self._3d_ax = None
        self._zoom_scale = 1.0

        self._setup_styles()
        self._build_ui()
        self._select_system("Lorenz Attractor")

    # ── STYLES ──────────────────────────────────
    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TFrame", background=BG_DARK)
        style.configure("Card.TFrame", background=BG_CARD)
        style.configure("Panel.TFrame", background=BG_PANEL)

        style.configure("TLabel",
            background=BG_DARK, foreground=TEXT_PRI,
            font=F_LABEL)

        style.configure("Title.TLabel",
            background=BG_DARK, foreground=ACCENT,
            font=F_HEADER)

        style.configure("Sub.TLabel",
            background=BG_DARK, foreground=TEXT_SEC,
            font=F_INFO)

        style.configure("Desc.TLabel",
            background=BG_CARD, foreground=TEXT_SEC,
            font=F_INFO, wraplength=300, justify="left")

        style.configure("SysName.TLabel",
            background=BG_CARD, foreground=TEXT_PRI,
            font=("Consolas", 13, "bold"))

        style.configure("TScale",
            background=BG_DARK, troughcolor=BORDER,
            sliderlength=18, sliderrelief="flat")

        style.configure("Accent.TButton",
            background=ACCENT, foreground=BG_DARK,
            font=F_BTN,
            relief="flat", borderwidth=0, padding=(14, 8))
        style.map("Accent.TButton",
            background=[("active", "#33ddff"), ("pressed", "#0099bb")])

        style.configure("TCombobox",
            fieldbackground=BG_PANEL, background=BG_PANEL,
            foreground=TEXT_PRI, selectbackground=ACCENT,
            font=F_CMAP)

        style.configure("TScrollbar",
            background=BG_PANEL, troughcolor=BG_DARK,
            arrowcolor=TEXT_SEC, relief="flat")

        style.configure("TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab",
            background=BG_PANEL, foreground=TEXT_SEC,
            font=F_INFO, padding=(12, 6))
        style.map("TNotebook.Tab",
            background=[("selected", BG_CARD)],
            foreground=[("selected", ACCENT)])

    # ── UI LAYOUT ────────────────────────────────
    def _build_ui(self):
        # ── Header ──────────────────────────────
        header = tk.Frame(self, bg=BG_DARK, pady=10)
        header.pack(fill="x", padx=20)

        tk.Label(header, text="⚛  CHAOS THEORY SIMULATOR",
            bg=BG_DARK, fg=ACCENT,
            font=F_HEADER).pack(side="left")

        tk.Label(header,
            text="Strange Attractors & Dynamical Systems",
            bg=BG_DARK, fg=TEXT_SEC,
            font=F_INFO).pack(side="left", padx=18, pady=4)

        tk.Label(header, text=" v2.0 ",
            bg=BORDER, fg=ACCENT,
            font=("Consolas", 9, "bold"),
            relief="flat", padx=4).pack(side="right", padx=6)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=10)

        # ── Main Panes ───────────────────────────
        panes = tk.PanedWindow(self, orient="horizontal",
            bg=BG_DARK, sashwidth=4, sashrelief="flat",
            sashpad=2, handlesize=0)
        panes.pack(fill="both", expand=True, padx=10, pady=6)

        # Left panel — slightly wider to accommodate bigger fonts
        left = tk.Frame(panes, bg=BG_DARK, width=330)
        panes.add(left, minsize=300)

        # Right panel (plot area) — expands to fill remaining space
        right = tk.Frame(panes, bg=BG_DARK)
        panes.add(right, minsize=600)

        self._build_left_panel(left)
        self._build_plot_area(right)

    def _build_left_panel(self, parent):
        parent.columnconfigure(0, weight=1)

        # System selector ──────────────────────
        sec = self._card(parent, "SELECT SYSTEM")
        sec.pack(fill="x", pady=(0,6))

        self._sys_var = tk.StringVar(value="Lorenz Attractor")
        lb_frame = tk.Frame(sec, bg=BG_CARD)
        lb_frame.pack(fill="both", expand=True, padx=8, pady=(0,8))

        scrollbar = tk.Scrollbar(lb_frame, bg=BG_PANEL,
            troughcolor=BG_DARK, relief="flat", width=8)
        self._lb = tk.Listbox(lb_frame,
            bg=BG_CARD, fg=TEXT_PRI,
            selectbackground=ACCENT, selectforeground=BG_DARK,
            font=F_LIST,              # ← larger listbox font
            relief="flat", borderwidth=0,
            activestyle="none", yscrollcommand=scrollbar.set,
            height=13, exportselection=False)
        scrollbar.config(command=self._lb.yview)

        for name in SYSTEMS:
            self._lb.insert(tk.END, f"  {name}")

        self._lb.selection_set(0)
        self._lb.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._lb.bind("<<ListboxSelect>>", self._on_lb_select)

        # Info card ────────────────────────────
        info_card = self._card(parent, "SYSTEM INFO")
        info_card.pack(fill="x", pady=(0,6))

        self._info_label = tk.Label(info_card,
            text="", bg=BG_CARD, fg=TEXT_SEC,
            font=F_INFO,              # ← larger info font
            wraplength=290,
            justify="left", anchor="nw")
        self._info_label.pack(fill="x", padx=8, pady=(0,8))

        # Parameters ───────────────────────────
        param_card = self._card(parent, "PARAMETERS")
        param_card.pack(fill="x", pady=(0,6))

        self._param_frame = tk.Frame(param_card, bg=BG_CARD)
        self._param_frame.pack(fill="x", padx=8, pady=(0,8))

        # Controls ─────────────────────────────
        ctrl_card = self._card(parent, "CONTROLS")
        ctrl_card.pack(fill="x", pady=(0,6))

        ctrl_inner = tk.Frame(ctrl_card, bg=BG_CARD)
        ctrl_inner.pack(fill="x", padx=8, pady=(4,8))

        # Animate toggle
        self._anim_var = tk.BooleanVar(value=True)
        anim_frame = tk.Frame(ctrl_inner, bg=BG_CARD)
        anim_frame.pack(fill="x", pady=2)
        tk.Label(anim_frame, text="Animation", bg=BG_CARD,
            fg=TEXT_SEC, font=F_CTRL).pack(side="left")
        self._anim_cb = tk.Checkbutton(anim_frame,
            variable=self._anim_var, bg=BG_CARD,
            fg=ACCENT, selectcolor=BG_CARD,
            activebackground=BG_CARD, relief="flat")
        self._anim_cb.pack(side="right")

        # Point density
        pt_frame = tk.Frame(ctrl_inner, bg=BG_CARD)
        pt_frame.pack(fill="x", pady=2)
        tk.Label(pt_frame, text="Point density",
            bg=BG_CARD, fg=TEXT_SEC,
            font=F_CTRL).pack(side="left")
        self._density_var = tk.IntVar(value=3)
        density_scale = tk.Scale(pt_frame,
            variable=self._density_var, from_=1, to=5,
            orient="horizontal", bg=BG_CARD, fg=TEXT_PRI,
            troughcolor=BORDER, highlightthickness=0,
            relief="flat", sliderrelief="flat",
            activebackground=ACCENT, showvalue=True,
            font=F_SCALE, length=110)
        density_scale.pack(side="right")

        # Colormap
        cm_frame = tk.Frame(ctrl_inner, bg=BG_CARD)
        cm_frame.pack(fill="x", pady=2)
        tk.Label(cm_frame, text="Colormap",
            bg=BG_CARD, fg=TEXT_SEC,
            font=F_CTRL).pack(side="left")
        self._cmap_var = tk.StringVar(value="plasma")
        cmap_combo = ttk.Combobox(cm_frame,
            textvariable=self._cmap_var,
            values=["plasma","inferno","viridis","magma",
                    "cool","hot","spring","rainbow","twilight"],
            width=10, state="readonly", font=F_CMAP)
        cmap_combo.pack(side="right")

        tk.Frame(ctrl_inner, bg=BORDER, height=1).pack(fill="x", pady=6)
        self._render_btn = tk.Button(ctrl_inner,
            text="▶  RENDER",
            bg=ACCENT, fg=BG_DARK,
            font=F_BTN,               # ← larger button font
            relief="flat", bd=0, cursor="hand2",
            activebackground="#33ddff",
            activeforeground=BG_DARK,
            command=self._render)
        self._render_btn.pack(fill="x", ipady=8)

        self._status = tk.Label(ctrl_inner,
            text="● READY", bg=BG_CARD, fg=ACCENT3,
            font=F_STATUS,            # ← larger status font
            anchor="w")
        self._status.pack(fill="x", pady=(4,0))

    def _build_plot_area(self, parent):
        # Make right panel expand fully
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        plt.style.use("dark_background")
        # Use tight_layout and constrained_layout so the figure always
        # fills the canvas without dead space
        self._fig = plt.figure(facecolor=BG_DARK, constrained_layout=True)
        self._fig.patch.set_facecolor(BG_DARK)

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        widget = self._canvas.get_tk_widget()
        # grid instead of pack so it truly fills the frame
        widget.grid(row=0, column=0, sticky="nsew")
        widget.configure(bg=BG_DARK)

        # Mouse events
        self._canvas.mpl_connect("button_press_event",   self._on_mouse_press)
        self._canvas.mpl_connect("motion_notify_event",  self._on_mouse_drag)
        self._canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self._canvas.mpl_connect("scroll_event",         self._on_scroll)

        # Toolbar area
        tb = tk.Frame(parent, bg=BG_PANEL, height=28)
        tb.grid(row=1, column=0, sticky="ew")
        tk.Label(tb,
            text="  CHAOS THEORY SIMULATOR  |  Nonlinear Dynamics Visualization",
            bg=BG_PANEL, fg=TEXT_SEC,
            font=F_TOOLBAR).pack(side="left", pady=4)

    # ── HELPER: card widget ──────────────────────
    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=BORDER, bd=1)
        outer.pack(fill="x", pady=2)
        inner = tk.Frame(outer, bg=BG_CARD, padx=2, pady=2)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Label(inner, text=f"  {title}",
            bg=BG_CARD, fg=ACCENT,
            font=F_CARD_TTL,          # ← larger card-title font
            anchor="w").pack(fill="x", pady=(4,2))
        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=6, pady=(0,4))
        return inner

    # ── EVENTS ──────────────────────────────────
    def _on_lb_select(self, event):
        sel = self._lb.curselection()
        if not sel:
            return
        name = list(SYSTEMS.keys())[sel[0]]
        self._select_system(name)

    def _select_system(self, name):
        self._current_system = name
        sys = SYSTEMS[name]

        self._info_label.config(text=sys["desc"])

        for w in self._param_frame.winfo_children():
            w.destroy()
        self._param_vars.clear()

        for pname, (lo, hi, default) in sys.get("params", {}).items():
            row = tk.Frame(self._param_frame, bg=BG_CARD)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=pname, width=6,
                bg=BG_CARD, fg=TEXT_PRI,
                font=F_LABEL,         # ← larger param-label font
                anchor="w").pack(side="left")
            var = tk.DoubleVar(value=default)
            self._param_vars[pname] = var
            scale = tk.Scale(row, variable=var,
                from_=lo, to=hi, orient="horizontal",
                bg=BG_CARD, fg=TEXT_SEC,
                troughcolor=BORDER, highlightthickness=0,
                relief="flat", sliderrelief="flat",
                activebackground=ACCENT,
                resolution=(hi-lo)/200,
                font=F_SCALE,         # ← larger scale readout font
                digits=4,
                length=170)
            scale.pack(side="right")

    # ── MOUSE DRAG ROTATION ─────────────────────
    def _on_mouse_press(self, event):
        if self._3d_ax is None or event.button != 1:
            return
        self._drag_active = True
        self._drag_x0 = event.x
        self._drag_y0 = event.y
        self._drag_elev0 = self._auto_elev
        self._drag_azim0 = self._auto_azim
        if self._anim is not None:
            try:
                self._anim.event_source.stop()
            except Exception:
                pass

    def _on_mouse_drag(self, event):
        if not self._drag_active or self._3d_ax is None:
            return
        dx = event.x - self._drag_x0
        dy = event.y - self._drag_y0
        new_azim = self._drag_azim0 - dx * 0.4
        new_elev = max(-90, min(90, self._drag_elev0 + dy * 0.3))
        self._auto_azim = new_azim
        self._auto_elev = new_elev
        self._3d_ax.view_init(elev=new_elev, azim=new_azim)
        self._canvas.draw_idle()

    def _on_mouse_release(self, event):
        if not self._drag_active:
            return
        self._drag_active = False
        if self._anim is not None and self._anim_var.get():
            try:
                self._anim.event_source.start()
            except Exception:
                pass

    def _on_scroll(self, event):
        if self._3d_ax is None:
            return
        factor = 0.90 if event.step > 0 else 1.11
        self._zoom_scale *= factor
        ax = self._3d_ax
        for get_lim, set_lim in [
            (ax.get_xlim3d, ax.set_xlim3d),
            (ax.get_ylim3d, ax.set_ylim3d),
            (ax.get_zlim3d, ax.set_zlim3d),
        ]:
            lo, hi = get_lim()
            mid = (lo + hi) / 2
            half = (hi - lo) / 2 * factor
            set_lim(mid - half, mid + half)
        self._canvas.draw_idle()

    # ── RENDER ──────────────────────────────────
    def _render(self):
        if self._computing:
            return
        self._computing = True
        self._status.config(text="● COMPUTING...", fg=ACCENT2)
        self._render_btn.config(state="disabled")

        if self._anim is not None:
            try:
                self._anim.event_source.stop()
            except Exception:
                pass
            self._anim = None

        threading.Thread(target=self._compute_and_plot, daemon=True).start()

    def _compute_and_plot(self):
        try:
            name = self._current_system
            sys_data = SYSTEMS[name]
            stype = sys_data["type"]
            fn = sys_data["fn"]
            cmap = self._cmap_var.get()
            density = self._density_var.get()
            params = {k: v.get() for k, v in self._param_vars.items()}

            self._fig.clear()
            self._3d_ax = None

            if stype == "3d_ode":
                self._plot_3d_ode(name, sys_data, fn, params, cmap, density)
            elif stype == "2d_map":
                self._plot_2d_map(name, sys_data, fn, params, cmap, density)
            elif stype == "2d_map_dense":
                self._plot_dense(name, sys_data, fn, params, cmap)
            elif stype == "bifurcation":
                self._plot_bifurcation(name, sys_data, cmap)

            self._canvas.draw()
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Render Error", str(e)))
        finally:
            self._computing = False
            self.after(0, lambda: (
                self._status.config(text="● READY", fg=ACCENT3),
                self._render_btn.config(state="normal")
            ))

    # ── 3D ODE ──────────────────────────────────
    def _plot_3d_ode(self, name, sys_data, fn, params, cmap, density):
        t_end = sys_data["t_end"]
        dt = sys_data["dt"]
        ic = sys_data["ic"]
        t = np.arange(0, t_end, dt)
        n_pts = len(t) * density // 3

        sig = list(params.values())
        sol = odeint(fn, ic, t, args=tuple(sig))
        x, y, z = sol[:,0], sol[:,1], sol[:,2]

        idx = np.linspace(0, len(x)-1, min(n_pts*2000, len(x)), dtype=int)
        x, y, z = x[idx], y[idx], z[idx]

        ax = self._fig.add_subplot(111, projection='3d',
            facecolor=BG_DARK)

        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(x)))
        ax.scatter(x, y, z, c=colors, s=0.28, alpha=1.0, linewidths=0)

        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor((0, 0, 0, 0))
        ax.grid(False)
        ax.set_facecolor(BG_DARK)
        for spine in [ax.xaxis.line, ax.yaxis.line, ax.zaxis.line]:
            spine.set_color((0, 0, 0, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("", labelpad=0)
        ax.set_ylabel("", labelpad=0)
        ax.set_zlabel("", labelpad=0)
        ax.set_title(f"{name}", color=sys_data["color"],
            fontsize=14, fontweight="bold", pad=15,
            fontfamily="monospace")

        sm = plt.cm.ScalarMappable(cmap=cmap,
            norm=plt.Normalize(vmin=0, vmax=t_end))
        sm.set_array([])
        cb = self._fig.colorbar(sm, ax=ax, shrink=0.5,
            pad=0.1, aspect=20)
        cb.set_label("Time", color=TEXT_SEC, fontsize=8)
        cb.ax.yaxis.set_tick_params(color=TEXT_SEC, labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_SEC)
        cb.outline.set_edgecolor(BORDER)

        self._fig.patch.set_facecolor(BG_DARK)

        self._3d_ax = ax
        self._auto_elev = 22.0
        self._auto_azim = 0.0
        self._zoom_scale = 1.0

        if self._anim_var.get():
            def update(frame):
                if self._drag_active:
                    return []
                self._auto_azim += 1.2
                self._auto_elev = 22 + 12 * np.sin(np.radians(self._auto_azim * 0.33))
                ax.view_init(elev=self._auto_elev, azim=self._auto_azim)
                return []
            self._anim = FuncAnimation(self._fig, update,
                frames=None, interval=16, blit=False)

    # ── 2D MAP ──────────────────────────────────
    def _plot_2d_map(self, name, sys_data, fn, params, cmap, density):
        n = 30000 * density
        sig = list(params.values())

        import inspect
        fargs = inspect.signature(fn).parameters
        call_args = [n] + sig[:len(fargs)-1]
        xs, ys = fn(*call_args)

        ax = self._fig.add_subplot(111, facecolor=BG_DARK)
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(xs)))
        ax.scatter(xs, ys, c=colors, s=0.2, alpha=0.6, linewidths=0)

        ax.set_facecolor(BG_DARK)
        ax.tick_params(colors=TEXT_SEC, labelsize=7)
        ax.set_xlabel("X", color=TEXT_SEC, fontsize=9)
        ax.set_ylabel("Y", color=TEXT_SEC, fontsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, linewidth=0.3, alpha=0.4)
        ax.set_title(f"{name}", color=sys_data["color"],
            fontsize=14, fontweight="bold",
            fontfamily="monospace")
        self._fig.patch.set_facecolor(BG_DARK)

    # ── DENSE ATTRACTOR ─────────────────────────
    def _plot_dense(self, name, sys_data, fn, params, cmap):
        sig = list(params.values())
        xs, ys = fn(200000, *sig[:3])

        ax = self._fig.add_subplot(111, facecolor=BG_DARK)

        h, xedges, yedges = np.histogram2d(xs, ys, bins=800, density=True)
        h = np.log1p(h)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax.imshow(h.T, extent=extent, origin='lower',
            cmap=cmap, aspect='auto', interpolation='bilinear')

        ax.set_facecolor(BG_DARK)
        ax.tick_params(colors=TEXT_SEC, labelsize=7)
        ax.set_xlabel("X", color=TEXT_SEC, fontsize=9)
        ax.set_ylabel("Y", color=TEXT_SEC, fontsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.set_title(f"{name}", color=sys_data["color"],
            fontsize=14, fontweight="bold",
            fontfamily="monospace")
        self._fig.patch.set_facecolor(BG_DARK)

    # ── BIFURCATION ─────────────────────────────
    def _plot_bifurcation(self, name, sys_data, cmap):
        ax = self._fig.add_subplot(111, facecolor=BG_DARK)

        r_vals = np.linspace(2.5, 4.0, 1500)
        x = 0.5 * np.ones(len(r_vals))
        for _ in range(300):
            x = r_vals * x * (1 - x)
        all_r, all_x = [], []
        for _ in range(300):
            x = r_vals * x * (1 - x)
            all_r.append(r_vals)
            all_x.append(x.copy())

        all_r = np.concatenate(all_r)
        all_x = np.concatenate(all_x)

        ax.scatter(all_r, all_x, s=0.02, alpha=0.4,
            c=all_r, cmap=cmap, linewidths=0)

        ax.set_facecolor(BG_DARK)
        ax.tick_params(colors=TEXT_SEC, labelsize=8)
        ax.set_xlabel("r  (growth rate)", color=TEXT_SEC, fontsize=10)
        ax.set_ylabel("x  (population)", color=TEXT_SEC, fontsize=10)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, linewidth=0.3, alpha=0.3)
        ax.set_title(f"{name}  —  Bifurcation Diagram",
            color=sys_data["color"],
            fontsize=14, fontweight="bold", fontfamily="monospace")

        for r_ann, lbl in [(3.0,"Period 2"), (3.449,"Period 4"),
                           (3.544,"Period 8"), (3.57,"Chaos begins")]:
            ax.axvline(r_ann, color=ACCENT2, linewidth=0.6,
                linestyle="--", alpha=0.5)
            ax.text(r_ann+0.005, 0.85, lbl,
                color=ACCENT2, fontsize=7,
                fontfamily="monospace", rotation=90, va="top")

        self._fig.patch.set_facecolor(BG_DARK)




# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = ChaosSimulator()
    app.mainloop()