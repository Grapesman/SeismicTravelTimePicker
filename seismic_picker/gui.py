"""Tk/Matplotlib desktop UI. Original seismic traces; no rasterized picking."""
from __future__ import annotations
import copy
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, colorchooser

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
from .workspace import SessionLibrary

from . import __version__
from .segy import read_segy, read_segy_headers, SeismicRecord, SegyError
from .picking import PickerState, extrema
from .qc import audit_picks, check_phase_continuity
from .picking import export_table, nearest_peak
from .waves import WaveLayer, WaveType, wave_color, valid_code, rows_for_layer
from .importer import read_pick_file, validate_pick_import
from .hodograph import from_workspace, geometry_relation
from .tooltips import ToolTip


class PickerApp(tk.Tk):
    def __init__(self, initial_file: str | None = None):
        super().__init__()
        self.title(f'Pic Editor (PE)  •  SEG-Y Picker  •  {__version__}')
        self.geometry('1550x930')
        self.minsize(1100, 680)
        self.configure(bg='#f3f6fb')
        # Branding icon: usable when running from sources and inside PyInstaller EXE.
        self._app_icon = None
        try:
            base_path = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent.parent))
            icon_file = base_path / 'assets' / 'pic_editor_icon.png'
            if icon_file.is_file():
                self._app_icon = tk.PhotoImage(file=str(icon_file))
                self.iconphoto(True, self._app_icon)
        except (tk.TclError, OSError):
            pass
        self.sessions = SessionLibrary()
        self.current_path: Path | None = None
        self.workspace_path: Path | None = None
        self._switching = False
        self.record: SeismicRecord | None = None
        self.state_model: PickerState | None = None
        self.display_signals: np.ndarray | None = None
        self.display_x: np.ndarray | None = None
        self._signals_cache_key = None
        self._signals_cache = None
        self.gain_var = tk.DoubleVar(value=1.1)
        self.gain_label_var=tk.StringVar(value='1.10×')
        self.gain_var.trace_add('write',lambda *_:self.gain_label_var.set(f'{self.gain_var.get():.2f}×'))
        self.unit_var = tk.StringVar(value='мс')
        self.agc_var = tk.BooleanVar(value=True)
        self.color_var = tk.BooleanVar(value=True)
        self.pol_var = tk.StringVar(value='positive')
        self.radius_var = tk.StringVar(value='6')
        self.show_all_var = tk.BooleanVar(value=False)
        self.mode_var = tk.StringVar(value='pick')
        self.phase_warning_var = tk.BooleanVar(value=True)
        self.tracking_radius_var = tk.StringVar(value='8')
        self.phase_tolerance_var = tk.StringVar(value='8')
        self.show_guides_var = tk.BooleanVar(value=True)
        self.show_grid_var = tk.BooleanVar(value=True)
        self.current_wave_code = '0'
        self.wave_code_var = tk.StringVar(value='0')
        self.auto_interp_var = tk.BooleanVar(value=True)
        self._wave_selector_codes: list[str] = []
        self._setting_visible = False
        self._pan_origin = None
        self._pan_active = False
        self._hodo_point_lookup = {}
        self._hodo_drag = None
        self._hodo_hover = None
        self.dragging_trace: int | None = None
        self.drag_press_y: float | None = None
        self.drag_artist = None
        self.drag_active = False
        self._motion_last = None
        self.result_window = None
        self.lock_trace_var = tk.BooleanVar(value=False)
        self.unsaved_changes = False
        self.info_var = tk.StringVar(value='Откройте SEG-Y / SGY, чтобы начать работу')
        self.stats_var = tk.StringVar(value='Файл не открыт')
        self.selected_trace: int | None = None
        self.history: list[dict[int,int]] = []
        self._draw_pending = False
        self._region_start = None
        self._region_patch = None
        self._region_origin = None
        self.full_xlim = None
        self.full_ylim = None
        self._make_ui()
        self._attach_tooltips()
        self.after(100,self._show_splash)
        self.bind('<Control-Next>',lambda e:self.shift_file(1))
        self.bind('<Control-Prior>',lambda e:self.shift_file(-1))
        self.bind('<Control-o>', lambda e: self.open_file())
        self.bind('<Control-p>', lambda e: self.propose_phase())
        self.bind('<Control-s>', lambda e: self.save_project())
        self.bind('<Control-z>', lambda e: self.undo())
        self.bind('<Delete>', lambda e: self.delete_selected())
        self.bind('<BackSpace>', lambda e: self.delete_selected())
        self.protocol('WM_DELETE_WINDOW', self.on_close)
        if initial_file:
            self.after(200, lambda: self.load_file(initial_file))

    def _make_ui(self) -> None:
        style = ttk.Style(self)
        font_main = ('Helvetica Neue', 10) if sys.platform == 'darwin' else ('Segoe UI', 10)
        font_emphasis = ('Helvetica Neue', 10, 'bold') if sys.platform == 'darwin' else ('Segoe UI', 10, 'bold')
        if 'clam' in style.theme_names(): style.theme_use('clam')
        style.configure('TFrame', background='#f3f6fb')
        style.configure('TLabel', background='#f3f6fb', foreground='#243b53', font=font_main)
        style.configure('TLabelframe', background='#f8fafc', relief='flat', bordercolor='#dbe4ef')
        style.configure('TLabelframe.Label', background='#f3f6fb', foreground='#164a73', font=font_emphasis)
        style.configure('TRadiobutton', background='#f8fafc', font=font_main)
        style.configure('TCheckbutton', background='#f8fafc', font=font_main)
        style.configure('TButton', padding=(12, 8), font=font_main, background='#e9f0f7', foreground='#183953', borderwidth=0)
        style.map('TButton', background=[('active', '#d8e8f5'), ('pressed', '#cadff0')])
        style.configure('Primary.TButton', padding=(12, 8), background='#146494', foreground='white', font=font_emphasis)
        style.map('Primary.TButton', background=[('active','#2581b6'), ('pressed','#0d527f')], foreground=[('disabled','#dbeafe')])
        style.configure('Warn.TButton', padding=(10, 7), background='#ffefdb', foreground='#955617', font=font_emphasis)
        style.map('Warn.TButton', background=[('active','#ffe1ba')])
        style.configure('Treeview', rowheight=27, font=font_main, background='#ffffff', fieldbackground='#ffffff', foreground='#213547', borderwidth=0)
        style.configure('Treeview.Heading', font=font_emphasis, background='#e3ecf5', foreground='#1c405b', padding=(4, 6))
        style.map('Treeview', background=[('selected', '#cde3f5')], foreground=[('selected','#16354d')])
        style.configure('TNotebook.Tab', padding=(10,6))
        menu = tk.Menu(self)
        fm = tk.Menu(menu, tearoff=0)
        fm.add_command(label='Открыть SEG-Y...  Ctrl+O', command=self.open_file)
        fm.add_command(label='Добавить папку с SGY...',command=self.open_folder)
        fm.add_command(label='Открыть проект / сеанс...', command=self.open_project)
        fm.add_command(label='Импорт SGY + TXT (восстановить пики)...',command=self.import_from_txt)
        fm.add_command(label='Сохранить все сейсмограммы...', command=self.save_workspace)
        fm.add_command(label='Сохранить проект...  Ctrl+S', command=self.save_project)
        fm.add_separator()
        fm.add_command(label='Экспорт TXT...', command=lambda: self.export('txt'))
        fm.add_command(label='Общий TXT по всем SGY...', command=self.export_all)
        fm.add_command(label='Экспорт CSV...', command=lambda: self.export('csv'))
        fm.add_command(label='Сохранить изображение...', command=self.export_image)
        fm.add_separator(); fm.add_command(label='Выход', command=self.on_close)
        menu.add_cascade(label='Файл', menu=fm)
        edit = tk.Menu(menu, tearoff=0)
        edit.add_command(label='Отменить  Ctrl+Z', command=self.undo)
        edit.add_command(label='Удалить выбранную точку  Del', command=self.delete_selected)
        edit.add_command(label='Удалить все точки', command=self.clear_picks)
        edit.add_separator()
        edit.add_command(label='Проверить непрерывность волны', command=self.audit_phase)
        edit.add_command(label='Проследить фазу между опорами  Ctrl+P', command=self.propose_phase)
        menu.add_cascade(label='Правка', menu=edit)
        info = tk.Menu(menu, tearoff=0)
        info.add_command(label='Инструкция', command=self.help_dialog)
        info.add_command(label='О программе', command=lambda: messagebox.showinfo('О программе',
            'Pic Editor (PE) · SEG-Y Picker v'+__version__+'\nТочное ручное пикирование по отсчётам SEG-Y.\nПрототип; необходима дополнительная проверка перед промышленным применением.'))
        menu.add_cascade(label='Помощь', menu=info)
        self.config(menu=menu)

        brand = tk.Frame(self, bg='#103b5a', height=48)
        brand.pack(fill='x')
        tk.Label(brand, text='◈  PE  /  PIC EDITOR', bg='#103b5a', fg='#ffffff',
                 font=('Helvetica Neue', 14, 'bold') if sys.platform == 'darwin' else ('Segoe UI', 13, 'bold'),
                 padx=16).pack(side='left', pady=11)
        tk.Label(brand, text=f'VERSION {__version__}   •   ПИКИРОВАНИЕ И ГОДОГРАФЫ',
                 bg='#103b5a', fg='#b4d2e8', font=font_main, padx=14).pack(side='right')
        bar = ttk.Frame(self, padding=(13,9))
        bar.pack(fill='x')
        ttk.Button(bar, text='📂  Открыть SGY', style='Primary.TButton', command=self.open_file).pack(side='left', padx=(0,5))
        ttk.Button(bar, text='Сохранить проект', command=self.save_project).pack(side='left', padx=5)
        ttk.Button(bar, text='Экспорт TXT', command=lambda: self.export('txt')).pack(side='left', padx=5)
        ttk.Button(bar, text='Импорт TXT', command=self.import_from_txt).pack(side='left', padx=5)
        ttk.Button(bar, text='Общий TXT', command=self.export_all).pack(side='left', padx=5)
        ttk.Separator(bar, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Button(bar, text='Отменить', command=self.undo).pack(side='left', padx=3)
        ttk.Button(bar, text='Удалить точку', command=self.delete_selected).pack(side='left', padx=3)
        viewbar=ttk.Frame(self,padding=(13,0,13,4));viewbar.pack(fill='x')
        ttk.Label(viewbar, text='  Усиление', foreground='#334155').pack(side='left', padx=(14,3))
        ttk.Button(viewbar, text='−', width=2, command=lambda: self.change_gain(-0.1)).pack(side='left',padx=(2,0))
        self.gain_slider = ttk.Scale(viewbar, from_=0.2, to=3, variable=self.gain_var,
                  orient='horizontal', length=95, command=lambda _v: self.request_redraw())
        self.gain_slider.pack(side='left')
        ttk.Button(viewbar, text='+', width=2, command=lambda: self.change_gain(0.1)).pack(side='left',padx=(0,2))
        ttk.Label(viewbar,textvariable=self.gain_label_var,foreground='#475569').pack(side='left',padx=(1,2))
        ttk.Checkbutton(viewbar, text='AGC', variable=self.agc_var, command=self.request_redraw).pack(side='left', padx=8)
        ttk.Checkbutton(viewbar, text='Цвет', variable=self.color_var, command=self.request_redraw).pack(side='left')
        ttk.Checkbutton(viewbar, text='Проекции', variable=self.show_guides_var, command=self.request_redraw).pack(side='left',padx=8)
        ttk.Checkbutton(viewbar, text='Предупреждать о скачке',
                        variable=self.phase_warning_var).pack(side='left',padx=8)
        ttk.Button(viewbar, text='Сброс масштаба', command=self.reset_view).pack(side='right')

        # A separate thin navigation row remains usable on 1100-pixel displays.
        files_bar=ttk.Frame(self, padding=(13,0,13,7));files_bar.pack(fill='x')
        ttk.Label(files_bar,text='СЕЙСМОГРАММА',foreground='#164a73').pack(side='left',padx=(0,8))
        ttk.Button(files_bar,text='◀',width=3,command=lambda:self.shift_file(-1)).pack(side='left',padx=2)
        self.file_select_var=tk.StringVar()
        self.file_select=ttk.Combobox(files_bar,textvariable=self.file_select_var,state='readonly',width=55)
        self.file_select.pack(side='left',padx=4,fill='x',expand=True)
        self.file_select.bind('<<ComboboxSelected>>',self._selector_changed)
        ttk.Button(files_bar,text='▶',width=3,command=lambda:self.shift_file(1)).pack(side='left',padx=2)
        self.file_counter=ttk.Label(files_bar,text='0 / 0',foreground='#475569')
        self.file_counter.pack(side='left',padx=(8,0))
        ttk.Label(files_bar,text='Колесо: XY · Shift+колесо: Y · Ctrl+колесо: X · ЛКМ по фону: перетаскивание',
                  foreground='#64748b').pack(side='right',padx=(14,0))

        self.worktabs=ttk.Notebook(self)
        self.worktabs.pack(fill='both',expand=True,padx=8,pady=(0,4))
        main_page=ttk.Frame(self.worktabs)
        self.hodo_page=ttk.Frame(self.worktabs)
        self.worktabs.add(main_page,text='  Сейсмограмма  ')
        self.worktabs.add(self.hodo_page,text='  Годограф — развёрнутый  ')
        body = ttk.Panedwindow(main_page, orient='horizontal'); body.pack(fill='both',expand=True, padx=2,pady=(0,4))
        plot_frame = ttk.Frame(body)
        body.add(plot_frame, weight=5)
        tools = ttk.Frame(body, padding=(15,12))
        body.add(tools, weight=2)

        self.fig = Figure(figsize=(12,7), dpi=100, facecolor='#ffffff')
        self.ax = self.fig.add_axes([0.07,0.075,0.91,0.89])
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side='top',fill='both',expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.ax.text(.5,.5,'Откройте .SGY / .SEGY файл\nдля отображения сейсмограммы',
                     transform=self.ax.transAxes, ha='center',va='center',fontsize=15,color='#64748b')
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.canvas.draw_idle()

        hodo_frame=ttk.LabelFrame(tools,text='ГОДОГРАФ · все загруженные SGY',padding=(4,3))
        hodo_frame.pack(fill='x',pady=(0,6))
        self.hodo_mini_fig=Figure(figsize=(4,1.95),dpi=95,facecolor='#ffffff')
        self.hodo_mini_ax=self.hodo_mini_fig.add_axes([.12,.18,.84,.75])
        self.hodo_mini_canvas=FigureCanvasTkAgg(self.hodo_mini_fig,master=hodo_frame)
        self.hodo_mini_canvas.get_tk_widget().pack(fill='x')
        ttk.Button(hodo_frame,text='⤢  Развернуть годограф',command=self.open_full_hodo).pack(anchor='e',pady=(2,0))

        self.hodo_full_fig=Figure(figsize=(10,7),dpi=100,facecolor='white')
        self.hodo_full_ax=self.hodo_full_fig.add_axes([.1,.12,.85,.82])
        ttk.Button(self.hodo_page,text='◀  Вернуться к сейсмограмме',
                   command=lambda:self.worktabs.select(0)).pack(anchor='w',padx=8,pady=5)
        self.hodo_full_canvas=FigureCanvasTkAgg(self.hodo_full_fig,master=self.hodo_page)
        self.hodo_full_canvas.get_tk_widget().pack(fill='both',expand=True)
        for canv,ax in [(self.hodo_mini_canvas,self.hodo_mini_ax),
                        (self.hodo_full_canvas,self.hodo_full_ax)]:
            canv.mpl_connect('button_press_event',lambda evt,c=canv,a=ax:self.hodo_press(evt,c,a))
            canv.mpl_connect('motion_notify_event',lambda evt,c=canv,a=ax:self.hodo_motion(evt,c,a))
            canv.mpl_connect('button_release_event',lambda evt,c=canv,a=ax:self.hodo_release(evt,c,a))

        wave_row=ttk.Frame(tools);wave_row.pack(fill='x',pady=(0,4))
        ttk.Label(wave_row,text='Волна:').pack(side='left',padx=(0,5))
        self.wave_selector=ttk.Combobox(wave_row,state='readonly',textvariable=self.wave_code_var,width=17)
        self.wave_selector.pack(side='left',fill='x',expand=True)
        self.wave_selector.bind('<<ComboboxSelected>>',self._on_wave_selected)
        ttk.Button(wave_row,text='+ Волна',command=self.create_wave_dialog).pack(side='left',padx=3)
        ttk.Button(wave_row,text='Палитра',command=self.edit_wave_dialog).pack(side='left',padx=3)
        self._refresh_wave_selector()

        self.settings_toggle=ttk.Button(tools,text='⚙ Настройки пикирования  ▾',command=self.toggle_settings)
        self.settings_toggle.pack(fill='x',pady=(0,3))
        self.settings_holder=ttk.Frame(tools)
        self.control_tabs=ttk.Notebook(self.settings_holder)
        self.control_tabs.pack(fill='x',pady=(0,5))
        pick_page=ttk.Frame(self.control_tabs,padding=(6,7))
        qc_page=ttk.Frame(self.control_tabs,padding=(6,7))
        track_page=ttk.Frame(self.control_tabs,padding=(6,7))
        self.control_tabs.add(pick_page,text='  Выбор пика  ')
        self.control_tabs.add(track_page,text='  Автофаза  ')
        self.control_tabs.add(qc_page,text='  Проверка фазы  ')
        modes=ttk.LabelFrame(pick_page, text='Режим мыши',padding=7); modes.pack(fill='x',pady=(0,6))
        ttk.Radiobutton(modes, text='Поставить пик (левая кнопка)',variable=self.mode_var,value='pick').pack(anchor='w')
        ttk.Radiobutton(modes, text='Удалить пик (левая кнопка)',variable=self.mode_var,value='erase').pack(anchor='w')
        ttk.Label(modes, text='Перетаскивание: переместить существующий пик',foreground='#64748b').pack(anchor='w')
        ttk.Label(modes, text='Правая кнопка: удалить точку',foreground='#64748b').pack(anchor='w')
        peak=ttk.LabelFrame(pick_page, text='Привязка к экстремуму',padding=7);peak.pack(fill='x',pady=(0,4))
        ttk.Radiobutton(peak,text='Положительный пик (красный)',variable=self.pol_var,value='positive').pack(anchor='w')
        ttk.Radiobutton(peak,text='Отрицательный пик (синий)',variable=self.pol_var,value='negative').pack(anchor='w')
        ttk.Radiobutton(peak,text='Оба варианта',variable=self.pol_var,value='both').pack(anchor='w')
        ttk.Checkbutton(peak,text='Только выбранная трасса (из таблицы)',variable=self.lock_trace_var).pack(anchor='w',pady=(5,0))
        row=ttk.Frame(peak); row.pack(fill='x',pady=(7,0))
        ttk.Label(row,text='Окно поиска, мс:').pack(side='left')
        ttk.Entry(row,textvariable=self.radius_var,width=7).pack(side='left',padx=6)
        control = ttk.LabelFrame(qc_page, text='Контроль выбора фазы', padding=9)
        control.pack(fill='x', pady=(0,9))
        ttk.Checkbutton(control, text='Предупреждать о возможной смене фазы',
                        variable=self.phase_warning_var).pack(anchor='w')
        tol = ttk.Frame(control);tol.pack(fill='x',pady=(6,3))
        ttk.Label(tol, text='Допустимое отклонение, мс:').pack(side='left')
        ttk.Entry(tol, textvariable=self.phase_tolerance_var, width=7).pack(side='left',padx=8)
        ttk.Button(control, text='Проверить все отмеченные пики',style='Warn.TButton',
                   command=self.audit_phase).pack(fill='x',pady=(5,0))
        ttk.Label(control, text='Проверка предупреждает, но не исправляет пики автоматически.',
                  foreground='#718096', wraplength=320).pack(anchor='w',pady=(5,0))
        track_box=ttk.LabelFrame(track_page,text='Автоматическое прослеживание',padding=8)
        track_box.pack(fill='x',pady=(0,7))
        desc=ttk.Label(track_box,text='Сначала укажите два и более пика одной фазы.\nАлгоритм предложит точки между опорами.',foreground='#52697c')
        desc.pack(anchor='w',pady=(0,5))
        rr=ttk.Frame(track_box);rr.pack(fill='x',pady=(0,5))
        ttk.Label(rr,text='Коридор, ±мс:').pack(side='left')
        ttk.Entry(rr,textvariable=self.tracking_radius_var,width=7).pack(side='left',padx=8)
        ttk.Button(track_box,text='▶  Проследить фазу между опорами  Ctrl+P',
                   style='Primary.TButton',command=self.propose_phase).pack(fill='x',pady=2)
        ttk.Checkbutton(track_box,text='Автоинтерполяция между опорами',variable=self.auto_interp_var).pack(anchor='w',pady=4)
        ttk.Label(track_box,text='Оценка качества — ориентир для проверки, не вероятность.',
                  foreground='#64748b',wraplength=320).pack(anchor='w',pady=(3,0))
        block=ttk.LabelFrame(tools,text='Расчёт и экспорт',padding=7);block.pack(fill='x',pady=(0,7))
        ttk.Button(block,text='Линейное предложение (старый режим)',command=self.propose).pack(fill='x',pady=2)
        ttk.Button(block,text='Подтвердить предложенные точки',command=self.accept_proposals).pack(fill='x',pady=2)
        ttk.Button(block,text='Рассчитать и показать таблицу',style='Primary.TButton',command=self.calculate).pack(fill='x',pady=(8,3))
        ttk.Checkbutton(block,text='Показывать все приёмники в таблице',variable=self.show_all_var,command=self.update_table).pack(anchor='w',pady=(6,0))
        self.count_label=ttk.Label(block,text='Подтверждено: 0  •  Предложено: 0',foreground='#475569')
        self.count_label.pack(anchor='w',pady=(6,0))

        results_head=ttk.Frame(tools);results_head.pack(fill='x',pady=(2,5))
        ttk.Label(results_head,text='РЕЗУЛЬТАТЫ · ДВОЙНОЙ КЛИК ДЛЯ ПРАВКИ',
                  font=('Segoe UI',9,'bold'),foreground='#174569').pack(side='left')
        ttk.Label(results_head,text='  Время:').pack(side='left',padx=(6,2))
        self.units_picker=ttk.Combobox(results_head,textvariable=self.unit_var,
                                       values=('мс','мкс'),state='readonly',width=4)
        self.units_picker.pack(side='left')
        self.units_picker.bind('<<ComboboxSelected>>',lambda _e:self._refresh_time_units())
        tree_frame = ttk.Frame(tools); tree_frame.pack(fill='both',expand=True)
        columns=('id','x','time','kind','quality')
        self.table=ttk.Treeview(tree_frame,columns=columns,show='headings',selectmode='browse',height=12)
        for name,title,width in [('id','RCV ID',56),('x','RCV X, м',69),('time','Travel time, мс',106),('kind','Статус',78),('quality','Оценка',85)]:
            self.table.heading(name,text=title)
            self.table.column(name,width=width,anchor='center',minwidth=58)
        self.table.tag_configure('manual',background='#e9fff0')
        self.table.tag_configure('suggest',background='#fff7d9')
        self.table.tag_configure('warning',background='#fff1de')
        self.table.tag_configure('weak',background='#ffeaeb')
        self.table.pack(side='left',fill='both',expand=True)
        scroll=ttk.Scrollbar(tree_frame,orient='vertical',command=self.table.yview)
        scroll.pack(side='right',fill='y');self.table.configure(yscrollcommand=scroll.set)
        self.table.bind('<<TreeviewSelect>>',self.select_table_row)
        self.table.bind('<Double-1>', self.edit_table_pick)
        status=ttk.Frame(self,padding=(10,5));status.pack(fill='x')
        ttk.Label(status,textvariable=self.info_var,foreground='#153c5a').pack(side='left')
        ttk.Label(status,textvariable=self.stats_var,foreground='#64748b').pack(side='right')

    def _show_splash(self):
        """Non-blocking startup brand splash; does not affect data handling."""
        try:
            window=tk.Toplevel(self)
            window.overrideredirect(True)
            window.configure(bg='#08192b')
            w,h=300,340
            x=self.winfo_rootx()+max(0,(self.winfo_width()-w)//2)
            y=self.winfo_rooty()+max(0,(self.winfo_height()-h)//2)
            window.geometry(f'{w}x{h}+{x}+{y}')
            window.lift()
            base=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent.parent))
            file=base/'assets'/'pic_editor_icon.png'
            self._splash_icon=tk.PhotoImage(file=str(file)) if file.exists() else None
            if self._splash_icon is not None:
                tk.Label(window,image=self._splash_icon,bg='#08192b').pack(padx=20,pady=(16,5))
            tk.Label(window,text='PIC EDITOR  ·  PE',bg='#08192b',fg='white',
                     font=('Segoe UI',15,'bold')).pack(pady=(1,5))
            tk.Label(window,text='SEG-Y  |  PICKING  |  HODOGRAPHS',bg='#08192b',
                     fg='#60d8ef',font=('Segoe UI',9)).pack()
            window.after(1150,window.destroy)
        except (OSError,tk.TclError):pass

    def _active_wave(self) -> str:
        return getattr(self,'current_wave_code','0')

    def _refresh_wave_selector(self):
        if not hasattr(self,'wave_selector'):return
        self._wave_selector_codes=list(self.sessions.wave_types)
        self.wave_selector['values']=[f'{code} · {self.sessions.wave_types[code].name}'
                                      for code in self._wave_selector_codes]
        code=self._active_wave()
        if code in self._wave_selector_codes:
            self.wave_selector.current(self._wave_selector_codes.index(code))

    def _on_wave_selected(self,_event=None):
        index=self.wave_selector.current()
        if 0<=index<len(self._wave_selector_codes):
            self.switch_wave(self._wave_selector_codes[index])

    def switch_wave(self,code:str):
        code=valid_code(code)
        if code not in self.sessions.wave_types:return
        if self.state_model and self.record and self.current_path:
            if self._active_wave()==code:
                self._refresh_wave_selector();return
            self._store_session()
        self.current_wave_code=code
        if self.record is not None and self.current_path in self.sessions.entries:
            session=self.sessions.entries[self.current_path]
            session.active_wave=code
            layer=session.layer(code)
            state=PickerState(self.record)
            state.picks=layer.picks.copy()
            state.proposals=layer.proposals.copy()
            state.proposal_info=layer.proposal_info.copy()
            state.failed_intervals=layer.failed_intervals
            self.state_model=state
            self.history=[snap.copy() for snap in layer.history]
            self.selected_trace=layer.selected_trace
            self.draw_plot(keep_view=True)
            self.update_table()
            self.refresh_hodographs()
            self.info_var.set(f'Активная волна: {code} — {self.sessions.wave_types[code].name}.')
        self._refresh_wave_selector()

    def _wave_dialog(self,edit:bool=False):
        current=self.sessions.wave_types.get(self._active_wave(),WaveType())
        dlg=tk.Toplevel(self)
        dlg.title('Палитра волны' if edit else 'Новая волна')
        dlg.geometry('430x310')
        dlg.transient(self)
        dlg.grab_set()
        holder=ttk.Frame(dlg,padding=18)
        holder.pack(fill='both',expand=True)
        code_var=tk.StringVar(value=current.code if edit else '')
        name_var=tk.StringVar(value=current.name if edit else '')
        colors={'line_color':current.line_color if edit else wave_color(len(self.sessions.wave_types)),
                'peak_color':current.peak_color,'guide_color':current.guide_color}
        ttk.Label(holder,text='Код Wave (цифры или буквы):').grid(row=0,column=0,sticky='w',pady=6)
        code_entry=ttk.Entry(holder,textvariable=code_var,width=24)
        code_entry.grid(row=0,column=1,sticky='ew',pady=6)
        if edit:code_entry.configure(state='disabled')
        ttk.Label(holder,text='Название волны:').grid(row=1,column=0,sticky='w',pady=6)
        ttk.Entry(holder,textvariable=name_var).grid(row=1,column=1,sticky='ew',pady=6)
        for row,(key,label) in enumerate([('line_color','Линия годографа'),
                                          ('peak_color','Точки пиков'),
                                          ('guide_color','Проекции')],start=2):
            ttk.Label(holder,text=label+':').grid(row=row,column=0,sticky='w',pady=6)
            btn=tk.Button(holder,text=colors[key],bg=colors[key],fg='black',width=16)
            btn.grid(row=row,column=1,sticky='ew',pady=4)
            def pick(k=key,b=btn):
                chosen=colorchooser.askcolor(color=colors[k],title='Выберите цвет',parent=dlg)[1]
                if chosen:
                    colors[k]=chosen
                    b.configure(bg=chosen,text=chosen)
            btn.configure(command=pick)
        holder.columnconfigure(1,weight=1)
        buttons=ttk.Frame(holder);buttons.grid(row=5,column=0,columnspan=2,sticky='e',pady=14)
        def save():
            try:
                code=valid_code(code_var.get())
                if not edit and code in self.sessions.wave_types:
                    raise ValueError('Такая волна уже существует. Укажите другой код.')
                wave=WaveType(code=code,name=name_var.get().strip(),**colors)
            except ValueError as exc:
                messagebox.showerror('Волна',str(exc),parent=dlg);return
            self.sessions.wave_types[code]=wave
            dlg.destroy()
            if edit:
                self.draw_plot(keep_view=True)
                self.refresh_hodographs()
                self._refresh_wave_selector()
            else:self.switch_wave(code)
        ttk.Button(buttons,text='Отмена',command=dlg.destroy).pack(side='right',padx=4)
        ttk.Button(buttons,text='Сохранить',style='Primary.TButton',command=save).pack(side='right',padx=4)
        code_entry.focus_set() if not edit else None
        self.wait_window(dlg)

    def create_wave_dialog(self):
        self._wave_dialog(False)

    def edit_wave_dialog(self):
        self._wave_dialog(True)

    def toggle_settings(self):
        if self._setting_visible:
            self.settings_holder.pack_forget()
            self.settings_toggle.configure(text='⚙ Настройки пикирования  ▾')
        else:
            self.settings_holder.pack(fill='x',after=self.settings_toggle,pady=(0,5))
            self.settings_toggle.configure(text='⚙ Настройки пикирования  ▴')
        self._setting_visible=not self._setting_visible

    def open_full_hodo(self):
        self.worktabs.select(self.hodo_page)
        self.after(70,self.refresh_hodographs)

    def refresh_hodographs(self):
        if not hasattr(self,'hodo_mini_ax'):return
        # Legacy v0.5 sessions did not persist trace geometry. Fill metadata lazily
        # for sessions that actually have confirmed picks, without decoding amplitudes.
        for session in self.sessions.entries.values():
            if session.geometry or not (session.picks or any(l.picks for l in session.wave_layers.values())):
                continue
            try:
                meta=read_segy_headers(session.path)
                session.validate_record(meta)
                session.geometry=[{'src_id':h.src_id,'src_x':h.src_x,'rcv_id':h.rcv_id,
                                   'rcv_x':h.rcv_x,'delay_ms':h.delay_ms} for h in meta.headers]
                session.dt_us=meta.dt_us
            except (OSError,ValueError,SegyError):
                continue
        series=from_workspace(self.sessions)
        for ax,canvas,mini in ((self.hodo_mini_ax,self.hodo_mini_canvas,True),
                               (self.hodo_full_ax,self.hodo_full_canvas,False)):
            ax.clear()
            ax.set_facecolor('white')
            for idx,curve in enumerate(series):
                wave=self.sessions.wave_types.get(curve.wave,WaveType())
                x=[p.rcv_x for p in curve.points]
                t=[p.time_ms for p in curve.points]
                active=(curve.path==self.current_path and curve.wave==self._active_wave())
                linestyle=('-' if active else ('--' if idx%2 else ':'))
                ax.plot(x,t,color=wave.line_color,linestyle=linestyle,
                        linewidth=2.0 if active else 1.2,alpha=1 if active else .75,
                        label=f'ПВ {curve.src_id} · {curve.wave} · {curve.path.name}' if not mini else None,
                        marker='o',markersize=4.5 if active else 2.8)
            ax.set_xlabel('RCV X, м',fontsize=8 if mini else 11)
            ax.set_ylabel('Время, мс',fontsize=8 if mini else 11)
            ax.grid(True,linewidth=.5,alpha=.25)
            ax.invert_yaxis()
            ax.tick_params(labelsize=7 if mini else 10)
            if not series:
                ax.text(.5,.5,'Пики пока не отмечены',ha='center',va='center',
                        color='#6b7b87',transform=ax.transAxes,fontsize=9)
            elif not mini:
                labels=[]
                for i,a in enumerate(series):
                    for b in series[i+1:]:
                        if a.wave!=b.wave:continue
                        result=geometry_relation(a,b)
                        if result in ('встречные','нагоняющие'):
                            labels.append(result)
                summary=(f'Встречные пары: {labels.count("встречные")}  ·  '
                         f'Нагоняющие пары: {labels.count("нагоняющие")}')
                ax.set_title('Общий годограф — '+summary+' (геометрия по SRC_X/RCV_X)',fontsize=11)
                if len(series)<=16:
                    ax.legend(loc='best',fontsize=7)
            canvas.draw_idle()
        self._hodo_series=series

    def _hodo_hit(self,event,ax):
        if event.inaxes is not ax or event.x is None or event.y is None:return None
        candidates=[p for curve in getattr(self,'_hodo_series',()) for p in curve.points]
        if not candidates:return None
        points=ax.transData.transform([(p.rcv_x,p.time_ms) for p in candidates])
        dists=np.hypot(points[:,0]-event.x,points[:,1]-event.y)
        idx=int(np.argmin(dists))
        return candidates[idx] if dists[idx]<=13 else None

    def hodo_press(self,event,canvas,ax):
        if event.button!=1:return
        hit=self._hodo_hit(event,ax)
        if hit is not None:
            self._hodo_drag=(hit,float(event.x),float(event.y),canvas,ax)

    def hodo_motion(self,event,canvas,ax):
        if self._hodo_drag is None or self._hodo_drag[3] is not canvas:return
        hit=self._hodo_drag[0]
        if event.ydata is not None:
            self.info_var.set(f'Годограф: SRC_ID={hit.src_id} · Wave={hit.wave} · '
                              f'X={hit.rcv_x:g} м · время → {event.ydata:.3f} мс')

    def hodo_release(self,event,canvas,ax):
        drag=self._hodo_drag
        self._hodo_drag=None
        if drag is None or drag[3] is not canvas:return
        point,x0,y0,_,_=drag
        moved=(event.x is not None and event.y is not None and
               np.hypot(float(event.x)-x0,float(event.y)-y0)>6)
        if not self.activate_file(point.path):return
        self.switch_wave(point.wave)
        self.selected_trace=point.trace_index
        if moved and event.ydata is not None:
            trace=point.trace_index
            estimated=int(round((float(event.ydata)-self.record.headers[trace].delay_ms)*1000/self.record.dt_us))
            radius=int(round(max(self._radius_ms(),4.0)*1000/self.record.dt_us))
            original=self.state_model.picks.get(trace)
            if original is None:return
            sign='positive' if self.display_signals[trace,original]>0 else 'negative'
            new=nearest_peak(self.display_signals[trace],estimated,radius,sign)
            if new is None:
                messagebox.showwarning('Годограф','Рядом с выбранным временем нет экстремума нужной полярности.')
                return
            if new!=original and self._approve_pick(trace,new):
                self._snapshot()
                self.state_model.set_pick(trace,new)
                self._changes(f'Пик исправлен на годографе и сейсмограмме: t={self.record.time_ms(trace,new):.3f} мс')
        else:
            # Highlight the relevant receiver and time on the seismic plot.
            width=self.ax.get_xlim()[1]-self.ax.get_xlim()[0]
            span=abs(self.ax.get_ylim()[1]-self.ax.get_ylim()[0])
            t=point.time_ms
            self.ax.set_xlim(point.rcv_x-width/2,point.rcv_x+width/2)
            self.ax.set_ylim(t+span/2,t-span/2)
            self.canvas.draw_idle()
            self.worktabs.select(0)
            self.info_var.set(f'Выбран пик на сейсмограмме: ПВ {point.src_id}, Wave={point.wave}, X={point.rcv_x:g} м')

    def _auto_propose_if_enabled(self):
        if not self.auto_interp_var.get() or not self.state_model or len(self.state_model.picks)<2:return
        try:
            count=self.state_model.suggest_between(radius_ms=self._radius_ms(),
                 polarity=self.pol_var.get(),signals=self.display_signals)
            self._changes(f'Автоинтерполяция: {count} предварительных точек. Проверьте и подтвердите.')
        except ValueError as exc:
            self.info_var.set(f'Интерполяция: {exc}')

    def import_from_txt(self):
        """Transactional geometry-checked import. Never silently skip missing SGY/rows."""
        txt=filedialog.askopenfilename(title='Выберите TXT с результатами пикирования',
                                       filetypes=[('Результаты TXT/CSV','*.txt *.csv'),('Все файлы','*.*')])
        if not txt:return
        files=filedialog.askopenfilenames(title='Выберите все SEG-Y, которые перечислены в TXT',
                     filetypes=[('SEG-Y','*.sgy *.segy *.SGY *.SEGY'),('Все файлы','*.*')])
        if not files:
            if not self.sessions.entries:return
            if not messagebox.askyesno('Импорт','Использовать уже загруженные SGY для сопоставления?'):
                return
            files=tuple(str(p) for p in self.sessions.entries)
        try:
            rows=read_pick_file(txt)
            indexes=[read_segy_headers(p) for p in files]
            matches=validate_pick_import(rows,indexes)
            n=sum(len(picks) for layers in matches.values() for picks in layers.values())
            if not n:raise ValueError('В TXT нет подтверждённых времён для импорта.')
        except (OSError,ValueError,SegyError) as exc:
            messagebox.showerror('Не удалось импортировать TXT',str(exc));return
        if not messagebox.askyesno('Импорт TXT',
                 f'Проверка завершена. SGY: {len(matches)}, точек: {n}.\n'
                 'Времена сопоставлены с геометрией и шагом дискретизации.\n\n'
                 'Добавить пики в открытый сеанс? Совпадающие точки будут заменены.'):
            return
        self._store_session()
        self.sessions.add(files)
        index_by_path={Path(r.path).resolve():r for r in indexes}
        for path,layers in matches.items():
            session=self.sessions.entries[path]
            rec=index_by_path[path]
            session.geometry=[{'src_id':h.src_id,'src_x':h.src_x,'rcv_id':h.rcv_id,
                               'rcv_x':h.rcv_x,'delay_ms':h.delay_ms} for h in rec.headers]
            session.n_traces=rec.n_traces
            session.n_samples=rec.n_samples
            session.dt_us=rec.dt_us
            session.file_size=Path(path).stat().st_size
            session.dirty=True
            for code,picks in layers.items():
                if code not in self.sessions.wave_types:
                    self.sessions.wave_types[code]=WaveType(code=code,name=f'Волна {code}',
                                              line_color=wave_color(len(self.sessions.wave_types)))
                if session.active_wave=='0' and not session.layer('0').picks:
                    session.active_wave=code
                layer=session.layer(code)
                layer.picks.update(picks)
                layer.proposals.clear();layer.proposal_info.clear()
        first=next(iter(matches))
        self.current_path=None
        self.record=None
        self.state_model=None
        if self.activate_file(first):
            self.info_var.set(f'Импортировано {n} пиков из {len(matches)} SGY; проверена геометрия и шаг времени.')
            self.refresh_hodographs()

    def _attach_tooltips(self):
        explanations={
          'Открыть SGY':'Загрузить одну или несколько сейсмограмм SEG-Y/SGY.',
          'Сохранить проект':'Сохранить все волны, точки и настройки сеанса для продолжения работы.',
          'Экспорт TXT':'Сохранить подтверждённые пики активной сейсмограммы и волны.',
          'Общий TXT':'Объединить пики всех волн и всех открытых SGY в один TXT.',
          'Импорт TXT':'Загрузить SGY и TXT, проверить идентификаторы и координаты и восстановить точки.',
          'Отменить':'Отменить последнее изменение пиков текущей волны.',
          'Удалить точку':'Удалить выбранный пик текущей волны.',
          'Проследить фазу':'Предложить промежуточные пики по согласованной форме сигнала; их нужно подтвердить.',
          'Сброс масштаба':'Показать весь временной и пространственный диапазон SGY.',
          'AGC':'Автоматически выравнивать усиление амплитуд по времени.',
          'Цвет':'Включить/выключить красно-синюю заливку сейсмических амплитуд.',
          'Проекции':'Показать/скрыть горизонтальные линии от пика к оси приёмника.',
          'Предупреждать о скачке':'Запрашивать подтверждение при подозрительной смене фазы.',
          '+ Волна':'Создать новую волну с собственным кодом, названием и цветами.',
          'Палитра':'Изменить название и цвета активной волны.',
          'Автоинтерполяция между опорами':'Предлагать пики между выбранными опорными точками; без подтверждения они не экспортируются.',
          'Развернуть годограф':'Открыть общий годограф во второй полноразмерной вкладке.',
          'Подтвердить предложенные точки':'Сделать проверенные автоматические точки подтверждёнными.',
          'Рассчитать и показать таблицу':'Показать времена по текущим подтверждённым пикам.',
          'Время:':'Выбрать отображение времени в миллисекундах или микросекундах.',
        }
        def walk(widget):
            for child in widget.winfo_children():
                if isinstance(child,(ttk.Button,ttk.Checkbutton,ttk.Radiobutton,ttk.Combobox,ttk.Scale)):
                    try:label=str(child.cget('text')).strip()
                    except tk.TclError:label=''
                    for prefix,text in explanations.items():
                        if prefix in label:
                            ToolTip(child,text);break
                    else:
                        if label:
                            ToolTip(child,f'{label}: используйте этот элемент для управления пикированием или просмотром.')
                        elif isinstance(child,ttk.Combobox):
                            ToolTip(child,'Выбрать доступный вариант из списка.')
                walk(child)
        walk(self)

    def change_gain(self, delta: float):
        """Nudge the same Tk variable that drives the slider (no alternative gain path)."""
        value=round(float(self.gain_var.get())+delta,2)
        self.gain_var.set(max(0.2,min(3.0,value)))
        self.request_redraw()

    def _refresh_selector(self):
        entries=list(self.sessions.entries)
        options=[f'{i+1:03d}  |  {path.name}' for i,path in enumerate(entries)]
        self.file_select['values']=options
        if self.current_path in self.sessions.entries:
            pos=entries.index(self.current_path)
            self.file_select.current(pos)
            self.file_counter.configure(text=f'{pos+1} / {len(entries)}')
        else:
            self.file_select_var.set('')
            self.file_counter.configure(text=f'0 / {len(entries)}')

    def _selector_changed(self, _event=None):
        i=self.file_select.current()
        names=list(self.sessions.entries)
        if 0<=i<len(names):self.activate_file(names[i])

    def shift_file(self, delta: int):
        names=list(self.sessions.entries)
        if not names:return
        index=names.index(self.current_path) if self.current_path in names else 0
        target=names[(index+delta)%len(names)]
        self.activate_file(target)

    def open_file(self) -> None:
        paths=filedialog.askopenfilenames(title='Открыть одну или несколько сейсмограмм SEG-Y',
            filetypes=[('SEG-Y','*.sgy *.segy *.SGY *.SEGY'),('Все файлы','*.*')])
        if not paths:return
        added=self.sessions.add(paths)
        self._refresh_selector()
        target=added[0] if added else Path(paths[0]).resolve()
        if self.activate_file(target):
            self.info_var.set(f'Загружено {len(added)} новых файлов, всего {len(self.sessions.entries)}. '
                              'Переключайтесь стрелками или списком вверху.')

    def open_folder(self):
        folder=filedialog.askdirectory(title='Папка с сейсмограммами SGY/SEG-Y')
        if not folder:return
        candidates=sorted(p for p in Path(folder).iterdir()
                          if p.is_file() and p.suffix.lower() in ('.sgy','.segy'))
        if not candidates:
            messagebox.showinfo('Нет SGY','В выбранной папке не найдены файлы SGY/SEG-Y.');return
        added=self.sessions.add(candidates)
        if not self.record:self.activate_file(candidates[0])
        self._refresh_selector()
        self.info_var.set(f'Добавлено {len(added)} файлов из папки; всего: {len(self.sessions.entries)}.')

    def export_all(self):
        """One merged TXT containing only confirmed picks, sourced from each SGY header."""
        self._store_session()
        ready=[entry for entry in self.sessions.entries.values() if any(layer.picks for layer in entry.wave_layers.values()) or entry.picks]
        if not ready:
            messagebox.showinfo('Общий экспорт','Нет SGY с подтверждёнными пиками.');return
        target=filedialog.asksaveasfilename(title='Сохранить общий TXT по всем сейсмограммам',
                 defaultextension='.txt', initialfile='PicEditor_all_traveltimes.txt',
                 filetypes=[('Текстовые файлы','*.txt')])
        if not target:return
        rows=[]
        failures=[]
        for session in ready:
            try:
                rec=read_segy(session.path)
                session.validate_record(rec)
                for code,layer in session.wave_layers.items():
                    rows.extend(rows_for_layer(rec,layer,code))
                if not session.wave_layers and session.picks:
                    rows.extend(rows_for_layer(rec,session.layer('0'),'0'))
            except (OSError,ValueError,SegyError) as exc:
                failures.append(f'{session.path.name}: {exc}')
        if failures:
            messagebox.showerror('Общий экспорт',
                  'Не удалось проверить все SGY. Файл не создан.\n\n'+'\n'.join(failures[:8]))
            return
        try:
            export_table(target,rows,delimiter='\t')
        except OSError as exc:
            messagebox.showerror('Общий экспорт',f'Не удалось сохранить файл: {exc}')
            return
        msg=f'Сохранён один TXT: {len(ready)} сейсмограмм, {len(rows)} подтверждённых пиков.'
        self.info_var.set(msg)
        messagebox.showinfo('Общий экспорт',msg+'\n\n'+str(target))

    def _store_session(self):
        """Store every editable UI field without holding extra SEG-Y arrays in RAM."""
        if self.current_path not in self.sessions.entries or not self.state_model:return
        session=self.sessions.entries[self.current_path]
        state=self.state_model
        code=self._active_wave()
        session.active_wave=code
        layer=WaveLayer(picks=state.picks.copy(),proposals=state.proposals.copy(),
                        proposal_info=state.proposal_info.copy(),failed_intervals=state.failed_intervals,
                        history=[h.copy() for h in self.history],selected_trace=self.selected_trace)
        session.wave_layers[code]=layer
        if code=='0':session.picks=state.picks.copy()
        session.geometry=[{'src_id':h.src_id,'src_x':h.src_x,'rcv_id':h.rcv_id,
                           'rcv_x':h.rcv_x,'delay_ms':h.delay_ms} for h in self.record.headers]
        session.proposals=state.proposals.copy()
        session.proposal_info=state.proposal_info.copy()
        session.failed_intervals=state.failed_intervals
        session.history=[h.copy() for h in self.history]
        session.selected_trace=self.selected_trace
        session.dirty=self.unsaved_changes
        session.gain=self.gain_var.get()
        session.agc=self.agc_var.get()
        session.color=self.color_var.get()
        session.n_traces=self.record.n_traces
        session.n_samples=self.record.n_samples
        session.dt_us=self.record.dt_us
        try: session.file_size=self.record.path.stat().st_size
        except OSError: session.file_size=None
        if self.ax.get_xlim()[1]>self.ax.get_xlim()[0] and self.ax.get_ylim()[0]>self.ax.get_ylim()[1]:
            session.view=(tuple(self.ax.get_xlim()), tuple(self.ax.get_ylim()))

    def activate_file(self, path: str | Path) -> bool:
        """Activate on demand. Failure leaves current file and its picks untouched."""
        dest=Path(path).resolve()
        if dest not in self.sessions.entries:self.sessions.add([dest])
        if self.current_path==dest and self.record is not None:
            self._refresh_selector();return True
        try:
            rec=read_segy(dest)
            session=self.sessions.entries[dest]
            session.validate_record(rec)
        except (OSError,SegyError,ValueError) as exc:
            messagebox.showerror('Ошибка чтения SEG-Y',f'{dest.name}\n\n{exc}')
            self._refresh_selector()
            return False
        self._store_session()
        if self.result_window is not None and self.result_window.winfo_exists():
            self.result_window.destroy()
        self.result_window=None
        self.current_path=dest
        self.record=rec
        if not session.geometry:
            session.geometry=[{'src_id':h.src_id,'src_x':h.src_x,'rcv_id':h.rcv_id,
                               'rcv_x':h.rcv_x,'delay_ms':h.delay_ms} for h in rec.headers]
            session.n_traces=rec.n_traces
            session.n_samples=rec.n_samples
            session.dt_us=rec.dt_us
        state=PickerState(rec)
        wave_code=session.active_wave if session.active_wave in self.sessions.wave_types else '0'
        self.current_wave_code=wave_code
        self.wave_code_var.set(wave_code)
        layer=session.layer(wave_code)
        state.picks=layer.picks.copy()
        state.proposals=layer.proposals.copy()
        state.proposal_info=layer.proposal_info.copy()
        state.failed_intervals=layer.failed_intervals
        self.state_model=state
        self.history=[h.copy() for h in layer.history]
        self.unsaved_changes=session.dirty
        self.selected_trace=layer.selected_trace
        self.gain_var.set(session.gain)
        self.agc_var.set(session.agc)
        self.color_var.set(session.color)
        self._signals_cache_key = None
        self._signals_cache = None
        self.display_signals = None
        self.display_x = None
        self.title(f'Pic Editor (PE)  •  {dest.name}  •  {len(self.sessions.entries)} файлов')
        coord=rec.receiver_x
        self.stats_var.set(f'{rec.n_traces} трасс  •  {rec.n_samples} отсчётов  •  '
                           f'{rec.dt_us/1000:g} мс  •  X: {coord.min():g}–{coord.max():g} м')
        self.draw_plot(keep_view=False)
        if session.view is not None:
            self.ax.set_xlim(*session.view[0]);self.ax.set_ylim(*session.view[1]);self.canvas.draw_idle()
        self._refresh_wave_selector();self.update_table();self._refresh_selector();self.refresh_hodographs()
        self.info_var.set(f'Открыта {dest.name}; подтверждено пиков: {len(state.picks)}.')
        return True

    def load_file(self, path: str) -> bool:
        """Backward-compatible API used by existing single-project loader/tests."""
        return self.activate_file(path)

    def on_scroll(self,event):
        """Mouse-centered zoom with wheel/trackpad; axes are in meters and milliseconds."""
        if not self.record or event.inaxes is not self.ax:return
        if event.xdata is None or event.ydata is None:return
        direction=1 if event.button=='up' else -1 if event.button=='down' else 0
        if not direction:return
        scale=1/1.22 if direction>0 else 1.22
        old_x=self.ax.get_xlim();old_y=self.ax.get_ylim()
        def zoom(limits, center, minimum):
            left,right=limits
            a=center+(left-center)*scale
            b=center+(right-center)*scale
            if abs(b-a)<minimum:return limits
            return a,b
        coords=self.record.receiver_x
        dx=float(np.median(np.diff(np.unique(coords)))) if len(np.unique(coords))>1 else 2.
        modifier=str(getattr(event,'key','') or '').lower()
        only_y='shift' in modifier
        only_x=('control' in modifier or 'ctrl' in modifier) and not only_y
        new_x=old_x if only_y else zoom(old_x,event.xdata,max(.1,abs(dx)*.5))
        new_y=old_y if only_x else zoom(old_y,event.ydata,self.record.dt_us/1000*5)
        if scale>1 and self.full_xlim is not None and abs(new_x[1]-new_x[0])>=abs(self.full_xlim[1]-self.full_xlim[0]):
            new_x=self.full_xlim
        if scale>1 and self.full_ylim is not None and abs(new_y[1]-new_y[0])>=abs(self.full_ylim[1]-self.full_ylim[0]):
            new_y=self.full_ylim
        self.ax.set_xlim(*new_x)
        self.ax.set_ylim(*new_y)
        self.canvas.draw_idle()

    def _refresh_time_units(self):
        self.update_table()
        if self.result_window is not None and self.result_window.winfo_exists():
            if hasattr(self,'_result_populate'):
                self._result_populate()

    def _time_display(self, t: float | None) -> str:
        if t is None:return '—'
        return f'{t*1000:.0f}' if self.unit_var.get()=='мкс' else f'{t:.3f}'

    def _normalized_signals(self) -> np.ndarray:
        assert self.record is not None
        key=(id(self.record), self.agc_var.get())
        if self._signals_cache_key == key and self._signals_cache is not None:
            return self._signals_cache
        y=self.record.amplitudes
        if self.agc_var.get():
            # Running RMS gain (sample spacing-aware, without amplifying empty sections arbitrarily).
            window=max(5,int(round(40_000/self.record.dt_us)))
            normalized=np.empty_like(y)
            for i,line in enumerate(y):
                squared=line*line
                cumulative=np.concatenate(([0.],np.cumsum(squared)))
                indexes=np.arange(len(line));lo=np.maximum(0,indexes-window//2)
                hi=np.minimum(len(line),indexes+window//2+1)
                rms=np.sqrt((cumulative[hi]-cumulative[lo]) / np.maximum(hi-lo,1))
                floor=np.percentile(rms,80)*0.08 + 1e-12
                normalized[i]=line/np.maximum(rms,floor)
            result = np.clip(normalized,-2.5,2.5).astype(np.float32)
            self._signals_cache_key, self._signals_cache = key, result
            return result
        scale=np.maximum(np.percentile(np.abs(y),98,axis=1),1e-12)
        result = np.clip(y/scale[:,None],-2.5,2.5).astype(np.float32)
        self._signals_cache_key, self._signals_cache = key, result
        return result

    def request_redraw(self) -> None:
        if self.record and not self._draw_pending:
            self._draw_pending=True
            self.after(100,self._delayed_redraw)

    def _delayed_redraw(self):
        self._draw_pending=False
        if self.record:self.draw_plot(keep_view=True)

    def draw_plot(self, keep_view: bool = True) -> None:
        rec=self.record;state=self.state_model
        if rec is None or state is None: return
        old_xlim=self.ax.get_xlim();old_ylim=self.ax.get_ylim()
        self.ax.clear()
        self.ax.set_facecolor('#ffffff')
        self.display_signals=self._normalized_signals()
        x=rec.receiver_x
        unique=np.unique(np.sort(x))
        step=float(np.median(np.diff(unique))) if len(unique)>1 else 2.
        step=max(step,0.001)
        self.display_x=x[:,None]+np.clip(self.display_signals*self.gain_var.get()*0.53*step,-2.25*step,2.25*step)
        for i in range(rec.n_traces):
            t=rec.times_ms(i)
            wavex=self.display_x[i]
            self.ax.axvline(x[i],color='#64748b',linewidth=.48,alpha=.45,zorder=0)
            if self.color_var.get():
                self.ax.fill_betweenx(t,x[i],wavex,where=self.display_signals[i]>0,
                    color='#f43f51',alpha=.49,linewidth=0,zorder=1)
                self.ax.fill_betweenx(t,x[i],wavex,where=self.display_signals[i]<0,
                    color='#2222bb',alpha=.57,linewidth=0,zorder=1)
            self.ax.plot(wavex,t,color='#101828',linewidth=.62,zorder=2,antialiased=True)
        # Inactive waves remain visible, but cannot be modified accidentally.
        session=self.sessions.entries.get(self.current_path)
        if session:
            for code,layer in session.wave_layers.items():
                if code==self._active_wave() or not layer.picks:continue
                wave=self.sessions.wave_types.get(code,WaveType())
                items=sorted(layer.picks.items(),key=lambda kv:x[kv[0]])
                self.ax.plot([x[i] for i,_ in items],
                             [rec.time_ms(i,s) for i,s in items],
                             color=wave.line_color,linewidth=1.3,linestyle='--',alpha=.6,zorder=8)
        wave=self.sessions.wave_types.get(self._active_wave(),WaveType())
        if state.proposals:
            prop_points = sorted(state.proposals.items())
            if len(prop_points)>1:
                self.ax.plot([x[i] for i,_ in prop_points],
                             [rec.time_ms(i,s) for i,s in prop_points],
                             color='#d99a28', linestyle=':', linewidth=1.2, alpha=.8, zorder=9)
            for i,s in prop_points:
                y=rec.time_ms(i,s); xx=x[i]
                detail=state.proposal_info.get(i)
                weak=(detail is not None and detail.quality=='проверить')
                self.ax.plot(xx,y,marker='o',color='#e15a55' if weak else '#d2a034',
                             markeredgecolor='#75390b',markeredgewidth=.6,
                             markersize=6 if weak else 4.5,zorder=10)
        warnings = audit_picks(rec, state.picks, self._tolerance_ms())
        pairs=sorted(state.picks.items(),key=lambda kv: x[kv[0]])
        if pairs:
            xx=[x[i] for i,_ in pairs]
            yy=[rec.time_ms(i,s) for i,s in pairs]
            self.ax.plot(xx,yy,color=wave.line_color,linewidth=2,alpha=.92,zorder=11)
            for i,s in pairs:
                yy_i=rec.time_ms(i,s);x_peak=self.display_x[i,s]
                if self.show_guides_var.get():
                    self.ax.plot([x[i],x_peak],[yy_i,yy_i],color=wave.guide_color,
                        linewidth=1.6,linestyle='--',zorder=12)
                self.ax.scatter([x_peak],[yy_i],s=36,c=wave.peak_color,edgecolors='#101828',linewidths=.8,zorder=13)
                self.ax.scatter([x[i]],[yy_i],s=17,c=wave.line_color,edgecolors='white',linewidths=.5,zorder=14)
                if i in warnings:
                    self.ax.scatter([x[i]],[yy_i],s=82,facecolors='none',edgecolors='#f28c28',linewidths=1.8,zorder=15)
        margin=step*1.5
        bounds=(float(x.min()-margin),float(x.max()+margin))
        all_t=[rec.time_ms(i,rec.n_samples-1) for i in range(rec.n_traces)]
        min_t=min(h.delay_ms for h in rec.headers);max_t=max(all_t)
        self.ax.set_xlabel('Координата приёмника X, м',fontsize=10)
        self.ax.set_ylabel('Время, мс',fontsize=10)
        self.ax.grid(axis='y',color='#c1d0e0',linewidth=.55,alpha=.6)
        self.ax.yaxis.set_major_locator(MultipleLocator(50))
        self.ax.tick_params(labelsize=9)
        self.ax.set_xlim(*bounds)
        self.ax.set_ylim(max_t,min_t)
        self.full_xlim=bounds
        self.full_ylim=(max_t,min_t)
        if keep_view:
            if old_xlim[1]>old_xlim[0] and old_ylim[0]>old_ylim[1]:
                self.ax.set_xlim(old_xlim);self.ax.set_ylim(old_ylim)
        self.canvas.draw_idle()

    def reset_view(self):
        if self.record:self.draw_plot(keep_view=False)

    def _snapshot(self):
        if not self.state_model:return
        self.history.append(copy.copy(self.state_model.picks))
        self.unsaved_changes=True
        if len(self.history)>100:self.history.pop(0)

    def undo(self):
        if not self.state_model or not self.history:return
        self.state_model.picks=self.history.pop()
        self.unsaved_changes=True
        self.state_model.clear_proposals()
        self._changes('Последнее действие отменено')

    def _changes(self, msg: str):
        self.unsaved_changes=True
        self.info_var.set(msg)
        self.update_table()
        self._store_session()
        self.draw_plot(keep_view=True)
        self.refresh_hodographs()

    def _radius_ms(self) -> float:
        try:
            v=float(self.radius_var.get().replace(',','.'))
            if not (0.25<=v<=50):raise ValueError
            return v
        except ValueError:
            raise ValueError('Окно поиска должно быть от 0,25 до 50 мс.')

    def _nearest_visual_peak(self, event, trace_only: int | None = None) -> tuple[int,int] | None:
        rec=self.record
        if rec is None or self.display_x is None:return None
        rad=self._radius_ms()
        pol=self.pol_var.get()
        choices=[]
        if trace_only is not None:
            indices=(trace_only,)
        elif self.lock_trace_var.get():
            if self.selected_trace is None:
                raise ValueError('Для привязки выберите приёмник в таблице справа.')
            indices=(self.selected_trace,)
        else:
            indices=range(rec.n_traces)
        for i in indices:
            row=rec.headers[i]
            guess=int(round((event.ydata-row.delay_ms)*1000/rec.dt_us))
            radius=int(round(rad*1000/rec.dt_us))
            candidate=extrema(self.display_signals[i],guess-radius,guess+radius,pol)
            if not len(candidate):continue
            pos=np.column_stack([self.display_x[i,candidate],rec.times_ms(i)[candidate]])
            pixels=self.ax.transData.transform(pos)
            dist=np.hypot(pixels[:,0]-event.x,pixels[:,1]-event.y)
            best=np.argmin(dist)
            choices.append((float(dist[best]),i,int(candidate[best])))
        if not choices:return None
        distance,i,s=min(choices)
        # A click must be near the displayed wave peak, not an empty part of the chart.
        if distance>30:return None
        return i,s

    def _tolerance_ms(self) -> float:
        try:
            value=float(self.phase_tolerance_var.get().replace(',','.'))
            if not 0.25 <= value <= 250:
                return 8.0
            return value
        except ValueError:
            return 8.0

    def _approve_pick(self, trace: int, sample: int) -> bool:
        if not self.phase_warning_var.get() or not self.state_model:
            return True
        concerns=check_phase_continuity(self.record, self.state_model.picks,
                                        trace, sample, self._tolerance_ms())
        if not concerns:
            return True
        h=self.record.headers[trace]
        t=self.record.time_ms(trace,sample)
        text='\n'.join('• '+v.message for v in concerns)
        return messagebox.askyesno('Проверка фазы — требуется подтверждение',
            f'Возможная смена фазы на приёмнике №{h.rcv_id}\n'
            f'X = {h.rcv_x:g} м  |  Travel time = {t:.2f} мс\n\n'
            f'{text}\n\n'
            'Вы уверены, что выбрали нужную фазу?\n'
            'Да — подтвердить пик; Нет — вернуться к выбору.',
            icon='warning', parent=self)

    def audit_phase(self):
        if not self.state_model or not self.state_model.picks:
            messagebox.showinfo('Контроль', 'Нет подтверждённых пиков для проверки.');return
        findings=audit_picks(self.record, self.state_model.picks, self._tolerance_ms())
        if not findings:
            messagebox.showinfo('Контроль фазы',
                f'Проверено {len(self.state_model.picks)} точек.\n'
                'Резких отклонений по выбранным критериям не обнаружено.\n\n'
                'Это не гарантирует правильность геофизической интерпретации.')
            return
        parts=[]
        for i,problems in findings.items():
            h=self.record.headers[i]
            parts.append(f'Приёмник {h.rcv_id}, X={h.rcv_x:g} м: '+
                         ' '.join(issue.message for issue in problems))
        messagebox.showwarning('Проверьте выбор фазы',
            f'Возможные отклонения: {len(findings)} из {len(self.state_model.picks)} точек.\n\n'
            + '\n\n'.join(parts[:12])
            + ('\n\n… и другие точки' if len(parts)>12 else '')
            + '\n\nПики не изменены. Перетащите подозрительные точки при необходимости.')
        self.update_table();self.draw_plot(keep_view=True)

    def _begin_drag(self, event) -> bool:
        if not self.state_model or self.mode_var.get() != 'pick' or event.button != 1:
            return False
        pairs=list(self.state_model.picks.items())
        if not pairs:return False
        xy=[]
        for i,s in pairs:
            tm=self.record.time_ms(i,s)
            xy.extend([(i,self.record.receiver_x[i],tm),
                       (i,self.display_x[i,s],tm)])
        coords=self.ax.transData.transform([[v[1],v[2]] for v in xy])
        ds=np.hypot(coords[:,0]-event.x,coords[:,1]-event.y)
        k=int(np.argmin(ds))
        if ds[k]>11:return False
        self.dragging_trace=int(xy[k][0])
        self.drag_press_y=event.y
        self.drag_active=False
        self._motion_last=None
        xx=self.record.receiver_x[self.dragging_trace]
        self.drag_artist,=self.ax.plot([xx],[xy[k][2]],'o',markersize=9,
                                    color='#ffd43b',markeredgecolor='#0f3958',zorder=30)
        self.canvas.draw_idle()
        return True

    def on_release(self,event):
        if self._pan_origin is not None:
            self._pan_origin=None
            self._pan_active=False
            return
        if self._region_start is not None:
            x0,y0=self._region_start
            origin=self._region_origin
            self._region_start=None
            self._region_origin=None
            if self._region_patch is not None:
                self._region_patch.remove();self._region_patch=None
            if event.inaxes is self.ax and event.xdata is not None and event.ydata is not None:
                if origin is not None and np.hypot(event.x-origin[0],event.y-origin[1])>=12:
                    x1,y1=float(event.xdata),float(event.ydata)
                    if abs(x1-x0)>.05 and abs(y1-y0)>self.record.dt_us/1000*4:
                        self.ax.set_xlim(min(x0,x1),max(x0,x1))
                        self.ax.set_ylim(max(y0,y1),min(y0,y1))
            self.canvas.draw_idle()
            return
        trace=self.dragging_trace
        if trace is None:return
        was_active=self.drag_active
        self.dragging_trace=None
        self.drag_press_y=None
        self.drag_active=False
        if self.drag_artist is not None:
            try:self.drag_artist.remove()
            except ValueError:pass
            self.drag_artist=None
        if not was_active or event.inaxes is not self.ax or event.ydata is None:
            self.canvas.draw_idle();return
        try:
            rad=self._radius_ms()
            guess=int(round((event.ydata-self.record.headers[trace].delay_ms)*1000/self.record.dt_us))
            radius=int(round(rad*1000/self.record.dt_us))
            polarity=self.pol_var.get()
            candidates=extrema(self.display_signals[trace], guess-radius, guess+radius,polarity)
            if not len(candidates):
                self.info_var.set('Не удалось перенести пик: рядом нет нужного экстремума.')
                self.canvas.draw_idle();return
            # Time is used for vertical dragging; waveform amplitude offset
            # is deliberately ignored because trace identity is fixed.
            nearest=int(candidates[np.argmin(abs(candidates-guess))])
            if nearest==self.state_model.picks.get(trace):
                self.canvas.draw_idle();return
            if not self._approve_pick(trace,nearest):
                self.canvas.draw_idle();return
            self._snapshot()
            self.state_model.set_pick(trace, nearest)
            self.selected_trace=trace
            self._changes(f'Пик приёмника {self.record.headers[trace].rcv_id} перенесён: '
                          f'{self.record.time_ms(trace, nearest):.2f} мс')
        except ValueError as exc:
            messagebox.showerror('Перемещение точки',str(exc))
            self.canvas.draw_idle()

    def edit_table_pick(self,event):
        if not self.record or not self.state_model:return
        selected=self.table.selection()
        if not selected:return
        i=int(selected[0])
        current=self.state_model.picks.get(i)
        if current is None:
            messagebox.showinfo('Редактирование','Чтобы поставить новый пик, нажмите на волну на графике.')
            return
        from tkinter import simpledialog
        value=simpledialog.askfloat('Изменить время',
            f'Приёмник №{self.record.headers[i].rcv_id}\n'
            'Введите приблизительное время в '+self.unit_var.get()+'.\n'
            'Значение будет привязано к ближайшему экстремуму:',
            initialvalue=(self.record.time_ms(i,current)*(1000 if self.unit_var.get()=='мкс' else 1)),parent=self)
        if value is None:return
        if self.unit_var.get()=='мкс':value/=1000
        guess=int(round((value-self.record.headers[i].delay_ms)*1000/self.record.dt_us))
        radius=int(round(self._radius_ms()*1000/self.record.dt_us))
        peak=extrema(self.display_signals[i], guess-radius, guess+radius, self.pol_var.get())
        if not len(peak):
            messagebox.showwarning('Нет экстремума','Рядом с указанным временем нужный экстремум не найден.');return
        s=int(peak[np.argmin(abs(peak-guess))])
        if s==current or not self._approve_pick(i,s):return
        self._snapshot();self.state_model.set_pick(i,s);self.selected_trace=i
        self._changes(f'Время приёмника {self.record.headers[i].rcv_id} изменено')

    def on_click(self,event):
        if not self.record or not self.state_model or event.inaxes is not self.ax:
            return
        if event.button==1 and 'space' in str(event.key or '').lower():
            self._pan_origin=(event.x,event.y,tuple(self.ax.get_xlim()),tuple(self.ax.get_ylim()))
            self._pan_active=False
            return
        if event.button==1 and event.key and 'shift' in str(event.key).lower():
            self._region_start=(float(event.xdata),float(event.ydata))
            self._region_origin=(event.x,event.y)
            self._region_patch=Rectangle(self._region_start,0,0,fill=False,
                                         edgecolor='#1671aa',linewidth=1.5,linestyle='--',zorder=99)
            self.ax.add_patch(self._region_patch)
            return
        if event.button not in (1,3):return
        try:
            if event.button==3 or self.mode_var.get()=='erase':
                dist=np.abs(self.record.receiver_x-event.xdata)
                i=int(np.argmin(dist))
                self._snapshot();self.state_model.remove(i)
                self.selected_trace=i
                self._changes(f'Удалена точка приёмника №{self.record.headers[i].rcv_id}')
                return
            if self._begin_drag(event):
                return
            found=self._nearest_visual_peak(event)
            if found is None:
                self._pan_origin=(event.x,event.y,tuple(self.ax.get_xlim()),tuple(self.ax.get_ylim()))
                self._pan_active=False
                self.info_var.set('Перетащите пустой участок мышью для перемещения графика.')
                return
            i,s=found
            if not self._approve_pick(i,s):return
            self._snapshot();self.state_model.set_pick(i,s)
            self.selected_trace=i
            h=self.record.headers[i];travel=self.record.time_ms(i,s)
            self._changes(f'Пик: приёмник {h.rcv_id}, X={h.rcv_x:g} м, t={travel:.3f} мс, отсчёт {s}')
            self._auto_propose_if_enabled()
        except ValueError as e:
            messagebox.showerror('Параметры пика',str(e))

    def on_motion(self,event):
        if self._pan_origin is not None and event.inaxes is self.ax:
            px,py,xlim,ylim=self._pan_origin
            if event.x is not None and event.y is not None:
                if abs(event.x-px)+abs(event.y-py)>=5:self._pan_active=True
                if self._pan_active:
                    # Pixel translation maps into the current data viewport.
                    dx=-(event.x-px)*(xlim[1]-xlim[0])/max(self.ax.bbox.width,1)
                    dy=-(event.y-py)*(ylim[1]-ylim[0])/max(self.ax.bbox.height,1)
                    self.ax.set_xlim(xlim[0]+dx,xlim[1]+dx)
                    self.ax.set_ylim(ylim[0]-dy,ylim[1]-dy)
                    self.canvas.draw_idle()
            return
        if self._region_start is not None:
            if event.inaxes is self.ax and event.xdata is not None and event.ydata is not None:
                x0,y0=self._region_start
                self._region_patch.set_x(min(x0,event.xdata))
                self._region_patch.set_y(min(y0,event.ydata))
                self._region_patch.set_width(abs(event.xdata-x0))
                self._region_patch.set_height(abs(event.ydata-y0))
                self.canvas.draw_idle()
            return
        if self.dragging_trace is not None and event.inaxes is self.ax and event.ydata is not None:
            self._motion_last = event
            if self.drag_press_y is not None and abs(event.y - self.drag_press_y) >= 4:
                self.drag_active = True
                if self.drag_artist is not None:
                    yy = float(event.ydata)
                    xx = float(self.record.receiver_x[self.dragging_trace])
                    self.drag_artist.set_data([xx], [yy])
                    self.canvas.draw_idle()
            return
        if not self.record or event.inaxes is not self.ax or event.xdata is None:return
        if self.state_model:
            self.info_var.set(f'Указатель: X={event.xdata:.2f} м  |  t={event.ydata:.2f} мс  |  отмечено {len(self.state_model.picks)} трасс')

    def delete_selected(self):
        if not self.record or not self.state_model:return
        if self.selected_trace is None:
            messagebox.showinfo('Удаление','Выбери точку на графике или строку таблицы.');return
        self._snapshot();self.state_model.remove(self.selected_trace)
        self._changes(f'Точка трассы {self.selected_trace+1} удалена')

    def clear_picks(self):
        if not self.state_model or not self.state_model.picks:return
        if not messagebox.askyesno('Очистить','Удалить все подтверждённые пики?'):return
        self._snapshot();self.state_model.picks.clear();self.state_model.clear_proposals()
        self._changes('Все точки удалены')

    def propose(self):
        if not self.state_model:return
        if len(self.state_model.picks)<2:
            messagebox.showinfo('Недостаточно опорных точек','Отметь минимум два пика на разных трассах.');return
        try: count=self.state_model.suggest_between(self._radius_ms(),self.pol_var.get(),self.display_signals)
        except ValueError as e:
            messagebox.showerror('Параметры',str(e));return
        self._changes(f'Предложено {count} точек (жёлтые). Они не попадут в экспорт без подтверждения.')

    def propose_phase(self):
        if not self.state_model:return
        if len(self.state_model.picks)<2:
            messagebox.showinfo('Нужны опорные точки',
                'Отметь минимум два пика одной волны на разных трассах.')
            return
        try:
            radius=float(self.tracking_radius_var.get().strip().replace(',','.'))
            count=self.state_model.suggest_phase(radius, self.pol_var.get())
        except ValueError as exc:
            messagebox.showerror('Автопрослеживание',str(exc));return
        failed=self.state_model.failed_intervals
        weak=sum(d.quality=='проверить' for d in self.state_model.proposal_info.values())
        info=f'Автопрослеживание: {count} предложений, {weak} требуют особенно внимательной проверки.'
        if failed:
            info+=f' Не удалось проследить {len(failed)} интервал(ов) между опорами.'
        self._changes(info)
        if count==0 or failed:
            pairs=', '.join(f'{a+1}–{b+1}' for a,b in failed[:8])
            messagebox.showwarning('Результаты прослеживания',
                f'Найдено предложений: {count}.\n'
                f'Интервалы без непрерывной цепочки пиков: {pairs or "нет"}.\n\n'
                'Попробуй увеличить коридор, выбрать другую полярность '
                'или добавить ручные опорные точки. Недостающие времена не выдумываются.')

    def accept_proposals(self):
        if not self.state_model:return
        if not self.state_model.proposals:return
        count=len(self.state_model.proposals)
        weak=sum(d.quality=='проверить' for d in self.state_model.proposal_info.values())
        predicted=audit_picks(self.record, {**self.state_model.picks, **self.state_model.proposals}, self._tolerance_ms())
        if not messagebox.askyesno('Подтверждение',
              f'Вы визуально проверили {count} предложенных точек?\n'
              f'Оценка качества: {weak} сомнительных точек.\n'
              f'Проверка непрерывности: {len(predicted)} возможных отклонений.\n\n'
              'Принять предложенные точки для расчёта?'):
            return
        self._snapshot();self.state_model.accept();self._changes(f'Подтверждено дополнительных точек: {count}')

    def calculate(self):
        if not self.state_model:return
        self._store_session()
        n=sum(len(layer.picks) for layer in self.sessions.entries[self.current_path].wave_layers.values())
        self.update_table()
        if n==0:
            messagebox.showinfo('Расчёт','Пока нет подтверждённых точек. Выбери пики мышкой.')
            return
        self.show_results()

    def show_results(self):
        """Open a navigable 8-column results table; no file is written here."""
        if not self.state_model:return
        if self.result_window is not None and self.result_window.winfo_exists():
            self.result_window.destroy()
        window=tk.Toplevel(self)
        self.result_window=window
        window.title('Результаты пикирования · Travel time')
        ww, hh = 1080, 520
        self.update_idletasks()
        x = self.winfo_rootx() + max(0,(self.winfo_width()-ww)//2)
        y = self.winfo_rooty() + max(0,(self.winfo_height()-hh)//2)
        window.geometry(f'{ww}x{hh}+{x}+{y}')
        window.minsize(850,360)
        window.configure(bg='#f3f6fb')
        window.transient(self)
        window.lift()
        top=ttk.Frame(window,padding=(14,12));top.pack(fill='x')
        self._store_session()
        session=self.sessions.entries[self.current_path]
        layers=[(code,layer) for code,layer in session.wave_layers.items() if layer.picks]
        n=sum(len(layer.picks) for _,layer in layers)
        warnings=audit_picks(self.record,self.state_model.picks,self._tolerance_ms())
        ttk.Label(top,text=f'РЕЗУЛЬТАТЫ РАСЧЁТА  ·  {len(layers)} волн · {n} пиков',
                  font=('Helvetica Neue',12,'bold') if sys.platform=='darwin' else ('Segoe UI',12,'bold'),
                  foreground='#143f60').pack(anchor='w')
        ttk.Label(top,text=f'Точность исходного времени: {self.record.dt_us/1000:g} мс  |  '
                  f'Возможных несоответствий фазы: {len(warnings)}  |  '
                  'Пустое время означает отсутствие подтверждённого пика.',
                  foreground='#64748b').pack(anchor='w',pady=(5,0))
        unit_bar=ttk.Frame(window,padding=(14,0,14,6));unit_bar.pack(fill='x')
        ttk.Label(unit_bar,text='Отображение времени:').pack(side='left')
        ttk.Radiobutton(unit_bar,text='мс',variable=self.unit_var,value='мс',
                        command=self._refresh_time_units).pack(side='left',padx=(7,3))
        ttk.Radiobutton(unit_bar,text='мкс',variable=self.unit_var,value='мкс',
                        command=self._refresh_time_units).pack(side='left',padx=3)
        ttk.Label(unit_bar,text='Экспорт TXT/CSV остаётся в миллисекундах',
                  foreground='#64748b').pack(side='right')
        area=ttk.Frame(window,padding=(14,3));area.pack(fill='both',expand=True)
        keys=('SRC_ID','SRC_X','SRC_Z','RCV_ID','RCV_X','RCV_Z','Traveltime','Wave')
        tree=ttk.Treeview(area,columns=keys,show='headings',height=15)
        for key in keys:
            label=key if key!='Traveltime' else 'Traveltime, '+self.unit_var.get()
            tree.heading(key,text=label)
            tree.column(key,width=130 if key=='Traveltime' else 100,anchor='center',stretch=True)
        tree.tag_configure('warning',background='#fff1dc')
        yscroll=ttk.Scrollbar(area,orient='vertical',command=tree.yview)
        xscroll=ttk.Scrollbar(area,orient='horizontal',command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set,xscrollcommand=xscroll.set)
        tree.grid(row=0,column=0,sticky='nsew')
        yscroll.grid(row=0,column=1,sticky='ns')
        xscroll.grid(row=1,column=0,sticky='ew')
        area.rowconfigure(0,weight=1);area.columnconfigure(0,weight=1)
        show_all=tk.BooleanVar(value=False)
        def populate():
            tree.heading('Traveltime',text='Traveltime, '+self.unit_var.get())
            for child in tree.get_children():tree.delete(child)
            for code,layer in layers:
                for i,h in enumerate(self.record.headers):
                    if i not in layer.picks and not show_all.get():continue
                    tt=self.record.time_ms(i,layer.picks[i]) if i in layer.picks else None
                    values=(h.src_id,f'{h.src_x:g}',0,h.rcv_id,f'{h.rcv_x:g}',0,
                            self._time_display(tt) if tt is not None else '',code)
                    tree.insert('','end',values=values,
                                tags=('warning',) if code==self._active_wave() and i in warnings else ())
        self._result_populate=populate
        populate()
        foot=ttk.Frame(window,padding=(14,12));foot.pack(fill='x')
        ttk.Checkbutton(foot,text='Показывать все приёмники',variable=show_all,
                        command=populate).pack(side='left')
        def copy_table():
            rows=[row for code,layer in layers for row in rows_for_layer(self.record,layer,code,all_traces=show_all.get())]
            from .picking import _format
            header=list(keys)
            header[6]='Traveltime_us' if self.unit_var.get()=='мкс' else 'Traveltime_ms'
            copied=[]
            for row in rows:
                converted=list(row)
                converted[6] = float(converted[6])*1000 if converted[6] is not None and self.unit_var.get()=='мкс' else converted[6]
                copied.append('\t'.join(_format(v) for v in converted))
            text='\t'.join(header)+'\n'+'\n'.join(copied)
            window.clipboard_clear();window.clipboard_append(text)
            self.info_var.set(f'Скопировано {len(rows)} строк в буфер обмена')
        ttk.Button(foot,text='Копировать таблицу',command=copy_table).pack(side='right',padx=4)
        ttk.Button(foot,text='Экспорт CSV',command=lambda:self.export('csv')).pack(side='right',padx=4)
        ttk.Button(foot,text='Экспорт TXT',style='Primary.TButton',
                   command=lambda:self.export('txt')).pack(side='right',padx=4)
        ttk.Button(foot,text='Закрыть',command=window.destroy).pack(side='right',padx=4)
        window.protocol('WM_DELETE_WINDOW',window.destroy)
        if self.state_model.picks:
            first=str(min(self.state_model.picks))
            if self.table.exists(first):self.table.see(first)
        self.info_var.set(f'Рассчитано {n} значений Travel time; открыта таблица результатов.')

    def update_table(self):
        state=self.state_model
        for row in self.table.get_children():self.table.delete(row)
        if state is None:return
        self.table.heading('time',text='Travel time, '+self.unit_var.get())
        warnings = audit_picks(state.record, state.picks, self._tolerance_ms())
        for i,h in enumerate(state.record.headers):
            picked=i in state.picks; suggestion=i in state.proposals
            if not self.show_all_var.get() and not (picked or suggestion):continue
            sample=state.picks.get(i,state.proposals.get(i))
            time_ms=self._time_display(state.record.time_ms(i,sample) if sample is not None else None)
            detail=state.proposal_info.get(i)
            quality=detail.quality if detail is not None else '—'
            kind='⚠ Фаза?' if i in warnings else 'Готово' if picked else 'Предложено' if suggestion else '—'
            tag=('warning' if i in warnings else 'manual' if picked else
                 'weak' if detail and detail.quality=='проверить' else 'suggest' if suggestion else '')
            self.table.insert('', 'end',iid=str(i),
                              values=(h.rcv_id,f'{h.rcv_x:g}',time_ms,kind,quality),tags=(tag,))
        low=sum(d.quality=='проверить' for d in state.proposal_info.values())
        self.count_label.configure(text=f'Подтверждено: {len(state.picks)}  •  Предложено: {len(state.proposals)}'
                                        f'  •  ⚠ {len(warnings)}  •  слабых: {low}')

    def select_table_row(self,event):
        selected=self.table.selection()
        if selected:
            self.selected_trace=int(selected[0])
            if self.record:
                h=self.record.headers[self.selected_trace]
                self.info_var.set(f'Выбран приёмник №{h.rcv_id}, X={h.rcv_x:g} м')

    def save_workspace(self):
        """Save all opened files' picks, pending proposals and zoom in one JSON."""
        if not self.sessions.entries:return
        self._store_session()
        path=filedialog.asksaveasfilename(title='Сохранить все сейсмограммы',
            defaultextension='.stp-session.json',initialfile='seismic_session.stp-session.json',
            filetypes=[('Сеанс сейсмограмм (*.json)','*.json')])
        if not path:return
        try:
            self.sessions.save(path,self.current_path)
            self.workspace_path=Path(path).resolve()
            for entry in self.sessions.entries.values():entry.dirty=False
            self.unsaved_changes=False
            self.info_var.set(f'Сохранён сеанс из {len(self.sessions.entries)} SGY: {path}')
        except (OSError,ValueError) as exc:
            messagebox.showerror('Ошибка сохранения сеанса',str(exc))

    def save_project(self):
        if not self.state_model:return
        # Always use schema v3: one or more waves must not be lost on save.
        self.save_workspace()

    def open_project(self):
        path=filedialog.askopenfilename(title='Открыть проект или сеанс',filetypes=[('Проекты JSON','*.json')])
        if not path:return
        try:
            import json
            data=json.loads(Path(path).read_text(encoding='utf-8'))
            if data.get('schema_version') in (2,3):
                if not self.confirm_discard():return
                library,current=SessionLibrary.load(path)
                missing=[p.name for p in library.entries if not p.is_file()]
                if missing:
                    messagebox.showerror('Недоступные файлы',
                        'Не найдены исходные SEG-Y: '+', '.join(missing[:8])+
                        '. Разместите файлы по исходным путям и откройте сеанс снова.')
                    return
                # Validate target file before replacing the current workspace.
                if current is None:return
                record=read_segy(current)
                library.entries[current].validate_record(record)
                self.sessions=library
                self.current_path=None
                self.record=None
                self.state_model=None
                self.workspace_path=Path(path).resolve()
                self.activate_file(current)
                self.info_var.set(f'Восстановлен сеанс: {len(library.entries)} SGY')
                return
            source=data['source_path']
            if not Path(source).is_file():
                source=filedialog.askopenfilename(title='Укажите SEG-Y, соответствующий проекту',
                         filetypes=[('SEG-Y','*.sgy *.segy'),('Все','*.*')])
            if not source:return
            if not self.load_file(source):return
            self.state_model.load_project(path)
            self.history.clear()
            self.unsaved_changes=False
            self.sessions.entries[self.current_path].dirty=False
            self._changes('Проект загружен: '+path)
            self.unsaved_changes=False
        except (OSError,ValueError,KeyError,TypeError) as e:
            messagebox.showerror('Не удалось открыть проект',str(e))

    def export(self,fmt):
        if not self.record or not self.state_model:return
        self._store_session()
        session=self.sessions.entries[self.current_path]
        layers=[(code,layer) for code,layer in session.wave_layers.items() if layer.picks]
        if not layers:
            messagebox.showinfo('Экспорт','Сначала отметьте и подтвердите пики хотя бы одной волны.');return
        filetypes=[('Текстовый табличный файл','*.txt')] if fmt=='txt' else [('CSV','*.csv')]
        path=filedialog.asksaveasfilename(title='Экспорт всех волн текущего SGY',defaultextension='.'+fmt,
             filetypes=filetypes,initialfile=self.record.path.stem+'_traveltime.'+fmt)
        if not path:return
        all_rows=False
        # One-wave compatibility: optionally include unpicked traces with blank time.
        if len(layers)==1 and len(layers[0][1].picks)<self.record.n_traces:
            missing=self.record.n_traces-len(layers[0][1].picks)
            choice=messagebox.askyesnocancel('Незаполненные трассы',
                f'Не пропикированы {missing} приёмников.\n\n'
                'Да — все трассы, с пустыми Traveltime.\n'
                'Нет — только подтверждённые пики.\nОтмена — вернуться.')
            if choice is None:return
            all_rows=bool(choice)
        rows=[]
        for code,layer in layers:
            rows.extend(rows_for_layer(self.record,layer,code,all_rows))
        try:
            export_table(path,rows,delimiter='\t' if fmt=='txt' else ';')
            self.info_var.set(f'Экспорт: {len(rows)} строк, {len(layers)} волн: '+path)
            messagebox.showinfo('Готово',f'Сохранено {len(rows)} строк, волн: {len(layers)}.\n{path}')
        except OSError as exc:
            messagebox.showerror('Ошибка экспорта',str(exc))

    def export_image(self):
        if not self.record:return
        path=filedialog.asksaveasfilename(title='Сохранить изображение',defaultextension='.png',filetypes=[('PNG','*.png')])
        if not path:return
        try:self.fig.savefig(path,dpi=180,facecolor='white');self.info_var.set('Изображение сохранено: '+path)
        except OSError as e:messagebox.showerror('Ошибка',str(e))

    def confirm_discard(self) -> bool:
        self._store_session()
        dirty=sum(session.dirty for session in self.sessions.entries.values())
        if not dirty:return True
        return messagebox.askyesno('Несохранённые изменения',
            f'Есть несохранённые изменения в {dirty} сейсмограммах.\n'
            'Закрыть без сохранения? Для сохранения всех файлов выберите «Сохранить проект».')

    def on_close(self) -> None:
        if self.confirm_discard():
            self.destroy()

    def help_dialog(self):
        messagebox.showinfo('Pic Editor — краткая инструкция',
            '1. Откройте SGY (один или несколько) и выберите нужную сейсмограмму вверху.\n'
            '2. Колесо — зум XY, Shift+колесо — только время Y, Ctrl+колесо — только X.\n'
            '   ЛКМ на пустом поле + протягивание — перемещение; Shift+протягивание — рамка зума.\n'
            '3. «+ Волна» создаёт тип волны с кодом, названием и цветами.\n'
            '   Выберите волну и отмечайте соответствующие пики на трассах.\n'
            '4. «⚙ Настройки пикирования» открывает параметры и автоинтерполяцию.\n'
            '   Предложения требуют подтверждения — только тогда они попадают в экспорт.\n'
            '5. Справа годограф всех SGY: точку можно выбрать или перетащить по времени.\n'
            '   «Развернуть годограф» откроет его во второй вкладке.\n'
            '6. «Рассчитать» показывает таблицу. «Общий TXT» сохраняет все волны и SGY.\n'
            '   Время в TXT всегда в миллисекундах, поле Wave — выбранный код.\n'
            '7. «Импорт TXT» восстанавливает пики после проверки SGY и геометрии.\n'
            '   «Сохранить проект» сохраняет всё в JSON-сеансе.\n\n'
            'Наведение на кнопки показывает подсказки. Исходные SGY не изменяются.')


def main():
    import sys
    PickerApp(sys.argv[1] if len(sys.argv)>1 else None).mainloop()


if __name__=='__main__':
    main()
