"""
Parkolóhely-foglaltság Rendszer - Vékony Kliens Grafikus Felhasználói Felület (GUI)
=================================================================================
Kétlépéses hálózati munkafolyamat (távoli AI szerver feldolgozással):
1. Távoli AI szerver címének megadása, tanító videó kiválasztása és elküldése,
   a visszakapott eredményvideó megtekintése.
2. Tesztelő videó kiválasztása, távoli elemzés indítása a kapott munkamenettel,
   a feldolgozott videó és az Összes / Foglalt / Szabad statisztikák ellenőrzése.
"""

import os
import queue
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

from client_api import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    ParkingRemoteClient,
    RemoteClientError,
    ServerResponseError,
    ServerUnreachableError,
)


class ParkingAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parkolóhely-foglaltság Észlelő Rendszer")
        self.root.geometry("840x760")
        self.root.minsize(780, 680)
        self.root.configure(bg="#0f172a")

        # Állapotváltozók
        if os.path.isfile("video_input.mp4"):
            default_video = os.path.abspath("video_input.mp4")
        elif os.path.isfile("parking_input.mp4"):
            default_video = os.path.abspath("parking_input.mp4")
        else:
            default_video = ""
        self.train_video_path = tk.StringVar(value=default_video)
        self.test_video_path = tk.StringVar(value="")
        self.server_host = tk.StringVar(value=DEFAULT_SERVER_HOST)
        self.server_port = tk.StringVar(value=str(DEFAULT_SERVER_PORT))

        self.train_output_video = os.path.abspath("train_output.mp4")
        self.test_output_video = os.path.abspath("test_output.mp4")

        self.is_processing = False
        self.session_id: Optional[str] = None
        self._active_server_target: Optional[tuple[str, int]] = None
        self._is_checking_server = False
        self._health_check_seq = 0
        self._server_config_seq = 0
        self._train_result_ready = False
        self._test_result_ready = False
        self._is_closing = False
        self._ui_lock = threading.Lock()
        self._ui_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._ui_poll_after_id: Optional[str] = None
        self._ui_drain_after_id: Optional[str] = None
        self._ui_drain_scheduled = False
        self._ui_drain_gen = 0

        self._setup_styles()
        self._build_ui()

        self.server_host.trace_add("write", self._on_server_config_changed)
        self.server_port.trace_add("write", self._on_server_config_changed)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Destroy>", self._on_destroy)
        self._ui_poll_after_id = self.root.after(15, self._poll_ui_queue)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background="#0f172a")
        style.configure("Card.TFrame", background="#1e293b", relief="flat")
        style.configure("TLabel", background="#1e293b", foreground="#f8fafc", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#0f172a", foreground="#ffffff", font=("Segoe UI", 16, "bold"))
        style.configure("SubHeader.TLabel", background="#0f172a", foreground="#94a3b8", font=("Segoe UI", 9))
        style.configure("Section.TLabel", background="#1e293b", foreground="#38bdf8", font=("Segoe UI", 12, "bold"))
        style.configure("Muted.TLabel", background="#1e293b", foreground="#94a3b8", font=("Segoe UI", 9))

        style.configure(
            "Primary.TButton",
            background="#3b82f6",
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=6,
        )
        style.map("Primary.TButton", background=[("active", "#2563eb"), ("disabled", "#475569")])

        style.configure(
            "Success.TButton",
            background="#10b981",
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=6,
        )
        style.map("Success.TButton", background=[("active", "#059669"), ("disabled", "#475569")])

        style.configure(
            "Secondary.TButton",
            background="#334155",
            foreground="#ffffff",
            font=("Segoe UI", 9),
            padding=5,
        )
        style.map("Secondary.TButton", background=[("active", "#475569"), ("disabled", "#1e293b")])

        style.configure("TProgressbar", thickness=14, troughcolor="#0f172a", background="#3b82f6")

    def _build_ui(self):
        main_container = ttk.Frame(self.root, padding=12)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Címsor
        ttk.Label(
            main_container,
            text="🚗 Parkolóhely-foglaltság Detektáló Rendszer",
            style="Header.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            main_container,
            text="Vékony Kliens • Távoli AI Szerver Feldolgozás",
            style="SubHeader.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        # ==========================================
        # 0. KÁRTYA: TÁVOLI AI SZERVER BEÁLLÍTÁSA
        # ==========================================
        self.server_card = ttk.Frame(main_container, style="Card.TFrame", padding=10)
        self.server_card.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            self.server_card,
            text="🌐 SZERVER BEÁLLÍTÁSOK: Távoli AI Szerver",
            style="Section.TLabel",
        ).pack(anchor="w", pady=(0, 4))

        server_row = ttk.Frame(self.server_card, style="Card.TFrame")
        server_row.pack(fill=tk.X)

        ttk.Label(server_row, text="Szerver IP / Host:").pack(side=tk.LEFT, padx=(0, 6))
        self.server_host_entry = tk.Entry(
            server_row,
            textvariable=self.server_host,
            width=26,
            font=("Segoe UI", 9),
            bg="#0f172a",
            fg="#ffffff",
            insertbackground="#ffffff",
            disabledbackground="#1e293b",
            disabledforeground="#64748b",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#334155",
        )
        self.server_host_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 10))

        ttk.Label(server_row, text="Port:").pack(side=tk.LEFT, padx=(0, 6))
        self.server_port_entry = tk.Entry(
            server_row,
            textvariable=self.server_port,
            width=8,
            font=("Segoe UI", 9),
            bg="#0f172a",
            fg="#ffffff",
            insertbackground="#ffffff",
            disabledbackground="#1e293b",
            disabledforeground="#64748b",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#334155",
        )
        self.server_port_entry.pack(side=tk.LEFT, ipady=4, padx=(0, 10))

        self.btn_check_server = ttk.Button(
            server_row,
            text="🔌 Kapcsolat ellenőrzése",
            style="Secondary.TButton",
            command=self.check_server_health,
        )
        self.btn_check_server.pack(side=tk.LEFT)

        self.server_status_lbl = ttk.Label(
            self.server_card,
            text=f"Szerver státusz: Nincs ellenőrizve (http://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_PORT})",
            style="Muted.TLabel",
        )
        self.server_status_lbl.pack(anchor="w", pady=(4, 0))

        # ==========================================
        # 1. KÁRTYA: TANÍTÓ VIDEÓ
        # ==========================================
        self.train_card = ttk.Frame(main_container, style="Card.TFrame", padding=10)
        self.train_card.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            self.train_card,
            text="1. LÉPÉS: Tanító Videó (Parkolóhelyek Feltérképezése)",
            style="Section.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            self.train_card,
            text="Válaszd ki a tanító videót a parkolóhelyek elrendezésének megtanulásához a távoli AI szerveren.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 5))

        file_row1 = ttk.Frame(self.train_card, style="Card.TFrame")
        file_row1.pack(fill=tk.X, pady=(0, 6))

        self.train_entry = tk.Entry(
            file_row1,
            textvariable=self.train_video_path,
            font=("Segoe UI", 9),
            bg="#0f172a",
            fg="#ffffff",
            insertbackground="#ffffff",
            disabledbackground="#1e293b",
            disabledforeground="#64748b",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#334155",
        )
        self.train_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 8))

        self.btn_browse_train = ttk.Button(
            file_row1,
            text="📁 Tallózás",
            style="Secondary.TButton",
            command=self.browse_train_video,
        )
        self.btn_browse_train.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_view_train_orig = ttk.Button(
            file_row1,
            text="👁️ Videó Megnézése",
            style="Secondary.TButton",
            command=lambda: self.open_video(self.train_video_path.get()),
        )
        self.btn_view_train_orig.pack(side=tk.LEFT)

        action_row1 = ttk.Frame(self.train_card, style="Card.TFrame")
        action_row1.pack(fill=tk.X)

        self.btn_train = ttk.Button(
            action_row1,
            text="▶️ Tanítás és Feldolgozás Indítása",
            style="Primary.TButton",
            command=self.start_training,
        )
        self.btn_train.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_view_train_result = ttk.Button(
            action_row1,
            text="🎬 Eredmény Videó Megnézése",
            style="Secondary.TButton",
            command=lambda: self.open_video(self.train_output_video),
        )
        self.btn_view_train_result.pack(side=tk.LEFT)
        self.btn_view_train_result.state(["disabled"])

        self.train_status_lbl = ttk.Label(
            self.train_card,
            text="Státusz: Készen áll a tanításra.",
            style="Muted.TLabel",
        )
        self.train_status_lbl.pack(anchor="w", pady=(5, 0))

        # ==========================================
        # 2. KÁRTYA: TESZTELŐ VIDEÓ
        # ==========================================
        self.test_card = ttk.Frame(main_container, style="Card.TFrame", padding=10)
        self.test_card.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            self.test_card,
            text="2. LÉPÉS: Tesztelő Videó (Elemzés az Új Videón)",
            style="Section.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            self.test_card,
            text="Miután a tanítás kész, tallózd be a másik videót a távoli AI szerveren történő elemzéshez.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 5))

        file_row2 = ttk.Frame(self.test_card, style="Card.TFrame")
        file_row2.pack(fill=tk.X, pady=(0, 6))

        self.test_entry = tk.Entry(
            file_row2,
            textvariable=self.test_video_path,
            font=("Segoe UI", 9),
            bg="#0f172a",
            fg="#ffffff",
            insertbackground="#ffffff",
            disabledbackground="#1e293b",
            disabledforeground="#64748b",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#334155",
        )
        self.test_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 8))

        self.btn_browse_test = ttk.Button(
            file_row2,
            text="📁 Tallózás",
            style="Secondary.TButton",
            command=self.browse_test_video,
        )
        self.btn_browse_test.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_view_test_orig = ttk.Button(
            file_row2,
            text="👁️ Videó Megnézése",
            style="Secondary.TButton",
            command=lambda: self.open_video(self.test_video_path.get()),
        )
        self.btn_view_test_orig.pack(side=tk.LEFT)

        action_row2 = ttk.Frame(self.test_card, style="Card.TFrame")
        action_row2.pack(fill=tk.X)

        self.btn_test = ttk.Button(
            action_row2,
            text="🚀 Elemzés Indítása a Teszt Videón",
            style="Success.TButton",
            command=self.start_testing,
        )
        self.btn_test.pack(side=tk.LEFT, padx=(0, 10))
        self.btn_test.state(["disabled"])

        self.btn_view_test_result = ttk.Button(
            action_row2,
            text="🎬 Feldolgozott Teszt Videó Megnézése",
            style="Secondary.TButton",
            command=lambda: self.open_video(self.test_output_video),
        )
        self.btn_view_test_result.pack(side=tk.LEFT)
        self.btn_view_test_result.state(["disabled"])

        self.test_status_lbl = ttk.Label(
            self.test_card,
            text="Státusz: Várakozás a tanítás befejezésére...",
            style="Muted.TLabel",
        )
        self.test_status_lbl.pack(anchor="w", pady=(5, 0))

        # ==========================================
        # FOLYAMATJELZŐ ÉS RÉSZLETEK
        # ==========================================
        status_frame = ttk.Frame(main_container, style="Card.TFrame", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True)

        self.progress_lbl = ttk.Label(status_frame, text="Folyamat: Inaktív", font=("Segoe UI", 10, "bold"))
        self.progress_lbl.pack(anchor="w", pady=(0, 4))

        self.progress_bar = ttk.Progressbar(status_frame, style="TProgressbar", mode="determinate", maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 6))

        self.stats_lbl = ttk.Label(
            status_frame,
            text="Legutóbbi eredmény: Még nem történt futtatás.",
            font=("Segoe UI", 11),
            foreground="#38bdf8",
        )
        self.stats_lbl.pack(anchor="w")

    def _on_destroy(self, event=None):
        if event is None or event.widget is self.root:
            with self._ui_lock:
                self._is_closing = True
                self._ui_drain_scheduled = False
                poll_id = self._ui_poll_after_id
                self._ui_poll_after_id = None
                drain_id = self._ui_drain_after_id
                self._ui_drain_after_id = None
            for after_id in (poll_id, drain_id):
                if after_id is not None:
                    try:
                        self.root.after_cancel(after_id)
                    except (tk.TclError, RuntimeError):
                        pass

    def _on_close(self):
        self._on_destroy()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _drain_ui_queue(self):
        with self._ui_lock:
            if self._is_closing:
                self._ui_drain_after_id = None
                self._ui_drain_scheduled = False
                return
            self._ui_drain_after_id = None
            self._ui_drain_scheduled = False
            self._ui_drain_gen += 1
        while True:
            with self._ui_lock:
                if self._is_closing:
                    return
            try:
                callback, args = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            with self._ui_lock:
                if self._is_closing:
                    return
            try:
                callback(*args)
            except tk.TclError:
                pass
        with self._ui_lock:
            need_reschedule = (
                not self._is_closing
                and not self._ui_queue.empty()
                and self._ui_drain_after_id is None
                and not self._ui_drain_scheduled
            )
            if need_reschedule:
                self._ui_drain_scheduled = True
                gen = self._ui_drain_gen
        if need_reschedule:
            try:
                after_id = self.root.after(0, self._drain_ui_queue)
            except (tk.TclError, RuntimeError):
                with self._ui_lock:
                    self._ui_drain_scheduled = False
                return
            cancel_id = None
            with self._ui_lock:
                if self._is_closing:
                    self._ui_drain_scheduled = False
                    cancel_id = after_id
                elif self._ui_drain_gen == gen:
                    self._ui_drain_after_id = after_id
            if cancel_id is not None:
                try:
                    self.root.after_cancel(cancel_id)
                except (tk.TclError, RuntimeError):
                    pass

    def _poll_ui_queue(self):
        with self._ui_lock:
            if self._is_closing:
                self._ui_poll_after_id = None
                return
            self._ui_poll_after_id = None
        self._drain_ui_queue()
        with self._ui_lock:
            closing = self._is_closing
        if not closing:
            try:
                poll_id = self.root.after(15, self._poll_ui_queue)
            except (tk.TclError, RuntimeError):
                return
            cancel_id = None
            with self._ui_lock:
                if self._is_closing:
                    cancel_id = poll_id
                else:
                    self._ui_poll_after_id = poll_id
            if cancel_id is not None:
                try:
                    self.root.after_cancel(cancel_id)
                except (tk.TclError, RuntimeError):
                    pass

    def _safe_after(self, callback, *args):
        with self._ui_lock:
            if self._is_closing:
                return
            self._ui_queue.put((callback, args))
            if self._ui_drain_after_id is not None or self._ui_drain_scheduled:
                return
            self._ui_drain_scheduled = True
            gen = self._ui_drain_gen
        try:
            after_id = self.root.after(0, self._drain_ui_queue)
        except (tk.TclError, RuntimeError):
            with self._ui_lock:
                self._ui_drain_scheduled = False
            return
        cancel_id = None
        with self._ui_lock:
            if self._is_closing:
                self._ui_drain_scheduled = False
                cancel_id = after_id
            elif self._ui_drain_gen == gen:
                self._ui_drain_after_id = after_id
        if cancel_id is not None:
            try:
                self.root.after_cancel(cancel_id)
            except (tk.TclError, RuntimeError):
                pass

    @staticmethod
    def _parse_host_port_quiet(raw_host: str, raw_port: str) -> Optional[tuple[str, int]]:
        stripped_host = raw_host.strip()
        if not stripped_host or stripped_host.lower().startswith("https://"):
            return None
        no_scheme = stripped_host[7:] if stripped_host.lower().startswith("http://") else stripped_host
        no_scheme = no_scheme.rstrip("/")
        if (
            not no_scheme
            or any(ch.isspace() for ch in no_scheme)
            or "/" in no_scheme
            or any(ch in no_scheme for ch in (":", "?", "#", "@", "\\"))
        ):
            return None
        stripped_port = raw_port.strip()
        if not stripped_port or not stripped_port.isascii() or not stripped_port.isdigit():
            return None
        port_num = int(stripped_port)
        if not (1 <= port_num <= 65535):
            return None
        return (no_scheme, port_num)

    def _did_server_target_change(
        self, server_target: tuple[str, int], config_seq: int
    ) -> bool:
        if config_seq == self._server_config_seq:
            return False
        return (
            self._parse_host_port_quiet(self.server_host.get(), self.server_port.get())
            != server_target
        )

    def _on_server_config_changed(self, *_args):
        self._server_config_seq += 1
        self._health_check_seq += 1
        self._is_checking_server = False
        if not self.is_processing:
            self.btn_check_server.state(["!disabled"])
        self.server_status_lbl.config(
            text="Szerver státusz: Cím módosítva — ellenőrzés szükséges",
            foreground="#94a3b8",
        )
        if not self.is_processing and self._active_server_target is not None:
            parsed = self._parse_host_port_quiet(self.server_host.get(), self.server_port.get())
            if parsed != self._active_server_target:
                self.session_id = None
                self._active_server_target = None
                self.btn_test.state(["disabled"])
                self.test_status_lbl.config(
                    text="Státusz: A szerver címe megváltozott — futtasd újra az 1. lépést!"
                )

    def _build_validated_client(self) -> Optional[ParkingRemoteClient]:
        raw_host = self.server_host.get()
        raw_port = self.server_port.get()

        stripped_host = raw_host.strip()
        if not stripped_host:
            messagebox.showerror("Érvénytelen szervercím", "Add meg a távoli AI szerver IP-címét vagy hostnevét!")
            return None
        if stripped_host.lower().startswith("https://"):
            messagebox.showerror(
                "Érvénytelen szervercím",
                "Csak HTTP protokoll támogatott ('https://' nem engedélyezett).",
            )
            return None

        no_scheme = stripped_host[7:] if stripped_host.lower().startswith("http://") else stripped_host
        no_scheme = no_scheme.rstrip("/")
        if (
            not no_scheme
            or any(ch.isspace() for ch in no_scheme)
            or "/" in no_scheme
            or any(ch in no_scheme for ch in (":", "?", "#", "@", "\\"))
        ):
            messagebox.showerror(
                "Érvénytelen szervercím",
                f"Érvénytelen szerver cím (host): {raw_host!r}\n"
                f"Add meg csak az IP-címet vagy hostnevet (pl. {DEFAULT_SERVER_HOST}), port és útvonal nélkül.",
            )
            return None

        stripped_port = raw_port.strip()
        if not stripped_port or not stripped_port.isascii() or not stripped_port.isdigit():
            messagebox.showerror(
                "Érvénytelen port",
                "Érvénytelen port szám (1–65535 közötti egész szám szükséges).",
            )
            return None
        port_num = int(stripped_port)
        if not (1 <= port_num <= 65535):
            messagebox.showerror(
                "Érvénytelen port",
                "Érvénytelen port szám (1–65535 közötti egész szám szükséges).",
            )
            return None

        try:
            return ParkingRemoteClient(host=raw_host, port=port_num)
        except (ValueError, RemoteClientError) as exc:
            messagebox.showerror("Érvénytelen szerver beállítás", str(exc))
            return None

    def browse_train_video(self):
        prev = self.train_video_path.get().strip()
        path = filedialog.askopenfilename(
            title="Tanító videó kiválasztása",
            filetypes=[("Videófájlok", "*.mp4 *.avi *.mov *.mkv"), ("Minden fájl", "*.*")],
        )
        if path:
            changed = (not prev) or (os.path.abspath(path) != os.path.abspath(prev))
            self.train_video_path.set(path)
            self.train_status_lbl.config(text=f"Kiválasztva: {os.path.basename(path)}")
            if changed and (
                self.session_id is not None
                or self._train_result_ready
                or self._test_result_ready
            ):
                self.session_id = None
                self._active_server_target = None
                self._train_result_ready = False
                self._test_result_ready = False
                self.test_status_lbl.config(text="Státusz: Várakozás a tanítás befejezésére...")
                self._refresh_button_states()

    def browse_test_video(self):
        path = filedialog.askopenfilename(
            title="Tesztelő videó kiválasztása",
            filetypes=[("Videófájlok", "*.mp4 *.avi *.mov *.mkv"), ("Minden fájl", "*.*")],
        )
        if path:
            self.test_video_path.set(path)
            self.test_status_lbl.config(text=f"Kiválasztva: {os.path.basename(path)}")

    def open_video(self, video_file):
        cleaned = (video_file or "").strip()
        if not cleaned or not os.path.isfile(cleaned):
            messagebox.showwarning("Figyelmeztetés", "A megadott videófájl még nem létezik!")
            return
        target = os.path.abspath(cleaned)
        try:
            if sys.platform == "win32":
                os.startfile(target)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
        except Exception as exc:
            messagebox.showerror("Hiba", f"Nem sikerült megnyitni a videót: {exc}")

    def check_server_health(self):
        if self.is_processing or self._is_checking_server:
            return
        client = self._build_validated_client()
        if client is None:
            return

        self._is_checking_server = True
        self._health_check_seq += 1
        seq = self._health_check_seq
        self.btn_check_server.state(["disabled"])
        self.server_status_lbl.config(
            text=f"Szerver státusz: Kapcsolat ellenőrzése ({client.base_url})...",
            foreground="#facc15",
        )

        def run_health():
            try:
                info = client.check_health()
                self._safe_after(self._on_health_check_success, seq, client.base_url, info)
            except (ServerUnreachableError, ServerResponseError, RemoteClientError, Exception) as exc:
                err_msg = str(exc)
                self._safe_after(self._on_health_check_error, seq, client.base_url, err_msg)

        threading.Thread(target=run_health, daemon=True).start()

    def _on_health_check_success(self, seq: int, base_url: str, info: dict):
        if seq != self._health_check_seq:
            return
        self._is_checking_server = False
        self._refresh_button_states()

        status_text = str(info.get("status", "ok"))
        model_loaded = info.get("model_loaded", True)
        model_name = info.get("model_name")
        device = info.get("device")

        details = [f"állapot: {status_text}"]
        if model_name:
            details.append(f"modell: {model_name}")
        if device:
            details.append(f"eszköz: {device}")
        color = "#10b981" if status_text.lower() == "ok" and model_loaded else "#facc15"
        self.server_status_lbl.config(
            text=f"Szerver státusz: Elérhető ({', '.join(details)}) • {base_url}",
            foreground=color,
        )
        messagebox.showinfo(
            "Szerver Elérhető",
            f"Sikeres kapcsolat a távoli AI szerverrel!\n\n"
            f"Cím: {base_url}\n"
            f"Állapot: {status_text}\n"
            f"Modell betöltve: {'Igen' if model_loaded else 'Nem'}"
            + (f"\nModell: {model_name}" if model_name else "")
            + (f"\nEszköz: {device}" if device else ""),
        )

    def _on_health_check_error(self, seq: int, base_url: str, err_msg: str):
        if seq != self._health_check_seq:
            return
        self._is_checking_server = False
        self._refresh_button_states()
        self.server_status_lbl.config(
            text=f"Szerver státusz: Nem elérhető ({base_url})",
            foreground="#f87171",
        )
        messagebox.showerror(
            "Szerverkapcsolati hiba",
            f"Nem sikerült kapcsolódni a távoli AI szerverhez ({base_url}):\n{err_msg}\n\n"
            f"Helyi AI-feldolgozás nem áll rendelkezésre (vékony kliens mód).",
        )

    def start_training(self):
        if self.is_processing:
            return
        video = self.train_video_path.get().strip()
        if not video or not os.path.isfile(video):
            messagebox.showerror("Hiba", f"A tanító videó nem található:\n{video}")
            return
        try:
            if os.path.getsize(video) <= 0:
                messagebox.showerror("Hiba", f"A kiválasztott tanító videó üres (0 bájt):\n{video}")
                return
        except OSError as exc:
            messagebox.showerror("Hiba", f"Nem sikerült ellenőrizni a tanító videófájlt:\n{exc}")
            return

        client = self._build_validated_client()
        if client is None:
            return

        # Korábbi munkamenet és eredmények érvénytelenítése új tanítás előtt
        self._health_check_seq += 1
        self._is_checking_server = False
        self.session_id = None
        self._active_server_target = None
        self._train_result_ready = False
        self._test_result_ready = False

        self._set_ui_busy(True)
        self.progress_bar["value"] = 0
        self.train_status_lbl.config(text="Tanító videó feltöltése és feldolgozása a távoli AI szerveren...")
        self.test_status_lbl.config(text="Státusz: Várakozás a tanítás befejezésére...")
        self.progress_lbl.config(text="Tanító videó feltöltése a távoli AI szerverre...")
        self.stats_lbl.config(text="Élő állapot: Kapcsolódás a távoli AI szerverhez...")

        output_path = self.train_output_video
        server_target = (client.host, client.port)
        base_url = client.base_url
        config_seq = self._server_config_seq

        def run_thread():
            try:
                def on_progress(cur: int, total: int, occ: int, free: int):
                    self._safe_after(self._update_progress, "Feldolgozás", cur, total, occ, free)

                result = client.run_training(
                    video_path=video,
                    output_path=output_path,
                    progress_callback=on_progress,
                )
                self._safe_after(self._on_training_done, result, server_target, base_url, config_seq)
            except (ServerUnreachableError, ServerResponseError, RemoteClientError, ValueError, Exception) as exc:
                err_msg = str(exc)
                unreachable = isinstance(exc, ServerUnreachableError)
                self._safe_after(
                    self._on_error,
                    "Tanítási hiba",
                    err_msg,
                    "train",
                    base_url,
                    unreachable,
                    server_target,
                    config_seq,
                )

        threading.Thread(target=run_thread, daemon=True).start()

    def _on_training_done(
        self,
        result: dict,
        server_target: tuple[str, int],
        base_url: str,
        config_seq: Optional[int] = None,
    ):
        target_changed = (
            config_seq is not None and self._did_server_target_change(server_target, config_seq)
        )
        self.train_output_video = result["output_path"]
        self._train_result_ready = True

        if target_changed:
            self.session_id = None
            self._active_server_target = None
        else:
            self.session_id = result["session_id"]
            self._active_server_target = server_target

        self._set_ui_busy(False)
        self.progress_bar["value"] = 100
        self.progress_lbl.config(
            text=f"Feldolgozás: 100% ({result['frames_processed']}/{result['frames_processed']}) | "
                 f"Foglalt: {result['final_occupied']}, Szabad: {result['final_free']}"
        )
        self.stats_lbl.config(
            text=f"Eredmény: Összes: {result['total_slots']}  |  "
                 f"Foglalt: {result['final_occupied']} (Piros)  |  "
                 f"Szabad: {result['final_free']} (Zöld)"
        )

        if target_changed:
            self.server_status_lbl.config(
                text="Szerver státusz: Cím módosítva — ellenőrzés szükséges",
                foreground="#94a3b8",
            )
            self.train_status_lbl.config(
                text="✅ Tanítás kész a korábbi szerveren, de a szerver címe közben megváltozott."
            )
            self.test_status_lbl.config(
                text="Státusz: A szerver címe megváltozott — futtasd újra az 1. lépést!"
            )
            messagebox.showwarning(
                "Szervercím megváltozott",
                f"A tanító videó feldolgozása elkészült ({base_url}), de a szerver címe közben megváltozott.\n\n"
                f"A 2. lépéshez futtasd újra az 1. lépést az új szervercímen!",
            )
            return

        self.server_status_lbl.config(text=f"Szerver státusz: Elérhető ({base_url})", foreground="#10b981")
        self.train_status_lbl.config(text="✅ Tanítás és feltérképezés sikeresen befejeződött!")
        self.test_status_lbl.config(text="Készen áll a teszt videó elemzésére. Válassz ki egy teszt videót!")
        messagebox.showinfo(
            "Tanítás Befejeződött",
            f"A tanító videó feldolgozása sikeres volt a távoli AI szerveren!\n\n"
            f"Összes feltérképezett hely: {result['total_slots']}\n"
            f"Foglalt: {result['final_occupied']}\n"
            f"Szabad: {result['final_free']}\n\n"
            f"Most kiválaszthatod a tesztelő videót a 2. lépésben!",
        )

    def start_testing(self):
        if self.is_processing:
            return
        if not self.session_id:
            messagebox.showwarning(
                "Figyelmeztetés",
                "Először futtasd le sikeresen a tanítást (1. lépés) a távoli AI szerveren!",
            )
            return

        video = self.test_video_path.get().strip()
        if not video or not os.path.isfile(video):
            messagebox.showerror("Hiba", f"A tesztelő videó nem található:\n{video}")
            return
        try:
            if os.path.getsize(video) <= 0:
                messagebox.showerror("Hiba", f"A kiválasztott tesztelő videó üres (0 bájt):\n{video}")
                return
        except OSError as exc:
            messagebox.showerror("Hiba", f"Nem sikerült ellenőrizni a tesztelő videófájlt:\n{exc}")
            return

        client = self._build_validated_client()
        if client is None:
            return

        if self._active_server_target is not None and self._active_server_target != (client.host, client.port):
            self.session_id = None
            self._active_server_target = None
            self.btn_test.state(["disabled"])
            self.test_status_lbl.config(
                text="Státusz: A szerver címe megváltozott — futtasd újra az 1. lépést!"
            )
            messagebox.showwarning(
                "Munkamenet érvénytelen",
                "A szerver címe megváltozott az 1. lépés óta. Futtasd újra az 1. lépést ezen a szerveren!",
            )
            return

        sid = self.session_id
        output_path = self.test_output_video
        server_target = (client.host, client.port)
        base_url = client.base_url
        config_seq = self._server_config_seq

        self._health_check_seq += 1
        self._is_checking_server = False
        self._test_result_ready = False
        self._set_ui_busy(True)
        self.progress_bar["value"] = 0
        self.test_status_lbl.config(text="Teszt videó feltöltése és elemzése folyamatban a távoli AI szerveren...")
        self.progress_lbl.config(text="Teszt videó feltöltése és foglaltság-ellenőrzése...")
        self.stats_lbl.config(text="Élő állapot: Kapcsolódás a távoli AI szerverhez...")

        def run_thread():
            try:
                def on_progress(cur: int, total: int, occ: int, free: int):
                    self._safe_after(self._update_progress, "Tesztelés", cur, total, occ, free)

                result = client.run_analysis(
                    video_path=video,
                    output_path=output_path,
                    session_id=sid,
                    progress_callback=on_progress,
                )
                self._safe_after(self._on_testing_done, result, server_target, base_url, config_seq)
            except (ServerUnreachableError, ServerResponseError, RemoteClientError, ValueError, Exception) as exc:
                err_msg = str(exc)
                unreachable = isinstance(exc, ServerUnreachableError)
                self._safe_after(
                    self._on_error,
                    "Tesztelési hiba",
                    err_msg,
                    "test",
                    base_url,
                    unreachable,
                    server_target,
                    config_seq,
                )

        threading.Thread(target=run_thread, daemon=True).start()

    def _on_testing_done(
        self,
        result: dict,
        server_target: tuple[str, int],
        base_url: str,
        config_seq: Optional[int] = None,
    ):
        target_changed = (
            config_seq is not None and self._did_server_target_change(server_target, config_seq)
        )
        self.test_output_video = result["output_path"]
        self._test_result_ready = True

        if target_changed:
            self.session_id = None
            self._active_server_target = None

        self._set_ui_busy(False)
        self.progress_bar["value"] = 100
        self.progress_lbl.config(
            text=f"Tesztelés: 100% ({result['frames_processed']}/{result['frames_processed']}) | "
                 f"Foglalt: {result['final_occupied']}, Szabad: {result['final_free']}"
        )
        self.stats_lbl.config(
            text=f"Teszt eredmény: Összes: {result['total_slots']}  |  "
                 f"Foglalt: {result['final_occupied']} (Piros)  |  "
                 f"Szabad: {result['final_free']} (Zöld)"
        )

        if target_changed:
            self.server_status_lbl.config(
                text="Szerver státusz: Cím módosítva — ellenőrzés szükséges",
                foreground="#94a3b8",
            )
            self.test_status_lbl.config(
                text="Státusz: A szerver címe megváltozott — futtasd újra az 1. lépést!"
            )
            messagebox.showwarning(
                "Szervercím megváltozott",
                f"A teszt videó elemzése elkészült ({base_url}), de a szerver címe közben megváltozott.\n\n"
                f"A következő elemzéshez futtasd újra az 1. lépést az új szervercímen!",
            )
            return

        self.server_status_lbl.config(
            text=f"Szerver státusz: Elérhető ({base_url})", foreground="#10b981"
        )
        self.test_status_lbl.config(text="✅ Teszt videó elemzése sikeresen befejeződött!")

        messagebox.showinfo(
            "Tesztelés Kész",
            f"A teszt videó elemzése elkészült a távoli AI szerveren!\n\n"
            f"Összes hely: {result['total_slots']}\n"
            f"Foglalt: {result['final_occupied']}\n"
            f"Szabad: {result['final_free']}\n\n"
            f"Kattints a 'Feldolgozott Teszt Videó Megnézése' gombra az ellenőrzéshez!",
        )

    def _update_progress(self, stage: str, cur: int, total: int, occ: int, free: int):
        pct = max(0, min(100, int((cur / total) * 100))) if total > 0 else 0
        self.progress_bar["value"] = pct
        if total > 0:
            self.progress_lbl.config(text=f"{stage}: {pct}% ({cur}/{total}) | Foglalt: {occ}, Szabad: {free}")
        else:
            self.progress_lbl.config(text=f"{stage}: Várakozás a távoli feldolgozásra...")

        if total <= 0 and occ == 0 and free == 0:
            self.stats_lbl.config(text="Élő állapot: Szerveroldali előkészítés...")
        else:
            self.stats_lbl.config(
                text=f"Élő állapot: Összes: {occ + free}  |  Foglalt: {occ} (Piros)  |  Szabad: {free} (Zöld)"
            )

    def _on_error(
        self,
        title: str,
        msg: str,
        step: Optional[str] = None,
        base_url: str = "",
        unreachable: bool = False,
        server_target: Optional[tuple[str, int]] = None,
        config_seq: Optional[int] = None,
    ):
        target_changed = (
            server_target is not None
            and config_seq is not None
            and self._did_server_target_change(server_target, config_seq)
        )
        self.progress_bar["value"] = 0
        self.progress_lbl.config(text="Folyamat: Hiba történt.")
        self.stats_lbl.config(text="Legutóbbi eredmény: A művelet hiba miatt megszakadt.")

        if target_changed:
            self.session_id = None
            self._active_server_target = None

        if step == "train":
            self.session_id = None
            self._active_server_target = None
            self._train_result_ready = False
            self.train_status_lbl.config(text="❌ Hiba történt a tanítás során.")
            self.test_status_lbl.config(
                text=(
                    "Státusz: A szerver címe megváltozott — futtasd újra az 1. lépést!"
                    if target_changed
                    else "Státusz: Várakozás a tanítás befejezésére..."
                )
            )
        elif step == "test":
            self._test_result_ready = False
            self.test_status_lbl.config(
                text=(
                    "Státusz: A szerver címe megváltozott — futtasd újra az 1. lépést!"
                    if target_changed
                    else "❌ Hiba történt a tesztelés során."
                )
            )

        if target_changed:
            self.server_status_lbl.config(
                text="Szerver státusz: Cím módosítva — ellenőrzés szükséges",
                foreground="#94a3b8",
            )
        elif unreachable and base_url:
            self.server_status_lbl.config(
                text=f"Szerver státusz: Nem elérhető ({base_url})",
                foreground="#f87171",
            )

        self._set_ui_busy(False)
        messagebox.showerror(
            title,
            f"Hiba történt a távoli AI szerverrel végzett művelet közben:\n{msg}\n\n"
            f"Megjegyzés: Ez az alkalmazás vékony kliens, helyi AI-feldolgozás nem futtatható.",
        )

    def _refresh_button_states(self):
        if self.is_processing:
            for btn in (
                self.btn_train,
                self.btn_test,
                self.btn_view_train_result,
                self.btn_view_test_result,
                self.btn_browse_train,
                self.btn_view_train_orig,
                self.btn_browse_test,
                self.btn_view_test_orig,
                self.btn_check_server,
            ):
                btn.state(["disabled"])
            for entry in (
                self.train_entry,
                self.test_entry,
                self.server_host_entry,
                self.server_port_entry,
            ):
                entry.config(state="disabled")
            return

        for entry in (
            self.train_entry,
            self.test_entry,
            self.server_host_entry,
            self.server_port_entry,
        ):
            entry.config(state="normal")

        for btn in (
            self.btn_train,
            self.btn_browse_train,
            self.btn_view_train_orig,
            self.btn_browse_test,
            self.btn_view_test_orig,
        ):
            btn.state(["!disabled"])

        self.btn_check_server.state(["disabled"] if self._is_checking_server else ["!disabled"])
        self.btn_test.state(["!disabled"] if bool(self.session_id) else ["disabled"])

        train_ready = self._train_result_ready and os.path.isfile(self.train_output_video)
        self.btn_view_train_result.state(["!disabled"] if train_ready else ["disabled"])

        test_ready = self._test_result_ready and os.path.isfile(self.test_output_video)
        self.btn_view_test_result.state(["!disabled"] if test_ready else ["disabled"])

    def _set_ui_busy(self, busy: bool):
        self.is_processing = bool(busy)
        self._refresh_button_states()


def main():
    root = tk.Tk()
    app = ParkingAppGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
