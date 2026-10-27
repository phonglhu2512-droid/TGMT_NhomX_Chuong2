"""Tkinter/OpenCV application for image and video processing."""

import argparse
import base64
import os
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from urllib.request import urlretrieve

try:
    import cv2
    import numpy as np
except ImportError as import_error:
    cv2 = None
    np = None
    _CV_IMPORT_ERROR = import_error
else:
    _CV_IMPORT_ERROR = None


COLORS = {
    "background": "#F0F2F5",
    "panel": "#FFFFFF",
    "border": "#D8DEE6",
    "text": "#212529",
    "muted": "#667085",
    "accent": "#007A87",
    "accent_hover": "#00636D",
    "accent_light": "#E2F2F3",
    "canvas": "#F8FAFC",
}


class VideoApp:
    def __init__(self, root, video_path=None, initial_mode="rotate"):
        self.root = root
        self.root.title("NguyenNguyen | Xử lý ảnh & video")
        self.root.geometry("1180x820")
        self.root.minsize(850, 640)
        self.root.configure(bg=COLORS["background"])
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        self.capture = None
        self.video_path = None
        self.source_name = None
        self.is_webcam = False
        self.paused = False
        self.camera_stop_event = threading.Event()
        self.camera_thread = None
        self.camera_frame_lock = threading.Lock()
        self.camera_frame = None
        self.camera_error = None
        self.still_image = None
        self.still_image_path = None
        self.current_faces = []
        self.zoom = 1.0
        self.translate_x = 0
        self.translate_y = 0
        self.brightness = 0
        self.color_mode = "BGR"
        self.mode = "rotate"
        self.turns_clockwise = 0
        self.frame_interval_ms = 33
        self.duration_seconds = 0.0
        self.seeking = False
        self.face_detector = None
        self.face_cascade = None
        self.detector_loading = False
        self.balloon_image = None
        self.balloon_image_path = None
        self.balloon_detections = []
        self.plate_image = None
        self.plate_image_path = None
        self.current_frame = None
        self.current_faces = []
        self.photo = None
        self.image_item = None
        self.closing = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress>", self.on_key)
        self.set_mode(initial_mode)
        if video_path:
            if initial_mode == "balloons":
                self.open_balloon_image(video_path)
            elif initial_mode == "plate":
                self.open_plate_image(video_path)
            else:
                self.open_video(video_path)
        self.root.after(15, self.update_frame)

    def _build_ui(self):
        header = tk.Frame(self.root, bg=COLORS["background"], padx=24, pady=18)
        header.grid(row=0, column=0, sticky="ew")

        title = tk.Label(
            header,
            text="NGUYEN NGUYEN - CEO APPLE",
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Segoe UI", 18, "bold"),
        )
        title.pack(side="left")

        source_buttons = tk.Frame(header, bg=COLORS["background"])
        source_buttons.pack(side="right")
        self._make_button(source_buttons, "Mở ảnh", self.choose_image).pack(
            side="left", padx=(0, 6)
        )
        self.webcam_button = self._make_button(
            source_buttons, "Mở Webcam", self.open_webcam, primary=True
        )
        self.webcam_button.pack(side="left", padx=(0, 6))
        self.open_button = tk.Button(
            source_buttons,
            text="Mở video",
            command=self.choose_video,
            bg=COLORS["accent"],
            fg="#FFFFFF",
            activebackground=COLORS["accent_hover"],
            activeforeground="#FFFFFF",
            relief="flat",
            padx=18,
            pady=9,
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
        )
        self.open_button.pack(side="left", padx=(0, 6))
        self.replay_button = self._make_button(
            source_buttons, "Phát lại", self.replay_video
        )
        self.replay_button.configure(state=tk.DISABLED)
        self.replay_button.pack(side="left")
        self.transport_button = self._make_button(
            source_buttons, "⏸ Tạm dừng", self.toggle_pause, primary=True
        )
        self.transport_button.pack(side="left", padx=(6, 0))

        video_area = tk.Frame(
            self.root, bg=COLORS["panel"], padx=12, pady=12,
            highlightbackground=COLORS["border"], highlightthickness=1,
        )
        video_area.grid(row=1, column=0, sticky="nsew")
        video_area.grid_rowconfigure(0, weight=1)
        video_area.grid_columnconfigure(0, weight=1)
        self.video_canvas = tk.Canvas(
            video_area,
            width=640,
            height=360,
            bg=COLORS["canvas"],
            highlightthickness=0,
        )
        self.video_canvas.grid(row=0, column=0, sticky="nsew")
        self.placeholder_item = self.video_canvas.create_text(
            320,
            180,
            text="Mở ảnh, video hoặc webcam để bắt đầu",
            fill=COLORS["muted"],
            font=("Segoe UI", 16),
        )
        self.video_canvas.bind("<Configure>", self._resize_placeholder)

        seek_area = tk.Frame(
            self.root, bg=COLORS["background"], padx=24, pady=6
        )
        self.seek_area = seek_area
        seek_area.grid(row=2, column=0, sticky="ew")
        seek_area.grid_columnconfigure(1, weight=1)
        self.position_var = tk.StringVar(value="00:00")
        self.duration_var = tk.StringVar(value="00:00")
        tk.Label(
            seek_area,
            textvariable=self.position_var,
            bg=COLORS["background"],
            fg=COLORS["muted"],
            width=6,
            anchor="w",
            font=("Segoe UI", 9),
        ).grid(row=0, column=0, padx=(0, 8))
        self.seek_var = tk.DoubleVar(value=0)
        self.seek_scale = tk.Scale(
            seek_area,
            from_=0,
            to=1,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            showvalue=False,
            variable=self.seek_var,
            command=self._seek_changed,
            bg=COLORS["background"],
            fg=COLORS["text"],
            troughcolor=COLORS["border"],
            activebackground=COLORS["accent"],
            highlightthickness=0,
            borderwidth=0,
            sliderlength=18,
            state=tk.DISABLED,
        )
        self.seek_scale.grid(row=0, column=1, sticky="ew")
        self.seek_scale.bind("<ButtonPress-1>", self._start_seek)
        self.seek_scale.bind("<ButtonRelease-1>", self._finish_seek)
        tk.Label(
            seek_area,
            textvariable=self.duration_var,
            bg=COLORS["background"],
            fg=COLORS["muted"],
            width=6,
            anchor="e",
            font=("Segoe UI", 9),
        ).grid(row=0, column=2, padx=(8, 0))

        controls = tk.Frame(
            self.root, bg=COLORS["panel"], padx=18, pady=14,
            highlightbackground=COLORS["border"], highlightthickness=1,
        )
        controls.grid(row=3, column=0, sticky="ew", padx=24, pady=(8, 0))

        mode_group = tk.Frame(controls, bg=COLORS["panel"])
        mode_group.pack(fill="x")
        tk.Label(
            mode_group,
            text="CHẾ ĐỘ",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 7))

        self.mode_tabs = tk.Canvas(
            mode_group, height=40, bg=COLORS["panel"], highlightthickness=0
        )
        self.mode_tabs.pack(fill="x")
        self.mode_tab_values = ("rotate", "faces", "balloons", "plate")
        self.mode_tab_labels = (
            "Ảnh / Xoay video", "Nhận diện mặt", "Đếm bóng bay", "Tách biển số"
        )
        self.mode_tabs.bind("<Configure>", self._draw_mode_tabs)
        self.mode_tabs.bind("<Button-1>", self._mode_tab_click)

        self.action_area = tk.Frame(controls, bg=COLORS["panel"])
        self.action_area.pack(fill="x", pady=(10, 0))

        self.rotate_actions = tk.Frame(self.action_area, bg=COLORS["panel"])
        self._make_button(
            self.rotate_actions, "Xoay trái 90°", lambda: self.rotate(-1)
        ).pack(side="left", padx=(0, 7))
        self._make_button(
            self.rotate_actions, "Xoay phải 90°", lambda: self.rotate(1)
        ).pack(side="left", padx=(0, 6))
        self._make_button(
            self.rotate_actions, "−", lambda: self.change_zoom(0.8)
        ).pack(side="left", padx=(0, 2))
        self._make_button(
            self.rotate_actions, "Zoom +", lambda: self.change_zoom(1.25)
        ).pack(side="left", padx=(0, 6))
        self._make_button(
            self.rotate_actions, "←", lambda: self.translate(-30, 0)
        ).pack(side="left", padx=2)
        self._make_button(
            self.rotate_actions, "↑", lambda: self.translate(0, -30)
        ).pack(side="left", padx=2)
        self._make_button(
            self.rotate_actions, "↓", lambda: self.translate(0, 30)
        ).pack(side="left", padx=2)
        self._make_button(
            self.rotate_actions, "→", lambda: self.translate(30, 0)
        ).pack(side="left", padx=(2, 6))
        self._make_button(
            self.rotate_actions, "Đặt lại", self.reset_transform
        ).pack(side="left", padx=(0, 8))
        self.brightness_var = tk.IntVar(value=0)
        tk.Label(
            self.rotate_actions,
            text="Độ sáng",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
        ).pack(side="left", padx=(0, 4))
        self.brightness_scale = tk.Scale(
            self.rotate_actions, from_=-100, to=100, orient=tk.HORIZONTAL,
            variable=self.brightness_var, command=self._brightness_changed,
            bg=COLORS["panel"], fg=COLORS["text"], troughcolor=COLORS["border"],
            activebackground=COLORS["accent"], highlightthickness=0,
            borderwidth=0, length=110, showvalue=False,
        )
        self.brightness_scale.pack(side="left", padx=(0, 6))
        self.color_var = tk.StringVar(value="BGR")
        self.color_menu = tk.OptionMenu(
            self.rotate_actions, self.color_var, "BGR", "Grayscale", "HSV",
            command=self._color_mode_changed,
        )
        self.color_menu.configure(
            bg=COLORS["panel"], fg=COLORS["text"], relief="flat",
            highlightthickness=1, highlightbackground=COLORS["border"],
            activebackground=COLORS["accent_light"], cursor="hand2",
        )
        self.color_menu.pack(side="left")

        self.face_actions = tk.Frame(self.action_area, bg=COLORS["panel"])
        self.save_button = self._make_button(
            self.face_actions,
            "Lưu khuôn mặt",
            self.save_faces,
            primary=True,
        )
        self.save_button.pack(side="left")

        self.balloon_actions = tk.Frame(self.action_area, bg=COLORS["panel"])
        self.open_image_button = self._make_button(
            self.balloon_actions,
            "Mở ảnh và đếm",
            self.choose_balloon_image,
            primary=True,
        )
        self.open_image_button.pack(side="left")

        self.plate_actions = tk.Frame(self.action_area, bg=COLORS["panel"])
        self.open_plate_button = self._make_button(
            self.plate_actions,
            "Mở ảnh và tách số",
            self.choose_plate_image,
            primary=True,
        )
        self.open_plate_button.pack(side="left")

        self.status_var = tk.StringVar(value="Chưa chọn video")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#E5E9EF",
            fg=COLORS["text"],
            anchor="w",
            padx=24,
            pady=12,
            font=("Segoe UI", 9),
        )
        self.status_label.grid(row=4, column=0, sticky="ew")

    def _resize_placeholder(self, event):
        if self.image_item is None:
            self.video_canvas.coords(
                self.placeholder_item, event.width // 2, event.height // 2
            )

    @staticmethod
    def _draw_pill(canvas, x1, y1, x2, y2, fill):
        radius = max(1, (y2 - y1) // 2)
        canvas.create_rectangle(
            x1 + radius, y1, x2 - radius, y2, fill=fill, outline=fill
        )
        canvas.create_oval(x1, y1, x1 + 2 * radius, y2, fill=fill, outline=fill)
        canvas.create_oval(x2 - 2 * radius, y1, x2, y2, fill=fill, outline=fill)

    def _draw_mode_tabs(self, _event=None):
        if not hasattr(self, "mode_tabs"):
            return
        canvas = self.mode_tabs
        canvas.delete("all")
        width = max(canvas.winfo_width(), 4)
        height = 36
        self._draw_pill(canvas, 0, 2, width, height + 2, "#E9EDF2")
        active_index = self.mode_tab_values.index(self.mode)
        segment_width = width / len(self.mode_tab_values)
        x1 = int(active_index * segment_width + 3)
        x2 = int((active_index + 1) * segment_width - 3)
        self._draw_pill(canvas, x1, 5, x2, height - 1, COLORS["accent"])
        for index, label in enumerate(self.mode_tab_labels):
            canvas.create_text(
                (index + 0.5) * segment_width,
                height / 2 + 1,
                text=label,
                fill="#FFFFFF" if index == active_index else COLORS["text"],
                font=("Segoe UI", 9, "bold" if index == active_index else "normal"),
            )

    def _mode_tab_click(self, event):
        if not self.mode_tab_values:
            return
        index = min(
            len(self.mode_tab_values) - 1,
            int(event.x / max(self.mode_tabs.winfo_width(), 1) * len(
                self.mode_tab_values
            )),
        )
        self.set_mode(self.mode_tab_values[index])

    def _show_placeholder(self, text):
        if self.image_item is not None:
            self.video_canvas.delete(self.image_item)
            self.image_item = None
        self.video_canvas.delete(self.placeholder_item)
        self.placeholder_item = self.video_canvas.create_text(
            self.video_canvas.winfo_width() // 2,
            self.video_canvas.winfo_height() // 2,
            text=text,
            fill=COLORS["muted"],
            font=("Segoe UI", 16),
        )

    @staticmethod
    def _make_button(parent, text, command, primary=False):
        background = COLORS["accent"] if primary else "#E9EDF2"
        foreground = "#FFFFFF" if primary else COLORS["text"]
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=background,
            fg=foreground,
            activebackground=COLORS["accent_hover"] if primary else "#DCE3EA",
            activeforeground=foreground,
            disabledforeground="#98A2B3",
            relief="flat",
            padx=14,
            pady=10,
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
        )

    def set_mode(self, mode):
        self.mode = mode
        self.current_faces = []
        self._draw_mode_tabs()

        self.rotate_actions.pack_forget()
        self.face_actions.pack_forget()
        self.balloon_actions.pack_forget()
        self.plate_actions.pack_forget()
        if mode == "rotate":
            self.rotate_actions.pack(side="left")
            self.open_button.configure(text="Mở video", command=self.choose_video)
            self.status_var.set(self._video_status("Chế độ ảnh và video"))
        elif mode == "faces":
            self.face_actions.pack(side="left")
            self.open_button.configure(text="Mở video", command=self.choose_video)
            self.status_var.set(self._video_status("Đang chuẩn bị nhận diện khuôn mặt"))
            self._ensure_face_detector()
        elif mode == "balloons":
            self.balloon_actions.pack(side="left")
            self.open_button.configure(
                text="Mở ảnh", command=self.choose_balloon_image
            )
            self.status_var.set("Chọn ảnh để đếm bóng bay")
            self._show_placeholder("Chọn ảnh để đếm bóng bay")
        else:
            self.plate_actions.pack(side="left")
            self.open_button.configure(
                text="Mở ảnh", command=self.choose_plate_image
            )
            self.status_var.set("Chọn ảnh biển số để tách ký tự")
            self._show_placeholder("Chọn ảnh biển số để bắt đầu")

        if mode in ("balloons", "plate"):
            if self.capture is not None:
                self._stop_camera_reader()
                if self.capture is not None:
                    self.capture.release()
                    self.capture = None
            self.seek_scale.configure(state=tk.DISABLED)
            self.seek_area.grid_remove()
            self.replay_button.pack_forget()
            self.transport_button.pack_forget()
        else:
            self.seek_area.grid()
            if self.video_path and self.replay_button.winfo_manager() == "":
                self.replay_button.pack(side="right", padx=(0, 8))
            if self.transport_button.winfo_manager() == "":
                self.transport_button.pack(side="left", padx=(6, 0))
            self.seek_scale.configure(
                state=tk.NORMAL if self.duration_seconds > 0 else tk.DISABLED
            )
        self._update_transport_button()
        if mode in ("rotate", "faces") and self.still_image is not None:
            self._render_still_image()

    @staticmethod
    def _read_image(path):
        image_bytes = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    @staticmethod
    def _write_image(path, image):
        extension = Path(path).suffix or ".png"
        success, encoded = cv2.imencode(extension, image)
        if not success:
            return False
        encoded.tofile(str(path))
        return True

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=(
                ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("Tất cả file", "*.*"),
            ),
        )
        if path:
            self.open_image(path)

    def open_image(self, path):
        if self.mode == "balloons":
            self.open_balloon_image(path)
            return
        if self.mode == "plate":
            self.open_plate_image(path)
            return
        try:
            image = self._read_image(path)
        except (OSError, cv2.error) as error:
            self.status_var.set(f"Lỗi đọc ảnh: {error}")
            return
        if image is None:
            self.status_var.set(f"Không mở được ảnh: {path}")
            return
        if self.capture is not None:
            self._stop_camera_reader()
            if self.capture is not None:
                self.capture.release()
                self.capture = None
        self.video_path = None
        self.source_name = Path(path).name
        self.is_webcam = False
        self.still_image = image
        self.still_image_path = path
        self.current_frame = image
        self.current_faces = []
        self.duration_seconds = 0.0
        self.paused = True
        self.seek_scale.configure(state=tk.DISABLED)
        self.replay_button.configure(state=tk.DISABLED)
        self.position_var.set("--:--")
        self.duration_var.set("--:--")
        if self.mode == "faces":
            self._render_still_image()
            self.status_var.set(
                f"Ảnh: {self.source_name}  |  "
                f"Tìm thấy {len(self.current_faces)} khuôn mặt"
            )
        else:
            self._render_still_image()
            self.status_var.set(f"Đang hiển thị ảnh  |  {self.source_name}")
        self._update_transport_button()

    def open_webcam(self):
        if self.mode in ("balloons", "plate"):
            self.set_mode("rotate")
        self._stop_camera_reader()
        camera_sources = [
            (0, getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)),
            (0, getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)),
            (0, cv2.CAP_ANY),
            (1, cv2.CAP_ANY),
            (2, cv2.CAP_ANY),
        ]
        new_capture = None
        camera_errors = []
        for camera_index, backend in camera_sources:
            try:
                candidate = cv2.VideoCapture(camera_index, backend)
                if candidate.isOpened():
                    new_capture = candidate
                    break
                candidate.release()
            except cv2.error as error:
                camera_errors.append(str(error))

        if new_capture is None:
            detail = f" Chi tiết: {camera_errors[-1]}" if camera_errors else ""
            self.status_var.set(
                "Không mở được webcam (đã thử camera 0–2). "
                "Kiểm tra kết nối, quyền camera và ứng dụng đang sử dụng camera."
                + detail
            )
            return
        self.capture = new_capture
        self.video_path = None
        self.source_name = "Webcam"
        self.is_webcam = True
        self.paused = False
        self.duration_seconds = 0.0
        self.seek_scale.configure(state=tk.DISABLED)
        self.position_var.set("LIVE")
        self.duration_var.set("LIVE")
        self.replay_button.configure(state=tk.DISABLED)
        self.still_image = None
        self.still_image_path = None
        self.current_frame = None
        self.current_faces = []
        self.camera_stop_event = threading.Event()
        self.camera_frame = None
        self.camera_error = None
        try:
            new_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        self.camera_thread = threading.Thread(
            target=self._camera_reader, args=(new_capture, self.camera_stop_event),
            daemon=True,
        )
        self.camera_thread.start()
        self.status_var.set("Webcam đang phát")
        self._update_transport_button()

    def _camera_reader(self, capture, stop_event):
        try:
            while not stop_event.is_set():
                success, frame = capture.read()
                if not success or frame is None:
                    self.camera_error = "Không nhận được frame từ webcam."
                    break
                with self.camera_frame_lock:
                    self.camera_frame = frame
        except Exception as error:
            self.camera_error = f"Lỗi đọc webcam: {error}"

    def _stop_camera_reader(self):
        self.camera_stop_event.set()
        capture = self.capture
        if capture is not None:
            capture.release()
        thread = self.camera_thread
        if thread is not None and thread.is_alive() \
                and thread is not threading.current_thread():
            thread.join(timeout=0.25)
        self.capture = None
        self.camera_thread = None
        self.is_webcam = False

    def _update_transport_button(self):
        if self.capture is None:
            self.transport_button.configure(
                text="▶ Phát", state=tk.DISABLED
            )
        elif self.paused:
            self.transport_button.configure(
                text="▶ Phát tiếp", state=tk.NORMAL
            )
        else:
            self.transport_button.configure(
                text="⏸ Tạm dừng", state=tk.NORMAL
            )

    def toggle_pause(self):
        if self.capture is None:
            return
        self.paused = not self.paused
        self._update_transport_button()
        self.status_var.set(
            self._video_status("Đã tạm dừng" if self.paused else "Đang phát")
        )

    def _prepare_display_frame(self, frame):
        display = frame.copy()
        if self.brightness:
            display = np.clip(
                display.astype(np.int16) + int(self.brightness), 0, 255
            ).astype(np.uint8)
        if self.color_mode == "Grayscale":
            display = cv2.cvtColor(
                cv2.cvtColor(display, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
            )
        elif self.color_mode == "HSV":
            display = cv2.cvtColor(display, cv2.COLOR_BGR2HSV)
        height, width = display.shape[:2]
        center_x, center_y = width / 2, height / 2
        matrix = np.array(
            [
                [self.zoom, 0, (1 - self.zoom) * center_x + self.translate_x],
                [0, self.zoom, (1 - self.zoom) * center_y + self.translate_y],
            ],
            dtype=np.float32,
        )
        return cv2.warpAffine(
            display, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT,
            borderValue=(248, 250, 252),
        )

    def _render_still_image(self):
        if self.still_image is None:
            return
        display = self.still_image
        faces = []
        if self.mode == "faces":
            faces = self._find_faces(display)
            self.current_frame = display
            self.current_faces = faces
            display = self._draw_faces(display, faces)
        self._show_frame(self._prepare_display_frame(self._rotated_frame(display)))

    @staticmethod
    def _draw_faces(frame, faces):
        display = frame.copy()
        for x, y, width, height in faces:
            cv2.rectangle(
                display, (x, y), (x + width - 1, y + height - 1), (0, 160, 70), 2
            )
        cv2.putText(
            display, f"Faces: {len(faces)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 160, 70), 2,
        )
        return display

    def change_zoom(self, factor):
        self.zoom = min(4.0, max(0.25, self.zoom * factor))
        self.refresh_display()

    def translate(self, dx, dy):
        self.translate_x += dx
        self.translate_y += dy
        self.refresh_display()

    def reset_transform(self):
        self.zoom = 1.0
        self.translate_x = 0
        self.translate_y = 0
        self.brightness = 0
        self.brightness_var.set(0)
        self.refresh_display()

    def _brightness_changed(self, value):
        self.brightness = int(float(value))
        self.refresh_display()

    def _color_mode_changed(self, value):
        self.color_mode = value
        self.refresh_display()

    def refresh_display(self):
        if self.still_image is not None and self.capture is None:
            self._render_still_image()
        elif self.current_frame is not None:
            display = self.current_frame
            if self.mode == "faces":
                display = self._draw_faces(display, self.current_faces)
            self._show_frame(
                self._prepare_display_frame(self._rotated_frame(display))
            )

    def _video_status(self, message):
        if self.source_name:
            return f"{message}  |  {self.source_name}"
        return message

    def choose_video(self):
        path = filedialog.askopenfilename(
            title="Chọn file video",
            filetypes=(
                ("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Tất cả file", "*.*"),
            ),
        )
        if path:
            self.open_video(path)

    def choose_balloon_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh để đếm bóng bay",
            filetypes=(
                ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Tất cả file", "*.*"),
            ),
        )
        if path:
            self.open_balloon_image(path)

    def open_balloon_image(self, path):
        try:
            image = self._read_image(path)
        except (OSError, cv2.error) as error:
            self.status_var.set(f"Lỗi đọc ảnh: {error}")
            return
        if image is None:
            self.status_var.set(f"Không mở được ảnh: {path}")
            return

        self.balloon_image = image
        self.balloon_image_path = path
        self.balloon_detections = []
        self._show_frame(image)
        self.status_var.set(f"Đang đếm bóng bay  |  {Path(path).name}")
        self._process_balloon_image()

    def _process_balloon_image(self):
        try:
            self._detect_and_show_balloons()
        except Exception as error:
            self.status_var.set(f"Lỗi xử lý bóng bay: {error}")

    def choose_plate_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh có biển số xe",
            filetypes=(
                ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Tất cả file", "*.*"),
            ),
        )
        if path:
            self.open_plate_image(path)

    def open_plate_image(self, path):
        try:
            image = self._read_image(path)
        except (OSError, cv2.error) as error:
            self.status_var.set(f"Lỗi đọc ảnh: {error}")
            return
        if image is None:
            self.status_var.set(f"Không mở được ảnh: {path}")
            return

        self.plate_image = image
        self.plate_image_path = path
        self._show_frame(image)
        self.status_var.set(f"Đang tách ký tự  |  {Path(path).name}")
        self._process_plate_image()

    def _process_plate_image(self):
        try:
            self._detect_and_crop_plate_characters()
        except Exception as error:
            self.status_var.set(f"Lỗi xử lý biển số: {error}")

    def _detect_plates(self, image):
        image_height, image_width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(3, image_width // 100), max(3, image_height // 80)),
        )
        edge_mask = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, close_kernel, iterations=2
        )
        _, threshold_mask = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold_mask = cv2.morphologyEx(
            threshold_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2
        )

        candidates = []
        minimum_width = max(40, int(image_width * 0.12))
        minimum_height = max(16, int(image_height * 0.025))
        minimum_area = image_width * image_height * 0.002
        for mask in (edge_mask, threshold_mask, cv2.bitwise_not(threshold_mask)):
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, width, height = cv2.boundingRect(contour)
                ratio = width / max(height, 1)
                if (
                    area >= minimum_area
                    and width >= minimum_width
                    and height >= minimum_height
                    and 2.0 <= ratio <= 6.5
                ):
                    box = (x, y, x + width, y + height)
                    if not any(self._boxes_overlap(box, existing) for existing in candidates):
                        candidates.append(box)

        candidates.sort(
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
            reverse=True,
        )
        return candidates[:10]

    @staticmethod
    def _boxes_overlap(first, second):
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        union = first_area + second_area - intersection
        return union > 0 and (
            intersection / union > 0.5
            or intersection / min(first_area, second_area) > 0.7
        )

    @staticmethod
    def _segment_plate_characters(plate):
        plate_height, plate_width = plate.shape[:2]
        if plate_height < 12 or plate_width < 24:
            return []

        scale = 160 / plate_height
        resized_width = max(1, round(plate_width * scale))
        resized = cv2.resize(plate, (resized_width, 160), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        masks = [
            binary,
            cv2.bitwise_not(binary),
            cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 7,
            ),
        ]

        candidates = []
        for mask in masks:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            boxes = []
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                aspect_ratio = width / height if height else 0
                if (
                    height >= 0.2 * 160
                    and height <= 0.94 * 160
                    and width >= 0.018 * resized_width
                    and width <= 0.22 * resized_width
                    and 0.08 <= aspect_ratio <= 1.15
                ):
                    boxes.append((x, y, x + width, y + height))
            if len(boxes) > len(candidates):
                candidates = boxes

        if not candidates:
            return []

        outer_candidates = []
        for box in sorted(
            candidates,
            key=lambda item: (item[2] - item[0]) * (item[3] - item[1]),
            reverse=True,
        ):
            left, top, right, bottom = box
            area = (right - left) * (bottom - top)
            is_inner_contour = any(
                left >= outer[0]
                and top >= outer[1]
                and right <= outer[2]
                and bottom <= outer[3]
                and area < (outer[2] - outer[0]) * (outer[3] - outer[1]) * 0.7
                for outer in outer_candidates
            )
            if not is_inner_contour:
                outer_candidates.append(box)
        candidates = outer_candidates

        candidates.sort(key=lambda box: (box[1] + box[3]) / 2)
        rows = []
        for box in candidates:
            center_y = (box[1] + box[3]) / 2
            row = next(
                (items for items in rows if abs(items[0] - center_y) < 0.22 * 160),
                None,
            )
            if row is None:
                rows.append([center_y, [box]])
            else:
                row[1].append(box)

        characters = []
        for _, row_boxes in sorted(rows, key=lambda row: row[0]):
            for left, top, right, bottom in sorted(row_boxes, key=lambda box: box[0]):
                x1 = max(0, int(left / scale) - 2)
                y1 = max(0, int(top / scale) - 2)
                x2 = min(plate_width, int(right / scale) + 2)
                y2 = min(plate_height, int(bottom / scale) + 2)
                if x2 > x1 and y2 > y1:
                    characters.append((x1, y1, x2, y2))
        return characters

    def _detect_and_crop_plate_characters(self):
        plates = self._detect_plates(self.plate_image)
        display = self.plate_image.copy()
        output_dir = Path("output_data/plate_digits")
        os.makedirs("output_data/plate_digits", exist_ok=True)
        saved = 0
        character_boxes = []

        for plate_index, (left, top, right, bottom) in enumerate(plates, 1):
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                display,
                "Bien so",
                (left, max(20, top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            plate_crop = self.plate_image[top:bottom, left:right]
            for character_index, (x1, y1, x2, y2) in enumerate(
                self._segment_plate_characters(plate_crop), 1
            ):
                absolute_box = (left + x1, top + y1, left + x2, top + y2)
                character_boxes.append(absolute_box)
                character_crop = plate_crop[y1:y2, x1:x2]
                output_path = output_dir / (
                    f"plate_{plate_index:02d}_char_{character_index:02d}.png"
                )
                if self._write_image(output_path, character_crop):
                    saved += 1

        for left, top, right, bottom in character_boxes:
            cv2.rectangle(display, (left, top), (right, bottom), (255, 160, 0), 1)
        cv2.putText(
            display,
            f"Bien so: {len(plates)}  Ky tu da tach: {saved}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        self._show_frame(display)
        if saved:
            self.status_var.set(
                f"Đã tách {saved} ký tự, lưu trong {output_dir}"
            )
        elif plates:
            self.status_var.set(
                "Đã tìm thấy biển số nhưng chưa tách được ký tự rõ ràng"
            )
        else:
            self.status_var.set("Không tìm thấy biển số trong ảnh")

    def _detect_and_show_balloons(self):
        height, width = self.balloon_image.shape[:2]
        minimum_dimension = min(height, width)
        hsv = cv2.cvtColor(self.balloon_image, cv2.COLOR_BGR2HSV)
        saturated_colors = cv2.inRange(
            hsv, np.array((0, 40, 35)), np.array((179, 255, 255))
        )
        dark_objects = cv2.inRange(
            hsv, np.array((0, 0, 0)), np.array((179, 255, 75))
        )
        foreground = cv2.bitwise_or(saturated_colors, dark_objects)
        cleanup_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        )
        mask = cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN, cleanup_kernel
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        minimum_area = max(30.0, height * width * 0.0002)
        maximum_area = height * width * 0.15
        balloon_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not minimum_area <= area <= maximum_area:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            aspect = box_width / max(box_height, 1)
            if (
                box_width < minimum_dimension * 0.025
                or box_height < minimum_dimension * 0.035
                or not 0.3 <= aspect <= 1.2
            ):
                continue
            balloon_candidates.append((x, y, x + box_width, y + box_height))

        balloon_candidates.sort(key=lambda box: (box[1], box[0]))
        detections = [(*box, 1.0) for box in balloon_candidates]
        self.balloon_detections = detections
        display = self.balloon_image.copy()
        for left, top, right, bottom in balloon_candidates:
            cv2.rectangle(display, (left, top), (right, bottom), (0, 160, 70), 2)
            cv2.putText(
                display,
                "Bong bay",
                (left, max(20, top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 160, 70),
                2,
            )
        cv2.putText(
            display,
            f"So bong bay: {len(detections)}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 160, 70),
            2,
        )
        self._show_frame(display)
        output_dir = Path("output_data/balloons")
        os.makedirs("output_data/balloons", exist_ok=True)
        output_path = output_dir / f"{Path(self.balloon_image_path).stem}_counted.png"
        if not self._write_image(output_path, display):
            self.status_var.set(f"Không thể lưu ảnh kết quả đếm bóng bay: {output_path}")
            return
        self.status_var.set(
            f"Tìm thấy {len(detections)} bóng bay  |  "
            f"Đã lưu ảnh vào {output_path}"
        )

    def open_video(self, path):
        if self.mode in ("balloons", "plate"):
            self.set_mode("rotate")
        try:
            new_capture = cv2.VideoCapture(path)
            if not new_capture.isOpened():
                new_capture.release()
                self.status_var.set(f"Không mở được video: {path}")
                return
        except cv2.error as error:
            self.status_var.set(f"Lỗi mở video: {error}")
            return

        self._stop_camera_reader()
        if self.capture is not None:
            self.capture.release()
        self.capture = new_capture
        self.video_path = path
        self.source_name = Path(path).name
        self.is_webcam = False
        self.paused = False
        self.still_image = None
        self.still_image_path = None
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        frame_count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps > 0:
            self.frame_interval_ms = max(1, round(1000 / fps))
            self.duration_seconds = frame_count / fps
        else:
            self.duration_seconds = 0.0
        self.seek_scale.configure(
            to=max(self.duration_seconds, 0.1),
            state=tk.NORMAL if self.duration_seconds > 0 else tk.DISABLED,
        )
        self.seek_var.set(0)
        self.position_var.set("00:00")
        self.duration_var.set(self._format_time(self.duration_seconds))
        self.seeking = False
        self.replay_button.configure(state=tk.DISABLED)
        self.current_frame = None
        self._update_transport_button()
        self.status_var.set(self._video_status("Đang phát"))

    def replay_video(self):
        if self.video_path:
            self.open_video(self.video_path)

    @staticmethod
    def _format_time(seconds):
        total_seconds = max(0, int(seconds))
        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _seek_changed(self, value):
        self.position_var.set(self._format_time(float(value)))

    def _start_seek(self, _event):
        self.seeking = True

    def _finish_seek(self, _event):
        target_seconds = self.seek_var.get()
        self.seeking = False
        if not self.video_path:
            return
        if self.capture is None:
            self.open_video(self.video_path)
        if self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_MSEC, target_seconds * 1000)
            self.current_frame = None
            self.current_faces = []
            self.replay_button.configure(state=tk.DISABLED)
            self.status_var.set(self._video_status("Đang phát"))

    def rotate(self, direction):
        self.turns_clockwise = (self.turns_clockwise + direction) % 4
        self.refresh_display()

    def on_key(self, event):
        key = event.keysym.lower()
        if key in ("q", "escape"):
            self.close()
            return "break"
        if key == "space":
            self.toggle_pause()
            return "break"
        if self.mode == "rotate" and key == "a":
            self.rotate(-1)
            return "break"
        if self.mode == "rotate" and key == "d":
            self.rotate(1)
            return "break"
        if self.mode == "faces" and key == "s":
            self.save_faces()
            return "break"

    def _rotated_frame(self, frame):
        rotated = frame
        for _ in range(self.turns_clockwise):
            rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
        return rotated

    def _ensure_face_detector(self):
        if self.face_detector is not None or self.face_cascade is not None:
            return
        if self.detector_loading:
            return

        self.detector_loading = True
        self.status_var.set(self._video_status("Đang tải bộ nhận diện khuôn mặt"))

        def load_detector():
            try:
                if hasattr(cv2, "FaceDetectorYN_create"):
                    model_path = Path(__file__).with_name(
                        "face_detection_yunet_2023mar.onnx"
                    )
                    if not model_path.exists():
                        model_url = (
                            "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/"
                            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                        )
                        urlretrieve(model_url, model_path)
                    detector = cv2.FaceDetectorYN_create(
                        str(model_path), "", (320, 320), 0.8, 0.3, 5000
                    )
                    self.root.after(0, self._detector_ready, detector, None, None)
                else:
                    cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades
                        + "haarcascade_frontalface_default.xml"
                    )
                    if cascade.empty():
                        raise RuntimeError("Không tải được bộ nhận diện khuôn mặt.")
                    self.root.after(0, self._detector_ready, None, cascade, None)
            except Exception as error:
                try:
                    self.root.after(
                        0, self._detector_ready, None, None, str(error)
                    )
                except tk.TclError:
                    pass

        threading.Thread(target=load_detector, daemon=True).start()

    def _detector_ready(self, detector, cascade, error):
        self.detector_loading = False
        if error:
            self.status_var.set(f"Lỗi nhận diện: {error}")
            return

        self.face_detector = detector
        self.face_cascade = cascade
        if self.mode == "faces" and self.still_image is not None:
            self._render_still_image()
            self.status_var.set(
                f"Ảnh: {self.source_name}  |  "
                f"Tìm thấy {len(self.current_faces)} khuôn mặt"
            )
        elif self.mode == "faces":
            self.status_var.set(self._video_status("Đang nhận diện khuôn mặt"))

    def _find_faces(self, frame):
        frame_height, frame_width = frame.shape[:2]
        if self.face_detector is not None:
            self.face_detector.setInputSize((frame_width, frame_height))
            _, detections = self.face_detector.detect(frame)
            detections = detections if detections is not None else []
            faces = [
                tuple(int(value) for value in detection[:4])
                for detection in detections
            ]
        elif self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )
        else:
            return []

        bounded_faces = []
        for x, y, width, height in faces:
            left = max(0, x)
            top = max(0, y)
            right = min(frame_width, x + width)
            bottom = min(frame_height, y + height)
            if right > left and bottom > top:
                bounded_faces.append((left, top, right - left, bottom - top))
        return bounded_faces

    def _show_frame(self, frame):
        area_width = max(self.video_canvas.winfo_width(), 320)
        area_height = max(self.video_canvas.winfo_height(), 240)
        scale = min(
            area_width / frame.shape[1], area_height / frame.shape[0], 1.0
        )
        width = max(1, int(frame.shape[1] * scale))
        height = max(1, int(frame.shape[0] * scale))
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".png", resized)
        if not success:
            return

        image_data = base64.b64encode(encoded).decode("ascii")
        self.photo = tk.PhotoImage(data=image_data, format="png")
        position = (area_width // 2, area_height // 2)
        if self.image_item is None:
            self.video_canvas.delete(self.placeholder_item)
            self.image_item = self.video_canvas.create_image(
                *position, image=self.photo, anchor="center"
            )
        else:
            self.video_canvas.coords(self.image_item, *position)
            self.video_canvas.itemconfigure(self.image_item, image=self.photo)

    def update_frame(self):
        if self.closing:
            return

        frame_started = time.perf_counter()
        if self.capture is not None and not self.paused:
            try:
                camera_waiting = False
                if self.is_webcam:
                    if self.camera_error:
                        error = self.camera_error
                        self._stop_camera_reader()
                        self.paused = True
                        self._update_transport_button()
                        self.status_var.set(error)
                        success, frame = False, None
                        camera_waiting = True
                    else:
                        with self.camera_frame_lock:
                            frame = (
                                self.camera_frame.copy()
                                if self.camera_frame is not None else None
                            )
                        success = frame is not None
                        camera_waiting = not success
                else:
                    success, frame = self.capture.read()
                if success:
                    self.current_frame = frame
                    self.current_faces = []
                    if self.mode == "faces":
                        self.current_faces = self._find_faces(frame)
                        if self.is_webcam:
                            display_frame = cv2.flip(frame, 1)
                            mirrored_faces = [
                                (frame.shape[1] - x - width, y, width, height)
                                for x, y, width, height in self.current_faces
                            ]
                            display_frame = self._draw_faces(
                                display_frame, mirrored_faces
                            )
                        else:
                            display_frame = self._draw_faces(
                                frame, self.current_faces
                            )
                    else:
                        display_frame = frame
                    if self.is_webcam and self.mode != "faces":
                        display_frame = cv2.flip(display_frame, 1)
                    display_frame = self._rotated_frame(display_frame)
                    self._show_frame(self._prepare_display_frame(display_frame))
                    if not self.is_webcam:
                        position_seconds = (
                            self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        )
                        position_seconds = min(
                            position_seconds, self.duration_seconds
                        )
                        if not self.seeking:
                            self.seek_var.set(position_seconds)
                            self.position_var.set(
                                self._format_time(position_seconds)
                            )
                elif not camera_waiting:
                    was_webcam = self.is_webcam
                    self.capture.release()
                    self.capture = None
                    self.paused = True
                    self._update_transport_button()
                    if was_webcam:
                        self.status_var.set(
                            "Mất kết nối webcam. Kiểm tra camera rồi mở lại."
                        )
                    else:
                        self.seek_var.set(self.duration_seconds)
                        self.position_var.set(
                            self._format_time(self.duration_seconds)
                        )
                        self.replay_button.configure(state=tk.NORMAL)
                        self.status_var.set(self._video_status("Video đã phát hết"))
            except Exception as error:
                if self.is_webcam:
                    self._stop_camera_reader()
                elif self.capture is not None:
                    self.capture.release()
                    self.capture = None
                self.paused = True
                self._update_transport_button()
                self.status_var.set(f"Lỗi xử lý luồng video: {error}")

        if self.capture is None or self.paused or self.mode in ("balloons", "plate"):
            delay_ms = 100
        else:
            elapsed_ms = (time.perf_counter() - frame_started) * 1000
            delay_ms = max(1, round(self.frame_interval_ms - elapsed_ms))
        self.root.after(delay_ms, self.update_frame)

    def save_faces(self):
        if self.current_frame is None or not self.current_faces:
            self.status_var.set(self._video_status("Không có khuôn mặt để lưu"))
            return

        output_dir = Path("output_data/faces")
        os.makedirs("output_data/faces", exist_ok=True)
        timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000)}"
        saved = 0
        for index, (x, y, width, height) in enumerate(self.current_faces):
            face_image = self.current_frame[y:y + height, x:x + width]
            output_path = output_dir / f"face_{timestamp}_{index + 1:02d}.png"
            if cv2.imwrite(str(output_path), face_image):
                saved += 1
        self.status_var.set(
            self._video_status(
                f"Đã lưu {saved} ảnh khuôn mặt vào {output_dir}"
            )
        )

    def close(self):
        if self.closing:
            return
        self.closing = True
        if self.is_webcam:
            self._stop_camera_reader()
        elif self.capture is not None:
            self.capture.release()
        self.root.destroy()


def main():
    if _CV_IMPORT_ERROR is not None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Thiếu thư viện",
            "Không thể chạy ứng dụng vì thiếu OpenCV hoặc NumPy.\n"
            "Cài đặt bằng lệnh: pip install opencv-python numpy\n\n"
            f"Chi tiết: {_CV_IMPORT_ERROR}",
        )
        root.destroy()
        return

    parser = argparse.ArgumentParser(
        description="Image and video processing application."
    )
    parser.add_argument("video", nargs="?", help="Path to a video file")
    parser.add_argument(
        "--mode",
        choices=("rotate", "faces", "balloons", "plate"),
        default="rotate",
        help="Initial mode: rotate, faces, balloons, or plate",
    )
    args = parser.parse_args()

    root = tk.Tk()
    VideoApp(root, args.video, args.mode)
    root.mainloop()


if __name__ == "__main__":
    main()