"""Dark-mode Tkinter dashboard for six computer-vision exercises."""

from __future__ import annotations

import os
import tkinter as tk
from collections.abc import Callable
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


class ComputerVisionApp:
    """Single-window dashboard for the OpenCV workshop."""

    COLORS = {
        "background": "#1e1e2e",
        "sidebar": "#181825",
        "canvas": "#11111b",
        "button": "#313244",
        "selected": "#6366f1",
        "text": "#cdd6f4",
        "muted": "#a6adc8",
        "accent": "#cba6f7",
        "border": "#45475a",
        "green": "#a6e3a1",
    }
    WINDOW_TITLE = "CV Workshop 2026"
    DISPLAY_MAX_WIDTH = 850
    DISPLAY_MAX_HEIGHT = 580

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(self.WINDOW_TITLE)
        self.root.configure(bg=self.COLORS["background"])
        self.root.geometry("1200x780")
        self.root.minsize(1080, 700)
        self._center_window()

        self.cap: cv2.VideoCapture | None = None
        self.after_id: str | None = None
        self.video_mode: str | None = None
        self.current_angle = 0
        self.rotation_state = 0
        self.frame_delay_ms = 30
        self.face_cascade: cv2.CascadeClassifier | None = None
        self.current_frame: np.ndarray | None = None
        self.current_faces: list[tuple[int, int, int, int]] = []
        self.saved_face_count = 0
        self.display_frame: np.ndarray | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.active_task: str | None = None
        self.source_resolution = "—"
        self.detected_count: int | None = None
        self.ball_source: np.ndarray | None = None
        self.enhance_source: np.ndarray | None = None
        self.manual_ball_threshold = False

        self.threshold_var = tk.IntVar(value=127)
        self.brightness_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="Sẵn sàng")
        self.count_var = tk.StringVar(value="—")
        self.resolution_var = tk.StringVar(value="—")
        self._build_ui()
        self.root.bind("<Key>", self.handle_key_press)
        self.root.bind("<KeyPress-a>", lambda e: self._on_key_rotate('a'))
        self.root.bind("<KeyPress-A>", lambda e: self._on_key_rotate('a'))
        self.root.bind("<KeyPress-d>", lambda e: self._on_key_rotate('d'))
        self.root.bind("<KeyPress-D>", lambda e: self._on_key_rotate('d'))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _center_window(self) -> None:
        self.root.update_idletasks()
        width, height = 1200, 780
        x = max(0, (self.root.winfo_screenwidth() - width) // 2)
        y = max(0, (self.root.winfo_screenheight() - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _build_ui(self) -> None:
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, minsize=320, weight=0)
        self.root.grid_columnconfigure(1, weight=1)

        sidebar = tk.Frame(
            self.root, bg=self.COLORS["sidebar"], width=320, padx=20, pady=22
        )
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        tk.Label(
            sidebar,
            text="CV WORKSHOP 2026",
            bg=self.COLORS["sidebar"],
            fg=self.COLORS["accent"],
            font=("Segoe UI", 17, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 22))

        self.task_buttons: dict[str, tk.Button] = {}
        tasks: tuple[tuple[str, str, Callable[[], None]], ...] = (
            ("camera", "1. Camera & Grayscale", self.start_camera_grayscale),
            ("rotate", "2. Video Xoay A/D", self.start_rotating_video),
            ("faces", "3. Nhận Diện Khuôn Mặt", self.start_face_detection),
            ("balls", "4. Đếm 12 Quả Bóng", self.count_balls),
            ("plate", "5. Cắt Ký Tự Biển Số", self.extract_plate_characters),
            ("enhance", "6. Cân Bằng Sáng & Resize", self.enhance_brightness_and_resize),
        )
        for key, label, command in tasks:
            button = tk.Button(
                sidebar,
                text=label,
                command=command,
                anchor="w",
                padx=14,
                pady=10,
                relief="flat",
                bd=0,
                bg=self.COLORS["button"],
                fg=self.COLORS["text"],
                activebackground=self.COLORS["selected"],
                activeforeground="white",
                font=("Segoe UI", 10, "bold"),
                cursor="hand2",
            )
            button.pack(fill="x", pady=4)
            self.task_buttons[key] = button

        tk.Frame(sidebar, bg=self.COLORS["border"], height=1).pack(
            fill="x", pady=14
        )
        self._make_control_button(
            sidebar, "Dừng Luồng (Stop)", self.stop_media
        ).pack(fill="x", pady=4)
        self._make_control_button(
            sidebar, "Lưu Khuôn Mặt (Phím S)", self.save_detected_faces
        ).pack(fill="x", pady=4)

        tk.Label(
            sidebar,
            text="Ngưỡng Nhị Phân (0 - 255)",
            bg=self.COLORS["sidebar"],
            fg=self.COLORS["text"],
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(18, 2))
        self.threshold_scale = tk.Scale(
            sidebar,
            from_=0,
            to=255,
            orient="horizontal",
            variable=self.threshold_var,
            command=self._on_threshold_change,
            bg=self.COLORS["sidebar"],
            fg=self.COLORS["text"],
            troughcolor=self.COLORS["button"],
            highlightthickness=0,
            activebackground=self.COLORS["selected"],
            showvalue=True,
            font=("Segoe UI", 9),
        )
        self.threshold_scale.pack(fill="x")

        tk.Label(
            sidebar,
            text="Độ Sáng (0 - 100)",
            bg=self.COLORS["sidebar"],
            fg=self.COLORS["text"],
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(14, 2))
        self.brightness_scale = tk.Scale(
            sidebar,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.brightness_var,
            command=self._on_brightness_change,
            bg=self.COLORS["sidebar"],
            fg=self.COLORS["text"],
            troughcolor=self.COLORS["button"],
            highlightthickness=0,
            activebackground=self.COLORS["selected"],
            showvalue=True,
            font=("Segoe UI", 9),
        )
        self.brightness_scale.pack(fill="x")

        content = tk.Frame(self.root, bg=self.COLORS["background"], padx=16, pady=14)
        content.grid(row=0, column=1, sticky="nsew")
        content.grid_rowconfigure(1, weight=1)
        content.grid_columnconfigure(0, weight=1)

        tk.Label(
            content,
            text="Computer Vision Dashboard",
            bg=self.COLORS["background"],
            fg=self.COLORS["text"],
            font=("Segoe UI", 18, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", pady=(2, 12))

        canvas_frame = tk.Frame(
            content,
            bg=self.COLORS["canvas"],
            highlightbackground=self.COLORS["border"],
            highlightthickness=1,
        )
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            canvas_frame,
            bg=self.COLORS["canvas"],
            highlightthickness=0,
            width=self.DISPLAY_MAX_WIDTH,
            height=self.DISPLAY_MAX_HEIGHT,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._redraw_current_image)
        self._draw_welcome()

        status_card = tk.Frame(
            content,
            bg=self.COLORS["sidebar"],
            padx=14,
            pady=10,
            highlightbackground=self.COLORS["border"],
            highlightthickness=1,
        )
        status_card.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        status_card.grid_columnconfigure(0, weight=3)
        status_card.grid_columnconfigure(1, weight=1)
        status_card.grid_columnconfigure(2, weight=1)
        self._status_item(status_card, "TRẠNG THÁI", self.status_var, 0)
        self._status_item(status_card, "PHÁT HIỆN", self.count_var, 1)
        self._status_item(status_card, "ĐỘ PHÂN GIẢI", self.resolution_var, 2)

    def _make_control_button(
        self, parent: tk.Widget, text: str, command: Callable[[], None]
    ) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            anchor="w",
            padx=14,
            pady=8,
            relief="flat",
            bd=0,
            bg=self.COLORS["button"],
            fg=self.COLORS["text"],
            activebackground=self.COLORS["selected"],
            activeforeground="white",
            font=("Segoe UI", 9),
            cursor="hand2",
        )

    def _status_item(
        self, parent: tk.Frame, title: str, variable: tk.StringVar, column: int
    ) -> None:
        item = tk.Frame(parent, bg=self.COLORS["sidebar"])
        item.grid(row=0, column=column, sticky="ew", padx=(0, 12))
        tk.Label(
            item,
            text=title,
            bg=self.COLORS["sidebar"],
            fg=self.COLORS["muted"],
            font=("Segoe UI", 8, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            item,
            textvariable=variable,
            bg=self.COLORS["sidebar"],
            fg=self.COLORS["text"],
            font=("Segoe UI", 9),
            anchor="w",
            wraplength=360,
        ).pack(fill="x", pady=(3, 0))

    def _draw_welcome(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(
            max(1, self.canvas.winfo_width() // 2),
            max(1, self.canvas.winfo_height() // 2),
            text="Chọn một bài tập ở thanh điều khiển",
            fill=self.COLORS["muted"],
            font=("Segoe UI", 15),
        )

    def _set_task(self, task: str, status: str) -> None:
        self.active_task = task
        for key, button in self.task_buttons.items():
            selected = key == task
            button.configure(
                bg=self.COLORS["selected"] if selected else self.COLORS["button"],
                fg="white" if selected else self.COLORS["text"],
            )
        self.status_var.set(status)
        self.detected_count = None
        self.count_var.set("—")
        self.source_resolution = "—"
        self.resolution_var.set("—")

    def _set_resolution(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        self.source_resolution = f"{width} × {height}"
        self.resolution_var.set(self.source_resolution)

    def _set_count(self, count: int) -> None:
        self.detected_count = count
        self.count_var.set(str(count))

    def _show_error(self, title: str, error: Exception) -> None:
        self.status_var.set(str(error))
        messagebox.showerror(title, str(error), parent=self.root)

    def _show_image(self, image_bgr: np.ndarray) -> None:
        self.display_frame = image_bgr.copy()
        self._render_display_frame()

    def _render_display_frame(self) -> None:
        if self.display_frame is None:
            return
        image_rgb = cv2.cvtColor(self.display_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        max_width = min(
            self.DISPLAY_MAX_WIDTH, max(1, self.canvas.winfo_width() - 20)
        )
        max_height = min(
            self.DISPLAY_MAX_HEIGHT, max(1, self.canvas.winfo_height() - 20)
        )
        pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self.tk_image,
            anchor="center",
        )

    def _redraw_current_image(self, _event: tk.Event | None = None) -> None:
        if self.display_frame is not None:
            self._render_display_frame()
        else:
            self._draw_welcome()

    def _stop_stream(self) -> None:
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_mode = None
        self.current_frame = None
        self.current_faces = []

    def stop_media(self) -> None:
        self._stop_stream()
        self.active_task = None
        for button in self.task_buttons.values():
            button.configure(bg=self.COLORS["button"], fg=self.COLORS["text"])
        self.status_var.set("Đã dừng camera/video")
        self.count_var.set("—")

    def _open_camera(self) -> cv2.VideoCapture:
        for camera_index in range(3):
            candidate = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if candidate.isOpened():
                return candidate
            candidate.release()
        raise RuntimeError("Không thể mở camera. Vui lòng kiểm tra webcam.")

    def start_camera_grayscale(self) -> None:
        self._stop_stream()
        self._set_task("camera", "Bài 1 đang khởi động camera")
        try:
            self.cap = self._open_camera()
            self.video_mode = "grayscale"
            self.status_var.set("Bài 1 đang chạy · nhấn Q hoặc Stop để dừng")
            self._schedule_frame()
        except (cv2.error, RuntimeError) as error:
            self._stop_stream()
            self._show_error("Lỗi camera", error)

    def start_rotating_video(self) -> None:
        self.root.focus_set()
        self.root.focus_force()
        self._stop_stream()
        self._set_task("rotate", "Chọn video để bắt đầu Bài 2")
        self.root.bind("<KeyPress-a>", lambda e: self._on_key_rotate('a'))
        self.root.bind("<KeyPress-A>", lambda e: self._on_key_rotate('a'))
        self.root.bind("<KeyPress-d>", lambda e: self._on_key_rotate('d'))
        self.root.bind("<KeyPress-D>", lambda e: self._on_key_rotate('d'))
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Chọn video",
            filetypes=(("Video", "*.mp4 *.avi *.mov *.mkv"),),
        )
        if not path:
            self.status_var.set("Sẵn sàng")
            return
        try:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError("Không thể mở file video đã chọn.")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_delay_ms = max(1, int(1000 / fps)) if fps > 0 else 30
            self.current_angle = 0
            self.rotation_state = 0
            self.video_mode = "rotate"
            self.status_var.set("Bài 2 đang chạy · phím A/D xoay, Q dừng")
            self.root.focus_set()
            self.root.focus_force()
            self._schedule_frame(self.frame_delay_ms)
        except (cv2.error, RuntimeError) as error:
            self._stop_stream()
            self._show_error("Lỗi video", error)

    def start_face_detection(self) -> None:
        self._stop_stream()
        self._set_task("faces", "Đang khởi động nhận diện khuôn mặt")
        try:
            cascade_path = (
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                self.face_cascade = None
                self.status_var.set("Lỗi: Không tìm thấy file haarcascade XML")
                return
            os.makedirs("dataset_faces", exist_ok=True)
            self.saved_face_count = self._next_numeric_filename("dataset_faces")
            self.cap = self._open_camera()
            self.video_mode = "faces"
            self.status_var.set("Bài 3 đang chạy · nhấn S lưu mặt, Q dừng")
            self._schedule_frame()
        except (cv2.error, OSError, RuntimeError) as error:
            self._stop_stream()
            self._show_error("Lỗi nhận diện khuôn mặt", error)

    def _schedule_frame(self, delay_ms: int = 30) -> None:
        self.after_id = self.root.after(delay_ms, self._update_frame)

    def _update_frame(self) -> None:
        self.after_id = None
        if self.cap is None or self.video_mode is None:
            return
        try:
            success, frame = self.cap.read()
            if not success:
                self._stop_stream()
                self.status_var.set("Video đã phát hết")
                return
            self._set_resolution(frame)

            if self.video_mode == "grayscale":
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                display_frame = np.hstack(
                    (frame, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                )
                cv2.putText(
                    display_frame, "COLOR", (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.putText(
                    display_frame, "GRAYSCALE", (frame.shape[1] + 16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
            elif self.video_mode == "rotate":
                if self.current_angle == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.current_angle == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.current_angle == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                display_frame = frame
            else:
                if self.face_cascade is None:
                    raise RuntimeError("Bộ phân loại khuôn mặt chưa được khởi tạo.")
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(60, 60),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                self.current_frame = frame.copy()
                self.current_faces = [
                    (int(x), int(y), int(w), int(h)) for x, y, w, h in faces
                ]
                for x, y, width, height in self.current_faces:
                    cv2.rectangle(
                        frame, (x, y), (x + width, y + height), (0, 255, 0), 2
                    )
                cv2.putText(
                    frame,
                    f"So luong mat: {len(self.current_faces)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                self._set_count(len(self.current_faces))
                display_frame = frame

            self._show_image(display_frame)
            if self.video_mode is not None:
                delay = self.frame_delay_ms if self.video_mode == "rotate" else 30
                self._schedule_frame(delay)
        except (cv2.error, RuntimeError, AttributeError) as error:
            self._stop_stream()
            self._show_error("Lỗi xử lý khung hình", error)

    def _on_key_rotate(self, key: str) -> None:
        if key == "a":
            self.current_angle = (self.current_angle - 90) % 360
        elif key == "d":
            self.current_angle = (self.current_angle + 90) % 360
        self.rotation_state = (self.current_angle // 90) % 4
        self.root.focus_set()

    def handle_key_press(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key == "q" and self.video_mode is not None:
            self.stop_media()
        elif self.video_mode == "rotate" and key in ("a", "d"):
            self._on_key_rotate(key)
        elif key == "s" and self.video_mode == "faces":
            self.save_detected_faces()

    def save_detected_faces(self) -> None:
        if self.video_mode != "faces" or self.current_frame is None:
            self.status_var.set("Hãy chạy Bài 3 trước khi lưu khuôn mặt")
            return
        if not self.current_faces:
            self.status_var.set("Không phát hiện khuôn mặt trong khung hiện tại")
            return
        try:
            os.makedirs("dataset_faces", exist_ok=True)
            self.saved_face_count = max(
                self.saved_face_count,
                self._next_numeric_filename("dataset_faces"),
            )
            saved = 0
            for x, y, width, height in self.current_faces:
                while os.path.exists(
                    os.path.join("dataset_faces", f"{self.saved_face_count}.png")
                ):
                    self.saved_face_count += 1
                crop = self.current_frame[y : y + height, x : x + width]
                output_path = os.path.join(
                    "dataset_faces", f"{self.saved_face_count}.png"
                )
                if not cv2.imwrite(output_path, crop):
                    raise RuntimeError(f"Không thể lưu ảnh: {output_path}")
                self.saved_face_count += 1
                saved += 1
            self.status_var.set(
                f"Đã lưu {saved} khuôn mặt vào thư mục dataset_faces"
            )
        except (cv2.error, OSError, RuntimeError) as error:
            self._show_error("Lỗi lưu khuôn mặt", error)

    def count_balls(self) -> None:
        self._stop_stream()
        self.ball_source = None
        self._set_task("balls", "Chọn ảnh 12 quả bóng")
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Chọn ảnh các quả bóng",
            filetypes=(("Image", "*.png *.jpg *.jpeg *.bmp"),),
        )
        if not path:
            self.status_var.set("Sẵn sàng")
            return
        try:
            image = cv2.imread(path)
            if image is None:
                raise RuntimeError("Không thể đọc file ảnh đã chọn.")
            self.ball_source = image.copy()
            self._set_resolution(image)
            self._process_ball_image()
        except (cv2.error, OSError, RuntimeError) as error:
            self._show_error("Lỗi đếm bóng", error)

    def _process_ball_image(self) -> None:
        if self.ball_source is None:
            return
        image = self.ball_source.copy()
        scale = 800.0 / image.shape[1]
        image = cv2.resize(
            image,
            (800, max(1, int(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        if self.manual_ball_threshold:
            _, thresh = cv2.threshold(
                blur,
                self.threshold_var.get(),
                255,
                cv2.THRESH_BINARY_INV,
            )
        else:
            _, thresh = cv2.threshold(
                blur,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        h_img = image.shape[0]
        valid_contours: list[np.ndarray] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            _, y, _, height = cv2.boundingRect(contour)
            if 800 < area < 35000 and (y + height) < (h_img - 15):
                valid_contours.append(contour)
        cv2.drawContours(image, valid_contours, -1, (0, 255, 0), 2)
        count = len(valid_contours)
        cv2.putText(
            image,
            f"So luong: {count}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 255),
            3,
        )
        self._show_image(image)
        self._set_count(count)
        threshold_mode = (
            f"ngưỡng thủ công {self.threshold_var.get()}"
            if self.manual_ball_threshold
            else "Otsu"
        )
        self.status_var.set(
            f"Bài 4 · Đã nhận diện {count} quả bóng ({threshold_mode})"
        )

    def _on_threshold_change(self, _value: str) -> None:
        self.manual_ball_threshold = True
        if self.active_task == "balls" and self.ball_source is not None:
            try:
                self._process_ball_image()
            except cv2.error as error:
                self._show_error("Lỗi phân ngưỡng ảnh bóng", error)

    def extract_plate_characters(self, file_path: str | None = None) -> None:
        self._stop_stream()
        self.ball_source = None
        self.enhance_source = None
        self._set_task("plate", "Chọn ảnh biển số xe")

        # 1. Đọc ảnh và tiền xử lý
        path = file_path or filedialog.askopenfilename(
            parent=self.root,
            title="Chọn ảnh biển số xe",
            filetypes=(("Image", "*.png *.jpg *.jpeg *.bmp"),),
        )
        if not path:
            self.status_var.set("Sẵn sàng")
            return

        try:
            image = cv2.imread(path)
            if image is None:
                raise RuntimeError("Không thể đọc file ảnh đã chọn.")
            self._process_plate_image(image)
        except (cv2.error, OSError, RuntimeError) as error:
            self._show_error("Lỗi xử lý biển số", error)

    def _process_plate_image(self, image: np.ndarray) -> None:
        # Resize ảnh về width = 900px (tính height theo đúng tỉ lệ aspect ratio)
        orig_h, orig_w = image.shape[:2]
        target_w = 900
        target_h = max(1, int(orig_h * (target_w / float(orig_w))))
        image = cv2.resize(
            image,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA if orig_w > target_w else cv2.INTER_LINEAR,
        )
        self._set_resolution(image)

        # Chuyển sang ảnh xám và làm mờ nhẹ
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Phân ngưỡng nhị phân thích ứng
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # 3. Lọc chính xác các ký tự biển số (loại bỏ dòng chữ slide phía trên và viền ngoài)
        valid_chars: list[tuple[int, int, int, int]] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            # Điều kiện nhận diện ký tự biển số trên ảnh width 900px:
            # - Chiều cao ký tự h từ 35px đến 160px
            # - Chiều rộng w từ 12px đến 90px
            # - Tỷ lệ w/h từ 0.2 đến 0.95
            # - Diện tích contour > 250 và không chiếm toàn bộ khung
            if 35 <= h <= 160 and 12 <= w <= 90 and 0.2 <= (w / float(h)) <= 0.95 and area > 250:
                valid_chars.append((x, y, w, h))

        # Loại bỏ các box lồng nhau (ví dụ lỗ bên trong ký tự) nếu có
        cleaned_chars: list[tuple[int, int, int, int]] = []
        for i, b1 in enumerate(valid_chars):
            x1, y1, w1, h1 = b1
            is_inside = any(
                i != j
                and x2 <= x1
                and y2 <= y1
                and (x2 + w2) >= (x1 + w1)
                and (y2 + h2) >= (y1 + h1)
                for j, (x2, y2, w2, h2) in enumerate(valid_chars)
            )
            if not is_inside:
                cleaned_chars.append(b1)
        valid_chars = cleaned_chars

        # 4. Sắp xếp và trích xuất ký tự
        valid_chars.sort(key=lambda b: b[0])
        os.makedirs("digits_plate", exist_ok=True)
        for i, (x, y, w, h) in enumerate(valid_chars):
            crop = image[y : y + h, x : x + w].copy()
            cv2.imwrite(f"digits_plate/digit_{i}.png", crop)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 5. Hiển thị kết quả lên Canvas Tkinter
        self._show_image(image)
        self._set_count(len(valid_chars))
        self.status_var.set(
            f"Bài 5: Đã trích xuất thành công {len(valid_chars)} ký tự vào thư mục digits_plate"
        )

    def enhance_brightness_and_resize(self) -> None:
        self._process_selected_image(
            "enhance",
            "Chọn ảnh cần cân bằng sáng",
            self._process_enhanced_image,
        )

    def _process_enhanced_image(self, image: np.ndarray) -> None:
        self.enhance_source = image.copy()
        self._set_resolution(image)
        self._render_enhanced_image()

    def _render_enhanced_image(self) -> None:
        if self.enhance_source is None:
            return
        image = self.enhance_source.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        brightness = self.brightness_var.get()
        if brightness:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.0, beta=brightness)
        resized = cv2.resize(
            enhanced, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR
        )
        target_height, target_width = resized.shape[:2]
        original = cv2.resize(
            image, (target_width, target_height), interpolation=cv2.INTER_LINEAR
        )
        comparison = np.hstack((original, resized))
        cv2.putText(
            comparison,
            "Original",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            comparison,
            f"Enhanced + Resize 1.5x | Brightness +{brightness}",
            (target_width + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        self._show_image(comparison)
        self.status_var.set(
            f"Bài 6 · Cân bằng sáng, resize 1.5× · Độ sáng +{brightness}"
        )

    def _on_brightness_change(self, _value: str) -> None:
        if self.active_task == "enhance" and self.enhance_source is not None:
            try:
                self._render_enhanced_image()
            except cv2.error as error:
                self._show_error("Lỗi điều chỉnh độ sáng", error)

    def _process_selected_image(
        self,
        task: str,
        title: str,
        processor: Callable[[np.ndarray], None],
        error_title: str = "Lỗi xử lý ảnh",
    ) -> None:
        self._stop_stream()
        self.ball_source = None
        self.enhance_source = None
        self._set_task(task, title)
        path = filedialog.askopenfilename(
            parent=self.root,
            title=title,
            filetypes=(("Image", "*.png *.jpg *.jpeg *.bmp"),),
        )
        if not path:
            self.status_var.set("Sẵn sàng")
            return
        try:
            image = cv2.imread(path)
            if image is None:
                raise RuntimeError("Không thể đọc file ảnh đã chọn.")
            processor(image)
        except (cv2.error, OSError, RuntimeError) as error:
            self._show_error(error_title, error)

    def _next_numeric_filename(self, directory: str) -> int:
        numbers: list[int] = []
        for name in os.listdir(directory):
            stem, extension = os.path.splitext(name)
            if extension.lower() == ".png" and stem.isdigit():
                numbers.append(int(stem))
        return max(numbers, default=-1) + 1

    def _on_close(self) -> None:
        self._stop_stream()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ComputerVisionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
