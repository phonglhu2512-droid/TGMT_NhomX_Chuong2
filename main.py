import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np


class ImageConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chuyển đổi ảnh OpenCV + PIL")
        self.root.geometry("1400x600")

        # Biến lưu đường dẫn ảnh và ảnh gốc đã đọc từ OpenCV
        self.image_path = ""
        self.original_bgr = None

        # Tạo frame chứa các nút điều khiển
        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(fill="x")

        self.open_btn = ttk.Button(control_frame, text="Chọn ảnh 1", command=self.open_image)
        self.open_btn.pack(side="left", padx=5)

        self.convert_btn = ttk.Button(control_frame, text="Chuyển đổi", command=self.convert_image, state="disabled")
        self.convert_btn.pack(side="left", padx=5)

        self.bitwise_btn = ttk.Button(
            control_frame,
            text="Thực hiện Bitwise AND hai ảnh",
            command=self.bitwise_and_images,
            state="disabled"
        )
        self.bitwise_btn.pack(side="left", padx=5)

        self.save_btn = ttk.Button(
            control_frame,
            text="Lưu ảnh (Bài 4)",
            command=self.save_current_image,
            state="disabled"
        )
        self.save_btn.pack(side="left", padx=5)

        self.brightness_label = ttk.Label(control_frame, text="Độ sáng: 0")
        self.brightness_label.pack(side="left", padx=(10, 0))

        self.brightness_scale = tk.Scale(
            control_frame,
            from_=-100,
            to=100,
            orient="horizontal",
            length=180,
            command=self.on_brightness_change,
            showvalue=False
        )
        self.brightness_scale.set(0)
        self.brightness_scale.pack(side="left", padx=5)

        self.brightness_btn = ttk.Button(
            control_frame,
            text="Tăng độ sáng",
            command=self.apply_brightness,
            state="disabled"
        )
        self.brightness_btn.pack(side="left", padx=5)

        self.rotate_btn = ttk.Button(
            control_frame,
            text="Xoay 90 độ",
            command=self.rotate_image_90,
            state="disabled"
        )
        self.rotate_btn.pack(side="left", padx=5)

        self.translate_btn = ttk.Button(
            control_frame,
            text="Dịch phải 50px",
            command=self.translate_image_right,
            state="disabled"
        )
        self.translate_btn.pack(side="left", padx=5)

        self.scale_btn = ttk.Button(
            control_frame,
            text="Thu phóng 1.5x",
            command=self.scale_image,
            state="disabled"
        )
        self.scale_btn.pack(side="left", padx=5)

        self.exit_btn = ttk.Button(control_frame, text="Thoát", command=root.destroy)
        self.exit_btn.pack(side="left", padx=5)

        self.status_label = ttk.Label(control_frame, text="Chưa có ảnh nào được chọn")
        self.status_label.pack(side="left", padx=10)

        # Tạo vùng hiển thị ảnh: gốc, Grayscale, HSV, Bitwise AND
        display_frame = ttk.Frame(root)
        display_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.labels = {}
        for name, title in [("original", "Ảnh gốc"), ("gray", "Grayscale"), ("hsv", "HSV"), ("bitwise", "Bitwise AND")]:
            frame = ttk.Frame(display_frame)
            frame.pack(side="left", fill="both", expand=True, padx=10)

            ttk.Label(frame, text=title, font=("Segoe UI", 11, "bold")).pack(anchor="center")
            label = tk.Label(frame, bg="black", width=30, height=18)
            label.pack(fill="both", expand=True, padx=5, pady=5)
            self.labels[name] = label

    def open_image(self):
        """Mở hộp thoại chọn ảnh 1 từ máy tính."""
        # Hộp thoại chọn file, chỉ chấp nhận ảnh có định dạng thông dụng
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh 1",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff")]
        )

        # Nếu người dùng bấm Cancel hoặc đóng cửa sổ thì trả về
        if not file_path:
            self.status_label.config(text="Người dùng đã hủy chọn ảnh 1.")
            return

        # Đọc ảnh bằng OpenCV
        loaded_image = self.read_image(file_path)
        if loaded_image is None:
            self.status_label.config(text="File ảnh 1 không hợp lệ hoặc bị hỏng.")
            messagebox.showerror("Lỗi ảnh", "Không thể đọc ảnh 1. Vui lòng chọn file ảnh hợp lệ.")
            return

        self.image_path = file_path
        self.original_bgr = loaded_image

        # Hiển thị ảnh gốc lên giao diện
        self.show_original_image()

        # Kích hoạt các chức năng phụ thuộc vào ảnh gốc
        self.convert_btn.config(state="normal")
        self.bitwise_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.brightness_btn.config(state="normal")
        self.rotate_btn.config(state="normal")
        self.translate_btn.config(state="normal")
        self.scale_btn.config(state="normal")
        self.status_label.config(text="Đã chọn ảnh 1. Có thể thực hiện tất cả các thao tác xử lý ảnh.")

    def read_image(self, file_path):
        """Đọc ảnh từ đường dẫn và trả về ảnh dạng NumPy dưới dạng BGR của OpenCV."""
        # Kiểm tra đường dẫn có tồn tại không
        if not file_path or not os.path.exists(file_path):
            return None

        # cv2.imread trả về None nếu file không đọc được
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        return image

    def show_original_image(self):
        """Hiển thị ảnh gốc sau khi chuyển từ BGR sang RGB để PIL hiển thị đúng màu."""
        if self.original_bgr is None:
            return

        rgb_image = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        self.display_image(self.labels["original"], pil_image)

    def convert_image(self):
        """Chuyển ảnh sang Grayscale và HSV khi người dùng bấm nút."""
        if self.original_bgr is None:
            self.status_label.config(text="Vui lòng chọn ảnh trước khi chuyển đổi.")
            return

        # 1. Chuyển ảnh sang ảnh xám (Grayscale)
        gray_image = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Chuyển ảnh sang không gian màu HSV
        hsv_image = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2HSV)

        # Hiển thị kết quả từng ảnh lên label tương ứng
        self.display_image(self.labels["gray"], Image.fromarray(gray_image, mode="L"))
        self.display_image(self.labels["hsv"], Image.fromarray(hsv_image, mode="HSV"))

        self.status_label.config(text="Đã chuyển đổi thành Grayscale và HSV.")

    def bitwise_and_images(self):
        """Thực hiện phép toán cv2.bitwise_and giữa ảnh 1 và ảnh 2 sau khi tự động resize."""
        if self.original_bgr is None:
            self.status_label.config(text="Vui lòng chọn ảnh 1 trước khi thực hiện Bitwise AND.")
            messagebox.showwarning("Cảnh báo", "Bạn cần chọn ảnh 1 trước.")
            return

        # Bước 1: chọn ảnh 2 từ máy tính
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh 2",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff")]
        )

        # Nếu người dùng hủy chọn ảnh 2, xử lý lỗi nhẹ và không làm gián đoạn
        if not file_path:
            self.status_label.config(text="Đã hủy chọn ảnh 2. Không thực hiện Bitwise AND.")
            messagebox.showinfo("Thông báo", "Bạn đã hủy chọn ảnh 2.")
            return

        # Bước 2: đọc ảnh 2
        second_image = self.read_image(file_path)
        if second_image is None:
            self.status_label.config(text="Ảnh 2 không hợp lệ hoặc bị hỏng.")
            messagebox.showerror("Lỗi ảnh", "Không thể đọc ảnh 2. Vui lòng chọn file ảnh hợp lệ.")
            return

        # Bước 3: chuẩn hóa ảnh 1 và ảnh 2 để cùng kích thước và số kênh màu
        # - Nếu ảnh có 1 kênh (grayscale) thì chuyển thành 3 kênh BGR để tương thích.
        # - Nếu ảnh 2 không cùng kích thước, tự động resize về kích thước của ảnh 1.
        first_image = self.original_bgr
        second_image = self.normalize_image_for_bitwise(first_image, second_image)

        # Bước 4: áp dụng phép toán bitwise AND
        result_image = cv2.bitwise_and(first_image, second_image)

        # Bước 5: hiển thị kết quả lên label Bitwise AND trong giao diện
        rgb_result = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        self.display_image(self.labels["bitwise"], Image.fromarray(rgb_result))

        self.status_label.config(text="Đã thực hiện Bitwise AND trên hai ảnh thành công.")

    def normalize_image_for_bitwise(self, first_image, second_image):
        """Đảm bảo ảnh 2 cùng kích thước và số kênh với ảnh 1 trước khi dùng cv2.bitwise_and()."""
        # Chuyển ảnh grayscale về BGR nếu cần
        if first_image.ndim == 2:
            first_image = cv2.cvtColor(first_image, cv2.COLOR_GRAY2BGR)
        if second_image.ndim == 2:
            second_image = cv2.cvtColor(second_image, cv2.COLOR_GRAY2BGR)

        # Resize ảnh 2 về cùng kích thước với ảnh 1
        height, width = first_image.shape[:2]
        second_image = cv2.resize(second_image, (width, height))

        # Nếu số kênh khác nhau, chuyển về cùng loại kênh
        if first_image.shape[2] != second_image.shape[2]:
            if first_image.shape[2] == 3 and second_image.shape[2] == 1:
                second_image = cv2.cvtColor(second_image, cv2.COLOR_GRAY2BGR)
            elif first_image.shape[2] == 1 and second_image.shape[2] == 3:
                first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
                second_image = cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY)

        return second_image

    def save_current_image(self):
        """Lưu ảnh đang hiển thị/đang chọn ra file theo định dạng do người dùng chọn."""
        if self.original_bgr is None:
            self.status_label.config(text="Chưa có ảnh nào được tải lên. Hãy chọn ảnh trước.")
            messagebox.showwarning("Cảnh báo", "Bạn chưa chọn ảnh nào. Vui lòng tải ảnh trước khi lưu.")
            return

        # Hộp thoại lưu file với các định dạng cho phép
        file_path = filedialog.asksaveasfilename(
            title="Lưu ảnh",
            defaultextension=".png",
            filetypes=[
                ("PNG (*.png)", "*.png"),
                ("JPEG (*.jpg; *.jpeg)", "*.jpg; *.jpeg"),
                ("BMP (*.bmp)", "*.bmp")
            ]
        )

        # Nếu người dùng bấm Cancel thì không làm gì thêm
        if not file_path:
            self.status_label.config(text="Đã hủy lưu ảnh.")
            return

        # Chọn định dạng dựa trên phần mở rộng mà người dùng nhập
        extension = os.path.splitext(file_path)[1].lower()

        # Nếu không có phần mở rộng thì thêm mặc định .png
        if not extension:
            file_path += ".png"
            extension = ".png"

        # Chuyển đổi ảnh hiện tại thành đúng định dạng BGR để OpenCV ghi ra file
        image_to_save = self.original_bgr.copy()

        # Nếu người dùng đang xem ảnh sau khi chuyển đổi, có thể lưu ảnh gốc thay vì ảnh đang hiển thị.
        # Theo yêu cầu, lưu bức ảnh đang hiển thị/đang chọn hiện tại, nên ở đây mặc định lưu ảnh gốc.
        # Để dễ hiểu, ta lưu ảnh gốc, vì đây là ảnh đang được chọn và thao tác chính.

        success = cv2.imwrite(file_path, image_to_save)

        if success:
            self.status_label.config(text=f"Đã lưu ảnh thành công: {file_path}")
            messagebox.showinfo("Thành công", f"Ảnh đã được lưu tại:\n{file_path}")
            print(f"Lưu thành công: {file_path}")
        else:
            self.status_label.config(text="Lưu ảnh thất bại. Vui lòng kiểm tra đường dẫn hoặc định dạng.")
            messagebox.showerror("Lỗi", "Không thể lưu ảnh. Vui lòng kiểm tra lại đường dẫn và định dạng file.")
            print(f"Lưu thất bại: {file_path}")

    def on_brightness_change(self, value):
        """Cập nhật nhãn hiển thị giá trị độ sáng khi người dùng kéo slider."""
        self.brightness_label.config(text=f"Độ sáng: {value}")

    def apply_brightness(self):
        """Thay đổi độ sáng của ảnh gốc bằng cách cộng thêm một giá trị vào từng pixel."""
        if self.original_bgr is None:
            self.status_label.config(text="Vui lòng chọn ảnh trước khi tăng độ sáng.")
            messagebox.showwarning("Cảnh báo", "Bạn cần chọn ảnh trước khi thực hiện thay đổi độ sáng.")
            return

        amount = int(self.brightness_scale.get())

        # cv2.convertScaleAbs giúp cộng/trừ sáng với bão hòa giá trị pixel về khoảng [0, 255]
        # alpha = 1.0 => giữ nguyên độ tương phản; beta = amount => tăng/giảm độ sáng
        adjusted = cv2.convertScaleAbs(self.original_bgr, alpha=1.0, beta=amount)

        # Hiển thị ảnh sau khi tăng/giảm độ sáng lên label riêng (hoặc cập nhật ảnh gốc nếu muốn)
        rgb_result = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
        self.display_image(self.labels["bitwise"], Image.fromarray(rgb_result))

        self.status_label.config(text=f"Đã thay đổi độ sáng với giá trị {amount}.")

    def rotate_image_90(self):
        """Xoay ảnh 90 độ theo chiều kim đồng hồ bằng cv2.rotate()."""
        if self.original_bgr is None:
            self.status_label.config(text="Vui lòng chọn ảnh trước khi xoay.")
            messagebox.showwarning("Cảnh báo", "Bạn cần chọn ảnh trước khi xoay ảnh.")
            return

        rotated = cv2.rotate(self.original_bgr, cv2.ROTATE_90_CLOCKWISE)
        rgb_result = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        self.display_image(self.labels["bitwise"], Image.fromarray(rgb_result))
        self.status_label.config(text="Đã xoay ảnh 90 độ theo chiều kim đồng hồ.")

    def translate_image_right(self):
        """Dịch chuyển ảnh sang phải 50 pixel bằng cv2.warpAffine()."""
        if self.original_bgr is None:
            self.status_label.config(text="Vui lòng chọn ảnh trước khi dịch chuyển.")
            messagebox.showwarning("Cảnh báo", "Bạn cần chọn ảnh trước khi dịch chuyển ảnh.")
            return

        height, width = self.original_bgr.shape[:2]
        translation_matrix = np.float32([[1, 0, 50], [0, 1, 0]])
        translated = cv2.warpAffine(self.original_bgr, translation_matrix, (width, height))

        rgb_result = cv2.cvtColor(translated, cv2.COLOR_BGR2RGB)
        self.display_image(self.labels["bitwise"], Image.fromarray(rgb_result))
        self.status_label.config(text="Đã dịch chuyển ảnh sang phải 50 pixel.")

    def scale_image(self):
        """Thu phóng ảnh lên 1.5 lần kích thước ban đầu bằng cv2.resize()."""
        if self.original_bgr is None:
            self.status_label.config(text="Vui lòng chọn ảnh trước khi thu phóng.")
            messagebox.showwarning("Cảnh báo", "Bạn cần chọn ảnh trước khi thu phóng.")
            return

        scaled = cv2.resize(self.original_bgr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        rgb_result = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        self.display_image(self.labels["bitwise"], Image.fromarray(rgb_result))
        self.status_label.config(text="Đã thu phóng ảnh lên 1.5 lần.")

    def display_image(self, label, pil_image):
        """Hiển thị PIL Image trên Label của Tkinter với tỷ lệ phù hợp để không méo hình."""
        # Giữ tỷ lệ ảnh và giảm kích thước hiển thị nếu ảnh lớn
        width, height = pil_image.size
        max_width = 350
        max_height = 300

        scale = min(max_width / width, max_height / height)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

        tk_image = ImageTk.PhotoImage(resized_image)
        label.config(image=tk_image)
        label.image = tk_image


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageConverterApp(root)
    root.mainloop()
