import sys
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image, ImageSequence  # noqa: F401
import imageio.v2 as imageio

from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QGroupBox, QFormLayout, QSpinBox, QRadioButton,
    QButtonGroup, QSlider, QMessageBox, QComboBox, QDoubleSpinBox
)

# -------------------------
# 实用工具
# -------------------------
def np_to_qimage(arr: np.ndarray) -> QImage:
    """把 (H, W, 3|4) uint8 ndarray 转为 QImage。"""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        return qimg.copy()
    if arr.shape[2] == 3:
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
        return qimg.copy()
    elif arr.shape[2] == 4:
        qimg = QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
        return qimg.copy()
    else:
        raise ValueError("Unsupported array shape for QImage.")

def safe_message(parent, title, text, icon=QMessageBox.Information):
    m = QMessageBox(parent)
    m.setIcon(icon)
    m.setWindowTitle(title)
    m.setText(text)
    m.exec()

@dataclass
class GifInfo:
    path: str
    frame_count: int
    duration_ms_total: int
    fps: float
    per_frame_durations_ms: List[int]  # 每帧时长（毫秒）

# -------------------------
# GIF 读取与信息统计
# -------------------------
class GifAnalyzer:
    @staticmethod
    def read_info(path: str) -> GifInfo:
        im = Image.open(path)
        durations = []
        frame_count = getattr(im, "n_frames", 1)
        total = 0
        for i in range(frame_count):
            im.seek(i)
            d = im.info.get("duration", 0)  # ms
            if d <= 0:
                d = 100
            d = int(d)
            durations.append(d)
            total += d
        avg = total / max(frame_count, 1)
        fps = 1000.0 / avg if avg > 0 else 0.0
        return GifInfo(
            path=path,
            frame_count=frame_count,
            duration_ms_total=total,
            fps=round(fps, 3),
            per_frame_durations_ms=durations
        )

    @staticmethod
    def read_frames_rgba(path: str) -> List[np.ndarray]:
        """用 imageio 解码为 RGBA。"""
        reader = imageio.get_reader(path)
        frames = []
        for frame in reader:
            if frame.ndim == 2:
                frame = np.stack([frame, frame, frame, np.full_like(frame, 255)], axis=-1)
            elif frame.shape[2] == 3:
                a = np.full((frame.shape[0], frame.shape[1], 1), 255, dtype=np.uint8)
                frame = np.concatenate([frame, a], axis=-1)
            frames.append(frame)
        reader.close()
        return frames

# -------------------------
# 主窗口
# -------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GIF 工具（信息 / 抽帧 / 预览 / 序列帧导出）")
        self.resize(1060, 700)

        # 数据
        self.current_gif_info: Optional[GifInfo] = None
        self.original_frames_rgba: List[np.ndarray] = []   # 原始所有帧
        self.preview_frames_rgba: List[np.ndarray] = []    # 当前用于预览的帧（原始或抽帧结果）
        self.preview_index: int = 0
        self.last_sampling_step: int = 1                   # 最近一次“应用抽帧”的步长

        # 预览计时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer_tick)

        # UI 组件
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        # 左侧：预览
        self.preview_label = QLabel("在这里预览 GIF（加载后显示）")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("QLabel { background: #111; color: #bbb; border-radius: 10px; }")
        self.preview_label.setMinimumSize(520, 520)
        left_box = QVBoxLayout()
        left_box.addWidget(self.preview_label, 1)

        # 播放控制
        ctrl_box = QHBoxLayout()
        self.btn_play = QPushButton("播放")
        self.btn_pause = QPushButton("暂停")
        self.btn_prev = QPushButton("上一帧")
        self.btn_next = QPushButton("下一帧")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(5, 200)  # 0.05x ~ 2.00x
        self.speed_slider.setValue(100)
        self.lbl_speed = QLabel("播放速度：1.00x")

        self.btn_play.clicked.connect(self.on_play)
        self.btn_pause.clicked.connect(self.on_pause)
        self.btn_prev.clicked.connect(lambda: self.step_frame(-1))
        self.btn_next.clicked.connect(lambda: self.step_frame(1))
        self.speed_slider.valueChanged.connect(self.on_speed_change)

        ctrl_box.addWidget(self.btn_play)
        ctrl_box.addWidget(self.btn_pause)
        ctrl_box.addWidget(self.btn_prev)
        ctrl_box.addWidget(self.btn_next)
        ctrl_box.addWidget(self.lbl_speed)
        ctrl_box.addWidget(self.speed_slider, 1)
        left_box.addLayout(ctrl_box)

        # 右侧：信息 + 操作
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)

        # 文件操作
        self.btn_open = QPushButton("打开 GIF...")
        self.btn_open.clicked.connect(self.on_open)
        right_panel.addWidget(self.btn_open)

        # 信息面板
        info_group = QGroupBox("GIF 信息")
        info_form = QFormLayout(info_group)
        self.lbl_path = QLabel("-")
        self.lbl_frames = QLabel("-")
        self.lbl_fps = QLabel("-")
        self.lbl_duration = QLabel("-")
        info_form.addRow("文件：", self.lbl_path)
        info_form.addRow("帧数：", self.lbl_frames)
        info_form.addRow("FPS：", self.lbl_fps)
        info_form.addRow("时长（秒）：", self.lbl_duration)

        # —— 抽帧后信息 —— #
        self.lbl_frames_sampled = QLabel("-")
        self.lbl_fps_sampled = QLabel("-")
        self.lbl_duration_sampled = QLabel("-")
        info_form.addRow("抽帧后帧数：", self.lbl_frames_sampled)
        info_form.addRow("抽帧后FPS：", self.lbl_fps_sampled)
        info_form.addRow("抽帧后时长（秒）：", self.lbl_duration_sampled)
        right_panel.addWidget(info_group)

        # 抽帧设置 + 预览源
        sample_group = QGroupBox("抽帧设置与预览")
        sample_form = QFormLayout(sample_group)
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 9999)
        self.spin_step.setValue(1)
        self.spin_step.setToolTip("每 N 帧取 1 帧。例如 2 表示隔一帧取一帧。")
        self.btn_apply_sampling = QPushButton("应用抽帧到预览")
        self.btn_apply_sampling.clicked.connect(self.on_apply_sampling)

        self.radio_src_original = QRadioButton("原始帧")
        self.radio_src_sampled = QRadioButton("抽帧结果")
        self.radio_src_original.setChecked(True)
        self.src_group = QButtonGroup(self)
        self.src_group.addButton(self.radio_src_original, 0)
        self.src_group.addButton(self.radio_src_sampled, 1)

        # 播放帧间基准（毫秒）
        self.spin_base_interval = QSpinBox()
        self.spin_base_interval.setRange(1, 5000)  # 扩大范围以覆盖更低 FPS
        self.spin_base_interval.setValue(100)
        self.spin_base_interval.setToolTip("播放计时器的基础帧间隔（毫秒）。速度滑块会在此基础上乘系数。")
        self.spin_base_interval.valueChanged.connect(self.on_base_interval_change)

        sample_form.addRow("抽帧步长 N：", self.spin_step)
        sample_form.addRow(self.btn_apply_sampling)
        sample_form.addRow("预览源：", self._wrap_h(self.radio_src_original, self.radio_src_sampled))
        sample_form.addRow("基础帧间隔(ms)：", self.spin_base_interval)

        # —— 抽帧信息计算方式（切换开关） —— #
        calc_group = QGroupBox("抽帧信息计算方式")
        calc_lay = QHBoxLayout(calc_group)
        self.radio_calc_durations = QRadioButton("按原始帧时长")
        self.radio_calc_fixed = QRadioButton("按固定播放间隔（基础间隔×速度）")
        self.radio_calc_durations.setChecked(True)
        self.calc_group = QButtonGroup(self)
        self.calc_group.addButton(self.radio_calc_durations, 0)
        self.calc_group.addButton(self.radio_calc_fixed, 1)
        self.radio_calc_durations.toggled.connect(self.on_calc_mode_change)
        self.radio_calc_fixed.toggled.connect(self.on_calc_mode_change)
        calc_lay.addWidget(self.radio_calc_durations)
        calc_lay.addWidget(self.radio_calc_fixed)
        calc_lay.addStretch(1)

        # —— 目标 FPS 反推基础间隔（新增功能） —— #
        target_group = QGroupBox("目标 FPS → 自动设定基础间隔")
        target_form = QFormLayout(target_group)

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        # 目标 FPS 数值框
        self.spin_target_fps = QDoubleSpinBox()
        self.spin_target_fps.setDecimals(3)
        self.spin_target_fps.setRange(0.1, 240.0)
        self.spin_target_fps.setSingleStep(1.0)
        self.spin_target_fps.setValue(24.0)
        self.spin_target_fps.setToolTip("设置期望播放 FPS（会结合当前速度一起计算基础间隔）。")

        # 常用预设
        self.combo_fps_preset = QComboBox()
        self.combo_fps_preset.addItems(["— 选择预设 —", "12", "24", "30", "60"])
        self.combo_fps_preset.currentIndexChanged.connect(self.on_preset_selected)

        # 应用按钮
        self.btn_apply_target_fps = QPushButton("应用目标FPS")
        self.btn_apply_target_fps.setToolTip("根据目标 FPS 和当前速度，反推基础间隔(ms)。")
        self.btn_apply_target_fps.clicked.connect(self.on_apply_target_fps)

        row_layout.addWidget(QLabel("目标FPS："))
        row_layout.addWidget(self.spin_target_fps, 1)
        row_layout.addWidget(self.combo_fps_preset)
        row_layout.addWidget(self.btn_apply_target_fps)

        target_form.addRow(row_widget)

        right_panel.addWidget(sample_group)
        right_panel.addWidget(calc_group)
        right_panel.addWidget(target_group)

        # 导出
        export_group = QGroupBox("导出序列帧")
        export_form = QFormLayout(export_group)
        self.combo_export_src = QComboBox()
        self.combo_export_src.addItems(["原始帧序列", "抽帧后的帧序列"])
        self.combo_format = QComboBox()
        self.combo_format.addItems(["PNG", "JPEG"])
        self.btn_export = QPushButton("选择文件夹并导出...")
        self.btn_export.clicked.connect(self.on_export)

        export_form.addRow("导出来源：", self.combo_export_src)
        export_form.addRow("图片格式：", self.combo_format)
        export_form.addRow(self.btn_export)

        right_panel.addWidget(export_group)
        right_panel.addStretch(1)

        root.addLayout(left_box, 3)
        root.addLayout(right_panel, 2)

        # 菜单（可选）
        file_menu = self.menuBar().addMenu("文件")
        act_open = QAction("打开...", self)
        act_open.triggered.connect(self.on_open)
        file_menu.addAction(act_open)

        act_quit = QAction("退出", self)
        act_quit.triggered.connect(QApplication.instance().quit)
        file_menu.addAction(act_quit)

        self.update_controls_enabled(False)

    def _wrap_h(self, *widgets):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        for x in widgets:
            lay.addWidget(x)
        lay.addStretch(1)
        return w

    # -------------------------
    # 交互
    # -------------------------
    def update_controls_enabled(self, enabled: bool):
        for w in [
            self.btn_play, self.btn_pause, self.btn_prev, self.btn_next,
            self.speed_slider, self.btn_apply_sampling, self.spin_step,
            self.radio_src_original, self.radio_src_sampled,
            self.spin_base_interval, self.btn_export, self.combo_export_src,
            self.combo_format,
            self.lbl_frames_sampled, self.lbl_fps_sampled, self.lbl_duration_sampled,
            self.radio_calc_durations, self.radio_calc_fixed,
            self.spin_target_fps, self.combo_fps_preset, self.btn_apply_target_fps
        ]:
            w.setEnabled(enabled)

    # === 工具：当前速度因子 ===
    def current_speed_factor(self) -> float:
        return max(0.01, self.speed_slider.value() / 100.0)

    # === 计算抽帧后信息（两种模式） ===
    def compute_sampled_info(self, step: int):
        """
        返回 (frame_count, fps, duration_seconds)，依据当前选择的计算方式：
        - 按原始帧时长：使用被保留帧的原始 duration 计算；
        - 按固定播放间隔：fps = 1000 / (基础间隔 / 速度)，总时长 = 抽帧帧数 × 实际间隔。
        """
        if not self.current_gif_info:
            return 0, 0.0, 0.0
        step = max(1, step)

        # 抽帧后帧数（不解码也能算）
        n_frames_sampled = (self.current_gif_info.frame_count + step - 1) // step

        if self.radio_calc_fixed.isChecked():
            base_ms = max(1, self.spin_base_interval.value())
            factor = self.current_speed_factor()
            interval_ms = max(1.0, base_ms / factor)  # 实际播放间隔
            fps = 1000.0 / interval_ms
            duration_s = n_frames_sampled * interval_ms / 1000.0
            return n_frames_sampled, round(fps, 3), round(duration_s, 3)
        else:
            # 原始帧时长模式
            dur_ms = self.current_gif_info.per_frame_durations_ms[::step]
            if not dur_ms:
                return 0, 0.0, 0.0
            total_ms = sum(dur_ms)
            avg_ms = total_ms / len(dur_ms)
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
            return len(dur_ms), round(fps, 3), round(total_ms / 1000.0, 3)

    # === 刷新“抽帧后信息”三项标签 ===
    def update_sampled_labels(self):
        if not self.current_gif_info:
            return
        step = max(1, self.last_sampling_step)
        s_frames, s_fps, s_secs = self.compute_sampled_info(step)
        self.lbl_frames_sampled.setText(str(s_frames) if s_frames else "-")
        self.lbl_fps_sampled.setText(f"{s_fps:.3f}" if s_frames else "-")
        self.lbl_duration_sampled.setText(f"{s_secs:.3f}" if s_frames else "-")

    # === 目标 FPS 预设选择 ===
    def on_preset_selected(self, idx: int):
        if idx <= 0:
            return
        try:
            val = float(self.combo_fps_preset.currentText())
            self.spin_target_fps.setValue(val)
        except Exception:
            pass

    # === 应用目标 FPS（反推基础间隔） ===
    def on_apply_target_fps(self):
        target_fps = max(0.0001, float(self.spin_target_fps.value()))
        factor = self.current_speed_factor()
        # interval_ms = 1000 / target_fps
        # base_ms = interval_ms * factor
        base_ms = (1000.0 / target_fps) * factor
        base_ms_int = int(round(min(max(base_ms, 1.0), float(self.spin_base_interval.maximum()))))
        self.spin_base_interval.setValue(base_ms_int)

        # 如果正在播放，按新间隔重启计时器
        if self.timer.isActive():
            interval = int(max(1, round(self.spin_base_interval.value() / factor)))
            self.timer.start(interval)

        # 固定播放间隔模式下，刷新抽帧信息
        if self.radio_calc_fixed.isChecked():
            self.update_sampled_labels()

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 GIF 文件", "", "GIF Images (*.gif)")
        if not path:
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            info = GifAnalyzer.read_info(path)
            frames = GifAnalyzer.read_frames_rgba(path)  # list of RGBA frames

            if len(frames) != info.frame_count:
                info.frame_count = len(frames)

            self.current_gif_info = info
            self.original_frames_rgba = frames
            self.preview_frames_rgba = frames[:]  # 默认预览原始
            self.preview_index = 0
            self.last_sampling_step = 1

            # 自动设置基础帧间隔：使用平均 duration
            avg_ms = max(1, int(round(info.duration_ms_total / max(info.frame_count, 1))))
            self.spin_base_interval.setValue(avg_ms)

            # 刷新信息（原始）
            self.lbl_path.setText(os.path.basename(path))
            self.lbl_frames.setText(str(info.frame_count))
            self.lbl_fps.setText(f"{info.fps:.3f}")
            self.lbl_duration.setText(f"{info.duration_ms_total/1000.0:.3f}")

            # 清空抽帧统计
            self.lbl_frames_sampled.setText("-")
            self.lbl_fps_sampled.setText("-")
            self.lbl_duration_sampled.setText("-")

            self.update_controls_enabled(True)
            self.render_current_frame()
        except Exception as e:
            safe_message(self, "打开失败", f"读取 GIF 出错：\n{e}", QMessageBox.Critical)
        finally:
            QApplication.restoreOverrideCursor()

    def on_apply_sampling(self):
        if not self.original_frames_rgba:
            return
        step = max(1, self.spin_step.value())
        sampled = self.original_frames_rgba[::step]
        if not sampled:
            safe_message(self, "抽帧结果为空", "请调整抽帧步长。", QMessageBox.Warning)
            return

        # 切换预览源到抽帧
        self.preview_frames_rgba = sampled
        self.radio_src_sampled.setChecked(True)
        self.preview_index = 0
        self.last_sampling_step = step
        self.render_current_frame()

        # 更新“抽帧后信息”
        self.update_sampled_labels()

    def on_play(self):
        if not self.preview_frames_rgba:
            return
        base_ms = max(1, self.spin_base_interval.value())
        factor = self.current_speed_factor()  # 0.05x ~ 2.00x
        interval = int(max(1, round(base_ms / factor)))
        self.timer.start(interval)

    def on_pause(self):
        self.timer.stop()

    def on_speed_change(self, val: int):
        factor = val / 100.0
        self.lbl_speed.setText(f"播放速度：{factor:.2f}x")
        if self.timer.isActive():
            base_ms = max(1, self.spin_base_interval.value())
            interval = int(max(1, round(base_ms / max(0.01, factor))))
            self.timer.start(interval)
        # 若当前选择“固定播放间隔”模式，动态更新抽帧信息
        if self.radio_calc_fixed.isChecked():
            self.update_sampled_labels()

    def on_base_interval_change(self, _):
        # 基础间隔变化：如果是固定播放间隔模式，更新抽帧信息
        if self.radio_calc_fixed.isChecked():
            self.update_sampled_labels()
        # 同时如果正在播放，按新间隔重启计时器
        if self.timer.isActive():
            base_ms = max(1, self.spin_base_interval.value())
            factor = self.current_speed_factor()
            interval = int(max(1, round(base_ms / factor)))
            self.timer.start(interval)

    def on_calc_mode_change(self):
        # 切换“抽帧信息计算方式”时刷新显示
        if self.current_gif_info:
            self.update_sampled_labels()

    def on_timer_tick(self):
        self.step_frame(1)

    def step_frame(self, delta: int):
        src = 0 if self.radio_src_original.isChecked() else 1
        frames = self.original_frames_rgba if src == 0 else self.preview_frames_rgba
        if not frames:
            return
        self.preview_index = (self.preview_index + delta) % len(frames)
        self.render_current_frame()

    def render_current_frame(self):
        src = 0 if self.radio_src_original.isChecked() else 1
        frames = self.original_frames_rgba if src == 0 else self.preview_frames_rgba
        if not frames:
            self.preview_label.setText("没有可预览的帧")
            return
        idx = max(0, min(self.preview_index, len(frames) - 1))
        frame = frames[idx]
        qimg = np_to_qimage(frame[:, :, :3] if frame.shape[2] == 3 else frame)
        pix = QPixmap.fromImage(qimg)
        target = self.preview_label.size() - QSize(20, 20)
        if target.width() > 0 and target.height() > 0:
            pix = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(pix)
        self.preview_label.setToolTip(f"帧 {idx+1}/{len(frames)}")

    def on_export(self):
        if not self.original_frames_rgba:
            safe_message(self, "未加载", "请先打开一个 GIF。", QMessageBox.Warning)
            return

        export_src = self.combo_export_src.currentIndex()  # 0: 原始 1: 抽帧
        if export_src == 0:
            frames = self.original_frames_rgba
        else:
            if self.radio_src_sampled.isChecked():
                frames = self.preview_frames_rgba
            else:
                step = max(1, self.spin_step.value())
                frames = self.original_frames_rgba[::step]
            if not frames:
                safe_message(self, "抽帧结果为空", "请调整抽帧步长。", QMessageBox.Warning)
                return

        out_dir = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
        if not out_dir:
            return

        fmt = self.combo_format.currentText().lower()  # png / jpeg
        zero_pad = len(str(len(frames)))
        base = os.path.splitext(os.path.basename(self.current_gif_info.path))[0] if self.current_gif_info else "gif"

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            for i, arr in enumerate(frames):
                if arr.shape[2] == 4:
                    pil_img = Image.fromarray(arr, mode="RGBA")
                    if fmt == "jpeg":
                        # JPEG 不支持透明通道，转白底
                        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
                        bg.paste(pil_img, mask=pil_img.split()[-1])
                        pil_img = bg
                        save_mode = "JPEG"
                    else:
                        save_mode = "PNG"
                else:
                    pil_img = Image.fromarray(arr[:, :, :3], mode="RGB")
                    save_mode = "JPEG" if fmt == "jpeg" else "PNG"

                fname = f"{base}_{str(i+1).zfill(zero_pad)}.{fmt}"
                pil_img.save(os.path.join(out_dir, fname), save_mode)
            safe_message(self, "导出完成", f"共导出 {len(frames)} 张。", QMessageBox.Information)
        except Exception as e:
            safe_message(self, "导出失败", f"保存出错：\n{e}", QMessageBox.Critical)
        finally:
            QApplication.restoreOverrideCursor()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
