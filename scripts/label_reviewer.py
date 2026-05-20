"""
scripts/label_reviewer.py

Інтерактивний переглядач і редактор YOLO-лейблів.

Запуск з кореня проєкту:
    python scripts/label_reviewer.py
    python scripts/label_reviewer.py --data data/interim/leaves_sam

Клавіші:
    ←  /  →         попередня / наступна (з автозбереженням)
    S               зберегти вручну
    D / Backspace   видалити картинку + лейбл назавжди
    N               режим малювання нових bbox (toggle)
    Escape          скасувати малювання / вийти з draw-mode
    H               сховати / показати всі bbox
    Z               відмінити останній доданий bbox
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    print("tkinter не знайдено.")
    sys.exit(1)

try:
    from PIL import Image, ImageTk, ImageDraw
except ImportError:
    print("pip install Pillow")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Кольори
# ---------------------------------------------------------------------------
PALETTE = [
    (74, 222, 128), (96, 165, 250), (244, 114, 182), (251, 146, 60),
    (167, 139, 250), (52, 211, 153), (251, 191, 36), (56, 189, 248),
]
DELETED_COLOR = (248, 113, 113)
IMAGE_EXTS    = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# ---------------------------------------------------------------------------
# Дані
# ---------------------------------------------------------------------------

class BBox:
    __slots__ = ('cls', 'cx', 'cy', 'w', 'h', 'deleted')

    def __init__(self, cls: int, cx: float, cy: float, w: float, h: float):
        self.cls     = cls
        self.cx      = cx
        self.cy      = cy
        self.w       = w
        self.h       = h
        self.deleted = False

    def to_line(self) -> str:
        return f"{self.cls} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


class LabeledImage:
    def __init__(self, image_path: Path, label_path: Path, tree: str):
        self.image_path = image_path
        self.label_path = label_path
        self.tree       = tree
        self.boxes: list[BBox] = self._load()

    def _load(self) -> list[BBox]:
        if not self.label_path.exists():
            return []
        boxes = []
        for line in self.label_path.read_text(encoding='utf-8').splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append(BBox(
                    int(parts[0]), float(parts[1]),
                    float(parts[2]), float(parts[3]), float(parts[4])))
        return boxes

    def save(self) -> None:
        kept = [b for b in self.boxes if not b.deleted]
        text = '\n'.join(b.to_line() for b in kept)
        self.label_path.write_text(
            text + '\n' if text else '', encoding='utf-8')

    def delete_files(self) -> None:
        self.image_path.unlink(missing_ok=True)
        self.label_path.unlink(missing_ok=True)


def collect_items(data_root: Path) -> list[LabeledImage]:
    images_root = data_root / 'images'
    labels_root = data_root / 'labels'
    result: list[LabeledImage] = []
    if not images_root.exists():
        return result
    for tree_dir in sorted(images_root.iterdir()):
        if not tree_dir.is_dir():
            continue
        tree = tree_dir.name
        for img in sorted(tree_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = labels_root / tree / f'{img.stem}.txt'
            if not lbl.exists():
                continue
            result.append(LabeledImage(img, lbl, tree))
    return result


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class App:
    BG        = '#0f0f0f'
    BG2       = '#1a1a1a'
    BG3       = '#242424'
    FG        = '#e8e8e8'
    FG_DIM    = '#555555'
    GREEN     = '#4ade80'
    RED       = '#f87171'
    YELLOW    = '#fbbf24'
    SIDEBAR_W = 215

    def __init__(self, root: tk.Tk, data_root: Path):
        self.root      = root
        self.data_root = data_root
        self.items:    list[LabeledImage] = []
        self.idx       = 0
        self.photo     = None
        self.hovered   = -1
        self.hide_boxes = False

        # display geometry
        self._scale = 1.0
        self._ox = 0
        self._oy = 0
        self._dw = 1
        self._dh = 1

        # draw mode
        self.draw_mode   = False
        self._drag_start = None   # (canvas_x, canvas_y)
        self._rubber_id  = None   # ID rubber-band прямокутника на canvas

        self._build_ui()
        self._load_all()

    # ── UI ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title('Label Reviewer')
        self.root.configure(bg=self.BG)
        self.root.geometry('1300x860')
        self.root.minsize(800, 560)
        self._bind_keys()

        # TOP
        top = tk.Frame(self.root, bg=self.BG2, height=46)
        top.pack(fill='x', side='top')
        top.pack_propagate(False)

        self.lbl_name = tk.Label(
            top, text='', bg=self.BG2, fg=self.FG,
            font=('Menlo', 13, 'bold'), anchor='w')
        self.lbl_name.pack(side='left', padx=14, pady=10)

        self.lbl_tree = tk.Label(
            top, text='', bg=self.BG2, fg=self.GREEN,
            font=('Menlo', 12))
        self.lbl_tree.pack(side='left')

        self.lbl_progress = tk.Label(
            top, text='', bg=self.BG2, fg=self.FG_DIM,
            font=('Menlo', 11))
        self.lbl_progress.pack(side='right', padx=14)

        # MAIN
        main = tk.Frame(self.root, bg=self.BG)
        main.pack(fill='both', expand=True)

        # Права панель
        right = tk.Frame(main, bg=self.BG2, width=self.SIDEBAR_W)
        right.pack(side='right', fill='y')
        right.pack_propagate(False)

        tk.Label(right, text='BBOX', bg=self.BG2, fg=self.FG_DIM,
                 font=('Menlo', 10, 'bold')).pack(pady=(12, 4), padx=10, anchor='w')

        self.bbox_frame = tk.Frame(right, bg=self.BG2)
        self.bbox_frame.pack(fill='both', expand=True, padx=6, pady=4)

        # Canvas
        self.canvas = tk.Canvas(
            main, bg='#080808', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<ButtonPress-1>',   self._mouse_down)
        self.canvas.bind('<B1-Motion>',       self._mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._mouse_up)
        self.canvas.bind('<Motion>',          self._motion)
        self.canvas.bind('<Leave>',           self._leave)
        self.canvas.bind('<Configure>',       lambda _: self._redraw())

        # BOTTOM
        bot = tk.Frame(self.root, bg=self.BG2, height=52)
        bot.pack(fill='x', side='bottom')
        bot.pack_propagate(False)

        def btn(parent, text, cmd, bg, fg):
            return tk.Button(
                parent, text=text, command=cmd,
                bg=bg, fg=fg, activebackground=bg, activeforeground=fg,
                font=('Menlo', 12), relief='flat',
                cursor='hand2', padx=12, pady=6, bd=0)

        btn(bot, '← Назад',           self.prev,             self.BG3,    self.FG     ).pack(side='left', padx=(10,3), pady=8)
        btn(bot, 'Зберегти  [S]',     self.save,             '#1a3d28',   self.GREEN  ).pack(side='left', padx=3,     pady=8)
        btn(bot, 'Далі →',            self.next,             self.BG3,    self.FG     ).pack(side='left', padx=3,     pady=8)
        btn(bot, 'Сховати  [H]',      self.toggle_hide,      self.BG3,    self.FG_DIM ).pack(side='left', padx=10,    pady=8)
        btn(bot, '↩  [Z]',            self.undo_last,        self.BG3,    self.FG_DIM ).pack(side='left', padx=0,     pady=8)

        self.btn_draw = btn(
            bot, '✏  Малювати  [N]',  self.toggle_draw_mode, self.BG3,    self.FG_DIM)
        self.btn_draw.pack(side='left', padx=10, pady=8)

        btn(bot, '🗑  Видалити  [D]',  self.delete,           '#3d1a1a',   self.RED    ).pack(side='right', padx=10, pady=8)

        self.lbl_status = tk.Label(
            bot, text='', bg=self.BG2, fg=self.GREEN, font=('Menlo', 11))
        self.lbl_status.pack(side='right', padx=6)

    def _bind_keys(self):
        self.root.bind('<Left>',     lambda _: self.prev())
        self.root.bind('<Right>',    lambda _: self.next())
        self.root.bind('s',          lambda _: self.save())
        self.root.bind('S',          lambda _: self.save())
        self.root.bind('n',          lambda _: self.toggle_draw_mode())
        self.root.bind('N',          lambda _: self.toggle_draw_mode())
        self.root.bind('h',          lambda _: self.toggle_hide())
        self.root.bind('H',          lambda _: self.toggle_hide())
        self.root.bind('z',          lambda _: self.undo_last())
        self.root.bind('Z',          lambda _: self.undo_last())
        self.root.bind('<BackSpace>', lambda _: self.delete())
        self.root.bind('<Delete>',   lambda _: self.delete())
        self.root.bind('<Escape>',   lambda _: self._escape())

    # ── Дані ───────────────────────────────────────────────────────────

    def _load_all(self):
        self.items = collect_items(self.data_root)
        if not self.items:
            messagebox.showerror(
                'Помилка',
                f'Не знайдено зображень з лейблами у:\n{self.data_root}')
            self.root.quit()
            return
        self.idx = 0
        self._show()

    def _item(self) -> LabeledImage | None:
        return self.items[self.idx] if self.items else None

    # ── Відображення ───────────────────────────────────────────────────

    def _show(self):
        item = self._item()
        if item is None:
            return
        self.lbl_name.config(text=item.image_path.name[:55])
        self.lbl_tree.config(text=f'  [{item.tree}]')
        self.lbl_status.config(text='')
        self.hovered = -1
        self._update_progress()
        self._rebuild_sidebar()
        self._redraw()

    def _update_progress(self):
        item = self._item()
        if item is None:
            return
        kept  = sum(1 for b in item.boxes if not b.deleted)
        total = len(item.boxes)
        self.lbl_progress.config(
            text=f'{self.idx + 1} / {len(self.items)}   boxes: {kept}/{total}')

    def _rebuild_sidebar(self):
        for w in self.bbox_frame.winfo_children():
            w.destroy()
        item = self._item()
        if item is None:
            return

        for i, box in enumerate(item.boxes):
            color   = PALETTE[i % len(PALETTE)]
            hex_col = '#{:02x}{:02x}{:02x}'.format(*color)
            is_del  = box.deleted
            row_bg  = '#2d1a1a' if is_del else self.BG2

            row = tk.Frame(self.bbox_frame, bg=row_bg, pady=3)
            row.pack(fill='x', pady=2)

            num_col = '#{:02x}{:02x}{:02x}'.format(
                *DELETED_COLOR) if is_del else hex_col
            tk.Label(row, text=f'{i}', bg=num_col, fg='#0a0a0a',
                     font=('Menlo', 10, 'bold'), width=3
                     ).pack(side='left', padx=(6, 4))

            pct = f'{box.w*100:.0f}×{box.h*100:.0f}%'
            tk.Label(row, text=pct, bg=row_bg,
                     fg=self.FG_DIM if is_del else self.FG,
                     font=('Menlo', 10)).pack(side='left')

            sym    = '✕' if is_del else '✓'
            sym_fg = self.RED if is_del else self.GREEN
            tk.Button(
                row, text=sym, bg=row_bg, fg=sym_fg,
                font=('Menlo', 11, 'bold'), relief='flat',
                cursor='hand2', bd=0,
                command=lambda ix=i: self._toggle(ix)
            ).pack(side='right', padx=6)

            for w in [row] + list(row.winfo_children()):
                try:
                    w.bind('<Button-1>', lambda _, ix=i: self._toggle(ix))
                except Exception:
                    pass

    def _redraw(self):
        item = self._item()
        if item is None:
            self.canvas.delete('all')
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        try:
            img = Image.open(item.image_path).convert('RGB')
        except Exception as e:
            self.canvas.delete('all')
            self.canvas.create_text(
                cw//2, ch//2, text=str(e), fill=self.RED, font=('Menlo', 12))
            return

        iw, ih = img.size
        self._scale = min(cw / iw, ch / ih, 1.0)
        self._dw = max(1, int(iw * self._scale))
        self._dh = max(1, int(ih * self._scale))
        self._ox = (cw - self._dw) // 2
        self._oy = (ch - self._dh) // 2

        disp = img.resize((self._dw, self._dh), Image.LANCZOS)

        if not self.hide_boxes:
            draw = ImageDraw.Draw(disp, 'RGBA')
            for i, box in enumerate(item.boxes):
                if i != self.hovered:
                    self._draw_box(draw, box, i, hover=False)
            if 0 <= self.hovered < len(item.boxes):
                self._draw_box(draw, item.boxes[self.hovered],
                               self.hovered, hover=True)

        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.delete('all')
        self.canvas.create_image(self._ox, self._oy, anchor='nw', image=self.photo)

        # жовта рамка canvas у draw-mode
        if self.draw_mode:
            self.canvas.create_rectangle(
                2, 2, cw - 2, ch - 2,
                outline=self.YELLOW, width=2, dash=(6, 4))

    def _draw_box(self, draw: ImageDraw.ImageDraw,
                  box: BBox, idx: int, hover: bool):
        color      = DELETED_COLOR if box.deleted else PALETTE[idx % len(PALETTE)]
        fill_a     = 30 if box.deleted else (70 if hover else 40)
        line_a     = 120 if box.deleted else (230 if hover else 190)
        lw         = 3 if hover else 2

        x1 = max(0, int((box.cx - box.w/2) * self._dw))
        y1 = max(0, int((box.cy - box.h/2) * self._dh))
        x2 = min(self._dw, int((box.cx + box.w/2) * self._dw))
        y2 = min(self._dh, int((box.cy + box.h/2) * self._dh))

        draw.rectangle([x1, y1, x2, y2], fill=(*color, fill_a))
        for t in range(lw):
            draw.rectangle([x1+t, y1+t, x2-t, y2-t],
                           outline=(*color, line_a))

        label = f'#{idx}'
        bw    = len(label) * 8 + 8
        by    = max(0, y1 - 17)
        draw.rectangle([x1, by, x1 + bw, by + 16],
                       fill=(*color, 200 if not box.deleted else 140))
        draw.text((x1 + 4, by + 2), label, fill=(10, 10, 10))

    # ── Canvas: координати ─────────────────────────────────────────────

    def _canvas_to_norm(self, cx: int, cy: int) -> tuple[float, float] | None:
        """Canvas px → нормалізовані YOLO координати."""
        ix = cx - self._ox
        iy = cy - self._oy
        if ix < 0 or iy < 0 or ix > self._dw or iy > self._dh:
            return None
        return ix / self._dw, iy / self._dh

    def _bbox_px(self, box: BBox) -> tuple[int, int, int, int]:
        x1 = int((box.cx - box.w/2) * self._dw) + self._ox
        y1 = int((box.cy - box.h/2) * self._dh) + self._oy
        x2 = int((box.cx + box.w/2) * self._dw) + self._ox
        y2 = int((box.cy + box.h/2) * self._dh) + self._oy
        return x1, y1, x2, y2

    def _hit(self, mx: int, my: int) -> int:
        item = self._item()
        if item is None:
            return -1
        best, best_area = -1, float('inf')
        for i, box in enumerate(item.boxes):
            x1, y1, x2, y2 = self._bbox_px(box)
            if x1 <= mx <= x2 and y1 <= my <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best, best_area = i, area
        return best

    # ── Mouse events ───────────────────────────────────────────────────

    def _mouse_down(self, event: tk.Event):
        if self.draw_mode:
            # починаємо малювати rubber-band
            self._drag_start = (event.x, event.y)
            self._rubber_id  = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline=self.YELLOW, width=2, dash=(5, 3))
        else:
            idx = self._hit(event.x, event.y)
            if idx >= 0:
                self._toggle(idx)

    def _mouse_drag(self, event: tk.Event):
        if self.draw_mode and self._drag_start and self._rubber_id:
            x0, y0 = self._drag_start
            self.canvas.coords(self._rubber_id, x0, y0, event.x, event.y)

    def _mouse_up(self, event: tk.Event):
        if not (self.draw_mode and self._drag_start):
            return

        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y

        # прибираємо rubber band
        if self._rubber_id:
            self.canvas.delete(self._rubber_id)
            self._rubber_id = None
        self._drag_start = None

        # мінімальний розмір: 10px в обох напрямках
        if abs(x1 - x0) < 10 or abs(y1 - y0) < 10:
            return

        # конвертуємо обидва кути в нормалізовані координати
        pt0 = self._canvas_to_norm(min(x0, x1), min(y0, y1))
        pt1 = self._canvas_to_norm(max(x0, x1), max(y0, y1))
        if pt0 is None or pt1 is None:
            return

        nx1, ny1 = pt0
        nx2, ny2 = pt1
        cx = (nx1 + nx2) / 2
        cy = (ny1 + ny2) / 2
        w  = nx2 - nx1
        h  = ny2 - ny1

        if w < 0.005 or h < 0.005:
            return

        item = self._item()
        if item is None:
            return

        item.boxes.append(BBox(cls=0, cx=cx, cy=cy, w=w, h=h))

        idx = len(item.boxes) - 1
        self._set_status(f'+ bbox #{idx} додано', color=self.YELLOW)
        self._update_progress()
        self._rebuild_sidebar()
        self._redraw()

    def _motion(self, event: tk.Event):
        if self.draw_mode:
            return   # у draw-mode hover не потрібен
        idx = self._hit(event.x, event.y)
        if idx != self.hovered:
            self.hovered = idx
            self._redraw()
            self.canvas.config(cursor='hand2' if idx >= 0 else 'crosshair')

    def _leave(self, _):
        self.hovered = -1
        self._redraw()

    # ── Toggle / Undo ──────────────────────────────────────────────────

    def _toggle(self, idx: int):
        item = self._item()
        if item and 0 <= idx < len(item.boxes):
            item.boxes[idx].deleted = not item.boxes[idx].deleted
            self._rebuild_sidebar()
            self._update_progress()
            self._redraw()

    def undo_last(self):
        """Видалити останній не-deleted bbox зі списку."""
        item = self._item()
        if not item or not item.boxes:
            return
        for i in range(len(item.boxes) - 1, -1, -1):
            if not item.boxes[i].deleted:
                item.boxes.pop(i)
                self._set_status('↩ відмінено', color=self.FG_DIM)
                self._update_progress()
                self._rebuild_sidebar()
                self._redraw()
                return

    # ── Draw mode ──────────────────────────────────────────────────────

    def toggle_draw_mode(self):
        self.draw_mode = not self.draw_mode
        if self.draw_mode:
            self.btn_draw.config(
                bg='#3d3500', fg=self.YELLOW,
                text='✏  Малювати  [N]  ●')
            self._set_status('Draw mode — клікни і тягни', color=self.YELLOW)
        else:
            self.btn_draw.config(
                bg=self.BG3, fg=self.FG_DIM,
                text='✏  Малювати  [N]')
            self.lbl_status.config(text='')
        self.canvas.config(cursor='crosshair')
        self._redraw()

    def _escape(self):
        # скасувати поточне drag
        if self._rubber_id:
            self.canvas.delete(self._rubber_id)
            self._rubber_id  = None
            self._drag_start = None
        # вийти з draw mode
        if self.draw_mode:
            self.toggle_draw_mode()

    # ── Дії ────────────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str = '#4ade80'):
        self.lbl_status.config(text=text, fg=color)
        self.root.after(2500, lambda: self.lbl_status.config(text=''))

    def save(self, silent: bool = False):
        item = self._item()
        if item is None:
            return
        item.save()
        if not silent:
            self._set_status('✓ збережено')

    def next(self):
        if not self.items:
            return
        self.save(silent=True)
        self.idx = (self.idx + 1) % len(self.items)
        self._show()

    def prev(self):
        if not self.items:
            return
        self.save(silent=True)
        self.idx = (self.idx - 1) % len(self.items)
        self._show()

    def delete(self):
        item = self._item()
        if item is None:
            return
        if not messagebox.askyesno(
                'Видалити?',
                f'Видалити назавжди:\n'
                f'• {item.image_path.name}\n'
                f'• {item.label_path.name}\n\n'
                f'Це незворотно.'):
            return
        item.delete_files()
        self.items.pop(self.idx)
        if not self.items:
            self.canvas.delete('all')
            self.lbl_name.config(text='Зображень більше немає')
            self.lbl_tree.config(text='')
            self.lbl_progress.config(text='')
            for w in self.bbox_frame.winfo_children():
                w.destroy()
            return
        if self.idx >= len(self.items):
            self.idx = len(self.items) - 1
        self._show()

    def toggle_hide(self):
        self.hide_boxes = not self.hide_boxes
        self._redraw()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Інтерактивний переглядач і редактор YOLO-лейблів')
    parser.add_argument(
        '--data', type=Path,
        default=Path('data/interim/leaves_sam'),
        help='Папка з images/ та labels/ (default: data/interim/leaves_sam)')
    args = parser.parse_args()

    root = tk.Tk()
    App(root, args.data)
    root.mainloop()


if __name__ == '__main__':
    main()