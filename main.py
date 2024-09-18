import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# Устройство для вычислений
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use device: {device}")

# Если CUDA доступен
if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Пути к модели и конфигурации
sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

# Создание предсказателя
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Функции отображения
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# Директория с кадрами
video_dir = "./videos/colon"

# Сканируем имена файлов с изображениями
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# Инициализируем состояние предсказания
inference_state = predictor.init_state(video_path=video_dir)

# Сброс состояния для нового объекта
predictor.reset_state(inference_state)

# Индексы для работы с кадрами и объектами
ann_frame_idx = 0
ann_obj_id = 1

# Списки для хранения кликов
clicked_points = []
clicked_labels = []

# Пороговое расстояние для удаления точки
remove_threshold = 10.0

# Функция для удаления точки, находящейся в пределах `remove_threshold` от клика
def remove_point(ix, iy):
    global clicked_points, clicked_labels
    for i, (x, y) in enumerate(clicked_points):
        dist = np.sqrt((ix - x)**2 + (iy - y)**2)
        if dist < remove_threshold:
            del clicked_points[i]
            del clicked_labels[i]
            print(f"Удалена точка: ({x}, {y})")
            return True
    return False

# Функция для обработки кликов
def onclick(event):
    ix, iy = event.xdata, event.ydata
    if ix is not None and iy is not None:
        if event.button == 1:  # Левый клик
            if event.key == 'control':  # Удерживается клавиша Ctrl
                if not remove_point(ix, iy):
                    print(f"Нет точек в радиусе {remove_threshold} для удаления")
            else:
                clicked_labels.append(1)  # Положительный клик
                clicked_points.append([ix, iy])
                print(f"Левый клик в точке: ({ix}, {iy})")
        elif event.button == 3:  # Правый клик
            clicked_labels.append(0)  # Отрицательный клик
            clicked_points.append([ix, iy])
            print(f"Правый клик в точке: ({ix}, {iy})")
        update_image()  # Обновляем изображение после клика
        update_prediction()  # Обновляем предсказание маски с новыми точками

# Обновляем изображение с маркерами
def update_image():
    plt.clf()  # Очищаем текущий график
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    if clicked_points:
        points = np.array(clicked_points, dtype=np.float32)
        labels = np.array(clicked_labels, np.int32)
        show_points(points, labels, plt.gca())
    plt.draw()  # Перерисовываем график

# Обновляем предсказание маски с новыми точками
def update_prediction():
    if clicked_points:
        points = np.array(clicked_points, dtype=np.float32)
        labels = np.array(clicked_labels, np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

# Инициализация изображения и обработчик кликов
fig, ax = plt.subplots()
ax.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))

# Подключаем обработчик кликов
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()  # Отображаем окно с изображением
