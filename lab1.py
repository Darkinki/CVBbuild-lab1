import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
import yaml
from ultralytics import YOLO
import pytesseract

def generate_markdown_from_document(image_path, model_path):
    print(f"\nАналіз документа: {image_path}")
    
    model = YOLO(model_path)
    results = model(image_path)
    
    classes = [
        'Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 
        'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'
    ]
    
    markdown_content = ""
    
    for r in results:
        img = r.orig_img
        boxes = r.boxes
        
        sorted_boxes = sorted(boxes, key=lambda b: b.xyxy[0][1])
        
        for box in sorted_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = classes[cls_id]
            
            cropped_segment = img[y1:y2, x1:x2]
            
            if cropped_segment.shape[0] == 0 or cropped_segment.shape[1] == 0:
                continue
            
            if label in ['Text', 'Title', 'Section-header', 'List-item', 'Caption']:
                # 3. Викликаємо OCR без параметра config
                text = pytesseract.image_to_string(cropped_segment, lang='eng').strip()
                
                if label in ['Title', 'Section-header']:
                    markdown_content += f"\n## {text}\n"
                elif label == 'List-item':
                    markdown_content += f"- {text}\n"
                else:
                    markdown_content += f"\n{text}\n"
                    
            elif label == 'Table':
                markdown_content += f"\n> **[Розпізнано таблицю, впевненість={conf:.2f}]**\n"
            elif label == 'Picture':
                markdown_content += f"\n> **[Розпізнано зображення, впевненість={conf:.2f}]**\n"
                
    return markdown_content


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    print("Завантаження датасету")
    dataset = load_dataset("pierreguillou/DocLayNet-small", split="train[:50]", trust_remote_code=True)
    print(f"Завантажено {len(dataset)} зображень документів.")

    base_dir = "yolo_dataset"
    os.makedirs(os.path.join(base_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "train"), exist_ok=True)

    print("\nСтарт конвертації")
    for idx, sample in enumerate(tqdm(dataset, desc="Конвертація")):
        image = np.array(sample['image'])
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        img_height, img_width = image.shape[:2]
        file_name = f"doc_{idx:04d}"
        img_path = os.path.join(base_dir, "images", "train", f"{file_name}.jpg")
        label_path = os.path.join(base_dir, "labels", "train", f"{file_name}.txt")
        
        cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        bboxes = sample['bboxes_block']
        categories = sample['categories']
        yolo_annotations = []
        
        for bbox, cat_id in zip(bboxes, categories):
            x, y, w, h = bbox
            x_center = x + (w / 2.0)
            y_center = y + (h / 2.0)
            
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = w / img_width
            height_norm = h / img_height
            
            yolo_class_id = max(0, cat_id - 1)
            yolo_annotations.append(f"{yolo_class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
            
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_annotations))

    doclaynet_classes = [
        'Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 
        'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'
    ]
    yaml_content = {
        'path': os.path.abspath('yolo_dataset'),
        'train': 'images/train',
        'val': 'images/train',
        'nc': len(doclaynet_classes),
        'names': doclaynet_classes
    }
    yaml_path = os.path.abspath(os.path.join('yolo_dataset', 'dataset.yaml'))
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print("\nЗапуск навчання")
    model = YOLO('yolov8n.pt') 

    results = model.train(
        data=yaml_path,
        epochs=30,          
        imgsz=640,
        batch=4,
        project='lab_results',
        name='baseline_experiment',
        device='0',       
        workers=4           
    )
    print("\nТренування завершено")

    print("\nЗапуск пайплайну розпізнавання OCR")
    test_img = os.path.join('yolo_dataset', 'images', 'train', 'doc_0000.jpg')
    
    trained_model_weights = r'runs\detect\lab_results\baseline_experiment\weights\best.pt'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'


    if os.path.exists(test_img) and os.path.exists(trained_model_weights):
        md_result = generate_markdown_from_document(test_img, trained_model_weights)
        
        output_file = 'output_document.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_result)
            
        print("\nЗгенерований Markdown")
        print(md_result)
        print(f"\nРезультат збережено в {output_file}")
    else:
        print(f"Помилка: Тестове зображення або навчена модель ({trained_model_weights}) не знайдені.")