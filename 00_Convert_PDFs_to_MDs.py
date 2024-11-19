from marker.convert import convert_single_pdf
from marker.models import load_all_models
import os
import ujson as json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_lst = load_all_models()
base_dir = "../arxiv_pdfs"

# 获取第一级子目录
sub_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# 遍历每个子目录
for sub_dir in sub_dirs:
    sub_dir_path = os.path.join(base_dir, sub_dir)
    
    # 遍历子目录中的所有文件
    for file_name in os.listdir(sub_dir_path):
        if not file_name.lower().endswith('.pdf'):
            continue
            
        pdf_path = os.path.join(sub_dir_path, file_name)
        # 生成输出文件的基础名称（不包含扩展名）
        base_name = os.path.splitext(file_name)[0]
        # 创建输出目录
        output_dir = f"output/{base_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Convert pdf file
        full_text, images, out_meta = convert_single_pdf(pdf_path, model_lst,ocr_all_pages=True,batch_multiplier=4)
        
        # Write to markdown file
        with open(f"{output_dir}/{base_name}.md", "w", encoding='utf-8') as f:
            f.write(full_text)
            
        # Save images and their metadata
        images_dir = f"{output_dir}"
        os.makedirs(images_dir, exist_ok=True)
        
        # Create a list to store image metadata
        image_metadata = []
        
        # Save each image and collect metadata
        for key, img in images.items():
            image_filename = f"{key}.png"
            image_path = os.path.join(images_dir, image_filename)
            img.save(image_path)
            
        # Save metadata
        with open(f"{output_dir}/meta.json", "w", encoding='utf-8') as f:
            json.dump(out_meta, f)
