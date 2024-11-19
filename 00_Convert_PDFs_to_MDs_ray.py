import ray
from marker.convert import convert_single_pdf
from marker.models import load_all_models
import os
import ujson as json
from hdfs import InsecureClient
from io import BytesIO
from PIL import Image

# 初始化Ray
ray.init()

# 初始化HDFS客户端
hdfs_client = InsecureClient('http://localhost:9900')  

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

@ray.remote
def process_pdf(pdf_path, model_lst):
    """Ray远程函数：处理单个PDF文件"""
    try:
        # 从HDFS读取PDF文件
        with hdfs_client.read(f'pdfs/{pdf_path}') as reader:
            pdf_content = reader.read()
        
        # 转换PDF
        full_text, images, out_meta = convert_single_pdf(
            BytesIO(pdf_content), 
            model_lst,
            ocr_all_pages=True,
            batch_multiplier=4
        )
        
        # 获取基础文件名
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = f"output/{base_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存文本
        with open(f"{output_dir}/{base_name}.md", "w", encoding='utf-8') as f:
            f.write(full_text)
        
        # 保存图片
        for key, img in images.items():
            image_filename = f"{key}.png"
            image_path = os.path.join(output_dir, image_filename)
            img.save(image_path)
            
        # 保存元数据
        with open(f"{output_dir}/meta.json", "w", encoding='utf-8') as f:
            json.dump(out_meta, f)
            
        return True
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return False

def main():
    # 加载模型
    model_lst = load_all_models()
    
    # 获取HDFS中的PDF文件列表
    pdf_files = [f for f in hdfs_client.list('pdfs') if f.lower().endswith('.pdf')]
    
    # 创建Ray任务
    tasks = [process_pdf.remote(pdf_file, model_lst) for pdf_file in pdf_files]
    
    # 等待所有任务完成
    results = ray.get(tasks)
    
    # 输出处理结果统计
    success_count = sum(1 for r in results if r)
    print(f"Successfully processed {success_count} out of {len(results)} files")

if __name__ == "__main__":
    main()
    ray.shutdown()