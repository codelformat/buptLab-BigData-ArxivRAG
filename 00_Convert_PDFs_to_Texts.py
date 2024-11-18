from deepdoc import RAGFlowPdfParser
from tqdm import tqdm

parser = RAGFlowPdfParser()

import ray
from deepdoc import RAGFlowPdfParser
from tqdm import tqdm
import os

# 初始化Ray
ray.init()

# 定义远程函数来处理单个PDF文件
@ray.remote(num_cpus=14)
def process_pdf(file_path: str) -> bool:
    try:
        parser = RAGFlowPdfParser()
        text, tables_and_figures = parser(file_path)
        output_path = file_path.replace('.pdf', '.txt')
        
        with open(output_path, "w") as f:
            cleaned_text = parser.remove_tag(text)
            f.write(cleaned_text)
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

# 主处理逻辑
def convert_pdfs_to_texts():
    # 获取所有PDF文件路径
    pdf_files = [
        f"data/Computation_and_Language/{file}" 
        for file in os.listdir("data/Computation_and_Language") 
        if file.endswith(".pdf")
    ]
    
    # 创建远程任务
    futures = [process_pdf.remote(pdf_file) for pdf_file in pdf_files]
    
    # 使用tqdm显示进度
    for _ in tqdm(ray.get(futures), total=len(futures)):
        pass

if __name__ == "__main__":
    convert_pdfs_to_texts()
    ray.shutdown()