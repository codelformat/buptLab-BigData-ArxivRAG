import arxiv
import requests
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
from marker.convert import convert_single_pdf
from marker.models import load_all_models
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from ltp import StnSplit
import time
from openai import OpenAI
import json
import os

# os.environ["OPENAI_API_KEY"] = "REDACTED"

class QueryProcessor:
    def __init__(self):
        self.client = OpenAI(
            api_key="REDACTED",
            base_url="https://api.minimax.chat/v1"
        )
        
    def rephrase_query(self, query):
        """将用户查询转换为适合arxiv的关键词"""
        response = self.client.chat.completions.create(
            model="abab7-chat-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a key information extractor and always respond with only one most relevant keyword or key phrase from the input for use in a search engine query. Focus on essential terms and maintain the original wording."
                },
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content.strip()
    
    def process_rag_results(self, query, chunks, metadata=None):
        """处理RAG结果并生成回答"""
        # 构建上下文
        context = []
        for i, chunk in enumerate(chunks):
            source = f"Source {i+1}"
            if metadata and i < len(metadata):
                source = metadata[i]
            context.append(f"{source}: {chunk}")
        
        context_str = "\n\n".join(context)
        
        # 生成回答
        response = self.client.chat.completions.create(
            model="abab7-chat-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an academic assistant. Please answer the question based on the provided literature fragments. When answering, please cite the source of the information."
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nContext:\n{context_str}"
                }
            ]
        )
        return response.choices[0].message.content
    
    def generate_followup_questions(self, query, answer):
        """生成后续问题"""
        response = self.client.chat.completions.create(
            model="abab7-chat-preview",
            messages=[
                {
                    "role": "system",
                    "content": "Based on the original query and answer, generate 3 insightful follow-up questions. Return in JSON format. Format: {\"follow_up\": [\"Question 1\", \"Question 2\", \"Question 3\"]}"
                },
                {
                    "role": "user",
                    "content": f"Original query: {query}\nAnswer: {answer}"
                }
            ]
        )
        print("后续的问题原文本：")
        # 如果被```json```包裹，则去掉
        content = response.choices[0].message.content
        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3]
            print(content)
        return json.loads(content)

class SemanticChunkProcessor:
    def __init__(self, threshold=70):
        self.threshold = threshold
        self.stn_splitter = StnSplit()
    
    def split_text(self, text):
        # 简化版的语义分块，使用句子分割
        sentences = self.stn_splitter.split(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            current_chunk.append(sentence)
            current_length += len(sentence)
            
            if current_length >= self.threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

class PaperProcessor:
    def __init__(self):
        # 设置各种目录
        self.download_dir = Path('papers')
        self.markdown_dir = Path('papers_markdown')
        self.vector_dir = Path('papers_vector')
        
        for dir_path in [self.download_dir, self.markdown_dir, self.vector_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 初始化模型
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        self.model_lst = load_all_models()
        
        # 初始化embedding模型
        model_name = "/home/codelformat/shared_models/bge-m3"
        self.embedding_function = HuggingFaceBgeEmbeddings(
            model_name=model_name, 
            model_kwargs={"device": "cuda"}
        )
        
        # 初始化语义分块处理器
        self.chunk_processor = SemanticChunkProcessor()
        
        # 创建任务队列
        self.conversion_queue = Queue()
        self.vector_queue = Queue()
        
        # 启动工作线程
        self.conversion_thread = Thread(target=self._conversion_worker, daemon=True)
        self.vector_thread = Thread(target=self._vector_worker, daemon=True)
        self.conversion_thread.start()
        self.vector_thread.start()
        
        # 添加查询处理器
        self.query_processor = QueryProcessor()
        
        # 保存当前的向量存储
        self.current_vectorstore = None

    def download_paper(self, args):
        i, result = args
        title = result.title
        pdf_url = result.pdf_url
        
        # 清理文件名
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{i}_{safe_title[:100]}"
        pdf_path = self.download_dir / f"{filename}.pdf"
        
        print(f"\n下载第 {i} 篇论文:")
        print(f"标题: {title}")
        
        try:
            # 下载PDF文件
            response = requests.get(pdf_url)
            response.raise_for_status()
            
            # 保存PDF文件
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            print(f"成功下载到: {pdf_path}")
            
            # 将转换任务加入队列
            self.conversion_queue.put((pdf_path, filename))
            
            return True
        except Exception as e:
            print(f"下载失败: {str(e)}")
            return False

    def _conversion_worker(self):
        while True:
            try:
                # 从队列获取转换任务
                pdf_path, filename = self.conversion_queue.get()
                if pdf_path is None:
                    break
                
                print(f"\n开始转换: {pdf_path}")
                
                try:
                    # 转换PDF到Markdown
                    full_text, images, out_meta = convert_single_pdf(
                        str(pdf_path), 
                        self.model_lst,
                        # ocr_all_pages=True,
                        batch_multiplier=4
                    )
                    
                    # 保存Markdown文件
                    md_path = self.markdown_dir / f"{filename}.md"
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(full_text)
                    
                    # 保存图片
                    img_dir = self.markdown_dir / filename
                    img_dir.mkdir(exist_ok=True)
                    for key, img in images.items():
                        img_path = img_dir / f"{key}.png"
                        img.save(img_path)
                    
                    print(f"成功转换: {pdf_path} -> {md_path}")
                    
                    # 将向量化任务加入队列
                    self.vector_queue.put((full_text, filename))
                
                except Exception as e:
                    print(f"转换失败 {pdf_path}: {str(e)}")
                
            except Exception as e:
                print(f"转换工作器错误: {str(e)}")
            finally:
                self.conversion_queue.task_done()

    def _vector_worker(self):
        while True:
            try:
                text, filename = self.vector_queue.get()
                if text is None:
                    break
                
                print(f"\n开始向量化: {filename}")
                
                try:
                    chunks = self.chunk_processor.split_text(text)
                    
                    # 创建向量存储
                    self.current_vectorstore = Chroma.from_texts(
                        chunks,
                        self.embedding_function,
                        persist_directory=str(self.vector_dir)
                    )
                    
                    print(f"成功向量化: {filename} -> {len(chunks)} chunks")
                
                except Exception as e:
                    print(f"向量化失败 {filename}: {str(e)}")
                
            except Exception as e:
                print(f"向量化工作器错误: {str(e)}")
            finally:
                self.vector_queue.task_done()

    def download_papers(self, keyword, max_results=5):
        # 设置搜索客户端
        client = arxiv.Client()
        
        # 创建搜索查询
        search = arxiv.Search(
            query = keyword,
            max_results = max_results,
            sort_by = arxiv.SortCriterion.Relevance
        )
        
        print(f"正在搜索关键词 '{keyword}' 相关的论文...")
        
        # 使用线程池并行下载论文
        with ThreadPoolExecutor(max_workers=4) as executor:
            papers = list(enumerate(client.results(search), 1))
            executor.map(self.download_paper, papers)
        
        # 等待所有任务完成
        self.conversion_queue.join()
        self.vector_queue.join()

    def process_user_query(self, user_query, top_k=3):
        """处理用户查询"""
        print("\n开始处理查询...")
        
        # 1. 优化查询关键词
        arxiv_query = self.query_processor.rephrase_query(user_query)
        print(f"\n优化后的arxiv查询关键词: {arxiv_query}")
        
        # 2. 下载并处理论文
        self.download_papers(arxiv_query)
        
        # 3. 等待向量化完成
        time.sleep(2)  # 确保向量化完成
        
        if self.current_vectorstore:
            # 4. 执行相似度搜索
            results = self.current_vectorstore.similarity_search_with_score(
                user_query,
                k=top_k
            )
            
            # 准备chunks和metadata
            chunks = []
            metadata = []
            for doc, score in results:
                chunks.append(doc.page_content)
                metadata.append(f"Score: {score:.4f}")
            
            # 5. 生成回答
            answer = self.query_processor.process_rag_results(
                user_query, 
                chunks,
                metadata
            )
            print("\n生成的回答:")
            print(answer)
            
            # 6. 生成后续问题
            followup = self.query_processor.generate_followup_questions(
                user_query,
                answer
            )
            print("\n后续问题:")
            for i, question in enumerate(followup['follow_up'], 1):
                print(f"{i}. {question}")
        else:
            print("没有找到相关的向量存储")

def main():
    processor = PaperProcessor()
    
    while True:
        query = input("\n请输入你的问题 (输入 'q' 退出): ")
        if query.lower() == 'q':
            break
            
        try:
            processor.process_user_query(query)
        except Exception as e:
            print(f"发生错误: {str(e)}")
    
    # 停止工作线程
    processor.conversion_queue.put((None, None))
    processor.vector_queue.put((None, None))

if __name__ == "__main__":
    main() 