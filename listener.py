import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings,)

path_db = "data/ChromaDB"

model_name = "/home/codelformat/shared_models/bge-m3"

embedding_function = SentenceTransformerEmbeddings(model_name=model_name, model_kwargs={"device": "cuda"})


# embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cuda"})
vector_store = Chroma(persist_directory=path_db, embedding_function=embedding_function)

# 初始化FastAPI应用
app = FastAPI()

# 加载模型路径
MODEL_PATH = "/home/codelformat/shared_models/bge-m3"
# 加载嵌入模型
model = SentenceTransformer(MODEL_PATH)

# 请求体模型
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5  # 返回最相似的K个结果，默认5个

# 将输入字符串向量化并查询Chroma数据库
def query_chroma_db(query_text, top_k=5):
    # 使用模型将输入字符串转化为向量
    query_embedding = model.encode([query_text])[0]

    # 查询Chroma数据库，获得最相似的文本
    results = vector_store.similarity_search(query_text,k=top_k)
    source_knowledge = "\n".join([x.page_content for x in results])

    return source_knowledge

# 接收查询请求
@app.post("/query")
def query_db(request: QueryRequest):
    try:
        print(f"Received query: {request.query}")
        results = query_chroma_db(request.query, request.top_k)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 启动服务器并监听9999端口
if __name__ == "__main__":
    print("Server is running on port 9999...")
    uvicorn.run(app, host="0.0.0.0", port=9999)