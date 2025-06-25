import numpy as np
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import CrossEncoder
import openai
import os
from FlagEmbedding import FlagReranker

class ReRanker:
    """重排模型"""
    
    def __init__(self, rag_config: 'RAGConfig'):
        self.logger = logging.getLogger(__name__)
        self.config = rag_config.retrieval
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3',use_fp16=True)

        print("重排模型初始化完成")
            
    def rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对检索结果进行重排"""
        #if not self.config.RERANK_ENABLE or len(chunks) == 0:
        #    return chunks
        
        pairs = [(query, chunk['text']) for chunk in chunks]

        scores = self.reranker.compute_score(pairs, normalize=True)

        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])
        
        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        self.logger.info(f"本地模型重排完成")
        return chunks[:self.config.FINAL_TOP_K]

class DocumentRetriever:
    """文档检索器"""
    
    def __init__(self, rag_config: 'RAGConfig', index_builder: 'VectorIndexBuilder'):
        self.config = rag_config.retrieval
        self.embedding_config = rag_config.embedding
        self.retrieval_config = rag_config.retrieval
        self.index_builder = index_builder
        self.reranker = ReRanker(rag_config)
        self.logger = logging.getLogger(__name__)
        
        # 初始化查询嵌入模型
        self._init_query_embedder()

    def _init_query_embedder(self):
        """初始化查询嵌入模型"""
        if self.embedding_config.EMBEDDING_PROVIDER == "openai":
            self.openai_client = openai.OpenAI(
                api_key=self.embedding_config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"),
                base_url=self.embedding_config.OPENAI_BASE_URL or None
            )
            self.query_model = None
        else:
            from sentence_transformers import SentenceTransformer
            self.query_model = SentenceTransformer(
                self.embedding_config.EMBEDDING_MODEL,
                device=self.embedding_config.DEVICE
            )
            self.openai_client = None
    
    def embed_query(self, query: str) -> np.ndarray:
        """生成查询嵌入"""
        if self.embedding_config.EMBEDDING_PROVIDER == "openai" and self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    input=[query],
                    model=self.embedding_config.EMBEDDING_MODEL
                )
                return np.array(response.data[0].embedding)
            except Exception as e:
                self.logger.error(f"OpenAI查询嵌入失败: {e}")
                raise
        else:
            return self.query_model.encode([query])[0]
    
    def search(self, query_embedding: np.ndarray, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        if top_k is None:
            top_k = self.config.INITIAL_TOP_K
        
        try:
            scores, indices = self.index_builder.index.search(
                query_embedding.astype('float32').reshape(1, -1), 
                top_k
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= self.config.SIMILARITY_THRESHOLD:
                    doc = self.index_builder.documents[idx].copy()
                    doc['similarity_score'] = float(score)
                    results.append(doc)
            
            self.logger.info(f"检索到 {len(results)} 个相关文档")
            #print(results)
            return results
            
        except Exception as e:
            self.logger.error(f"文档检索失败: {e}")
            raise
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """完整的检索流程"""
        # 生成查询嵌入
        query_embedding = self.embed_query(query)
        
        # 搜索相似文档
        documents = self.search(query_embedding, self.config.INITIAL_TOP_K)
        
        # 重排
        documents = self.reranker.rerank(query, documents)
        
        if top_k:
            documents = documents[:top_k]
        
        return documents