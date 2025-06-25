# ================================
# configs.py - 配置管理
# ================================

from dataclasses import dataclass
from typing import List, Optional
import torch
import os

@dataclass
class EmbeddingConfig:
    """嵌入配置"""
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_PROVIDER: str = "openai"
    DEVICE: str = "cpu"
    MAX_SEQ_LENGTH: int = 4000
    BATCH_SIZE: int = 16
    OPENAI_MAX_BATCH_SIZE: int = 64  
    NORMALIZE_EMBEDDINGS: bool = True
    SHOW_PROGRESS: bool = True

    OPENAI_API_KEY: str = ""  
    OPENAI_BASE_URL: str = "" 

    # 输入输出路径
    INPUT_CHUNKS_FILE: str = "chunks.jsonl"
    OUTPUT_DIR: str = "vectors"
    SAVE_FORMATS: List[str] = None
    
    def __post_init__(self):
        if self.SAVE_FORMATS is None:
            self.SAVE_FORMATS = ["numpy", "json", "faiss", "pickle"]

@dataclass
class ChunkingConfig:
    """分块配置"""
    # 语义分块参数
    SEMANTIC_EMBEDDING_PROVIDER: str = "openai"  # 新增
    SEMANTIC_EMBEDDING_MODEL: str = "BAAI/bge-m3"  # 新增
    SEMANTIC_CHUNK_THRESHOLD: float = 70
    SEMANTIC_MAX_CHUNK_SIZE: int = 1000
    SEMANTIC_MAX_BATCH_SIZE: int = 64
    
    # 标题检测参数
    HEADING_FONT_SIZE_THRESHOLD: float = 12.0
    HEADING_STYLE_WEIGHTS: dict = None
    COMMON_TITLES: List[str] = None
    
    def __post_init__(self):
        if self.HEADING_STYLE_WEIGHTS is None:
            self.HEADING_STYLE_WEIGHTS = {
                "font_size": 0.4,
                "centered": 0.3,
                "all_caps": 0.2,
                "bold": 0.3,
                "section_prefix": 0.4
            }
        
        if self.COMMON_TITLES is None:
            self.COMMON_TITLES = [
                "introduction", "abstract", "conclusion", "references", "appendix",
                "methodology", "results", "discussion", "background", "literature",
                "chapter", "section", "summary", "overview", "analysis",
                "引言", "摘要", "结论", "参考文献", "附录", "方法", "结果", "讨论", "背景"
            ]

@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    INDEX_TYPE: str = "faiss"  # faiss, qdrant
    INDEX_PATH: str = "vector_index"
    COLLECTION_NAME: str = "documents"
    
    # FAISS配置
    FAISS_INDEX_TYPE: str = "IndexFlatIP"  # IndexFlatIP, IndexFlatL2
    
    # Qdrant配置
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None

@dataclass
class RetrievalConfig:
    """检索配置"""
    INITIAL_TOP_K: int = 60  # 初始检索数量
    FINAL_TOP_K: int = 40     # 最终返回数量
    SIMILARITY_THRESHOLD: float = 0.3
    MAX_CONTEXT_LENGTH: int = 4000
    
    # 重排配置
    RERANK_MODEL_TYPE = "openai"
    RERANK_ENABLE: bool = False
    RERANK_PROVIDER: str = "openai"  # openai, local, none
    API_KEY: str = ""
    API_BASE: str = ""
    RERANK_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    RERANK_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    RERANK_BATCH_SIZE: int = 16

@dataclass
class LLMConfig:
    """LLM配置"""
    LLM_TYPE: str = "openai"  # openai, ollama, huggingface
    MODEL_NAME: str = "gpt-4.1-mini"
    API_KEY: str = ""
    API_BASE: str = ""
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 1000
    
    # 提示词模板
    SYSTEM_PROMPT: str = """你是一个专业的文档问答助手。请根据提供的上下文信息回答用户的问题。

规则:
1. 只基于提供的上下文信息回答问题
2. 如果上下文中没有相关信息,请明确说明
3. 保持回答的准确性和客观性
4. 可以适当引用具体的章节或段落
5. 使用中文回答我

上下文信息:
{context}

请回答以下问题:"""

@dataclass
class RAGConfig:
    """RAG主配置"""
    embedding: EmbeddingConfig = None
    chunking: ChunkingConfig = None
    vector_store: VectorStoreConfig = None
    retrieval: RetrievalConfig = None
    llm: LLMConfig = None
    USE_STRUCTURED_READER: bool = True
    
    # PDF处理相关
    PDF_DIR: str = "E:/rag/script/data/json1"
    
    def __post_init__(self):
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.vector_store is None:
            self.vector_store = VectorStoreConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()
        if self.llm is None:
            self.llm = LLMConfig()