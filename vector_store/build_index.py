import faiss
import numpy as np
import pickle
import json
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import logging

class VectorIndexBuilder:
    """向量索引构建器"""
    
    def __init__(self, rag_config: 'RAGConfig'):
        self.config = rag_config.vector_store
        self.logger = logging.getLogger(__name__)
        self.index = None
        self.documents = []
        
    def build_faiss_index(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """构建FAISS索引"""
        try:
            dimension = embeddings.shape[1]
            self.logger.info(f"构建FAISS索引，维度: {dimension}")
            
            if self.config.FAISS_INDEX_TYPE == "IndexFlatIP":
                self.index = faiss.IndexFlatIP(dimension)
            elif self.config.FAISS_INDEX_TYPE == "IndexFlatL2":
                self.index = faiss.IndexFlatL2(dimension)
            else:
                self.index = faiss.IndexFlatIP(dimension)
            
            self.index.add(embeddings.astype('float32'))
            self.documents = documents
            
            self.logger.info(f"FAISS索引构建完成，包含 {self.index.ntotal} 个向量")
            
        except Exception as e:
            self.logger.error(f"构建FAISS索引失败: {e}")
            raise
    
    def save_index(self):
        """保存索引到磁盘"""
        try:
            index_dir = Path(self.config.INDEX_PATH)
            index_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存FAISS索引
            faiss_path = index_dir / "index.faiss"
            faiss.write_index(self.index, str(faiss_path))
            
            # 保存文档
            docs_path = index_dir / "documents.pkl"
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # 保存配置
            config_path = index_dir / "config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "store_type": self.config.INDEX_TYPE,
                    "index_type": self.config.FAISS_INDEX_TYPE,
                    "num_vectors": self.index.ntotal,
                    "dimension": self.index.d
                }, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"索引已保存到: {index_dir}")
            
        except Exception as e:
            self.logger.error(f"保存索引失败: {e}")
            raise
    
    def load_index(self) -> bool:
        """从磁盘加载索引"""
        try:
            index_dir = Path(self.config.INDEX_PATH)
            
            if not index_dir.exists():
                self.logger.warning(f"索引目录不存在: {index_dir}")
                return False
            
            # 加载FAISS索引
            faiss_path = index_dir / "index.faiss"
            if not faiss_path.exists():
                self.logger.warning(f"FAISS索引文件不存在: {faiss_path}")
                return False
            
            self.index = faiss.read_index(str(faiss_path))
            
            # 加载文档
            docs_path = index_dir / "documents.pkl"
            if docs_path.exists():
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
            
            self.logger.info(f"索引加载成功，包含 {self.index.ntotal} 个向量")
            return True
            
        except Exception as e:
            self.logger.error(f"加载索引失败: {e}")
            return False