import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import pickle
import os
from tqdm import tqdm
import time
from pathlib import Path
import logging
import openai

class ChunkVectorizer:
    """文档块向量化器"""
    
    def __init__(self, rag_config: 'RAGConfig'):
        self.config = rag_config.embedding
        self.model = None
        self.openai_client = None
        self.chunks = []
        self.embeddings = None
        
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """加载嵌入模型"""
        if self.config.EMBEDDING_PROVIDER == "openai":
            self._load_openai_client()
        else:
            self._load_local_model()
        print(f"Loading model: {self.config.EMBEDDING_MODEL}")
        print(f"Using device: {self.config.DEVICE}")
        
    def _load_openai_client(self):
        """加载OpenAI客户端"""
        print(f"Loading OpenAI model: {self.config.EMBEDDING_MODEL}")
        
        try:
            self.openai_client = openai.OpenAI(
                api_key=self.config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"),
                base_url=self.config.OPENAI_BASE_URL or None
            )
            print("OpenAI client loaded successfully")
            
        except Exception as e:
            print(f"Error loading OpenAI client: {e}")
            print("Falling back to local model")
            self._load_local_model()

    def _load_local_model(self):
        """加载本地模型（备选方案）"""
        print(f"Loading local model: {self.config.EMBEDDING_MODEL}")
        print(f"Using device: {self.config.DEVICE}")
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                self.config.EMBEDDING_MODEL, 
                device=self.config.DEVICE
            )
            self.model.max_seq_length = self.config.MAX_SEQ_LENGTH
            print(f"Local model loaded successfully")
            
        except Exception as e:
            print(f"Error loading local model: {e}")
            raise
    
    def load_chunks(self, chunks_file: str = None) -> bool:
        """加载文档块"""
        chunks_file = chunks_file or self.config.INPUT_CHUNKS_FILE
        
        if not os.path.exists(chunks_file):
            print(f"Error: Chunks file {chunks_file} not found!")
            return False
            
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = [json.loads(line.strip()) for line in f if line.strip()]
                
            print(f"Loaded {len(self.chunks)} chunks from {chunks_file}")
            self._print_chunks_stats()
            return True
            
        except Exception as e:
            print(f"Error loading chunks: {e}")
            return False
    
    def load_chunks_from_list(self, chunks: List[Dict[str, Any]]) -> bool:
        """从列表加载文档块"""
        self.chunks = chunks
        print(f"Loaded {len(self.chunks)} chunks from list")
        self._print_chunks_stats()
        return True
            
    def _print_chunks_stats(self):
        """打印块统计信息"""
        if not self.chunks:
            return
            
        text_lengths = [len(chunk['text']) for chunk in self.chunks]
        avg_length = np.mean(text_lengths)
        max_length = max(text_lengths)
        min_length = min(text_lengths)
        
        print(f"Chunk Statistics:")
        print(f"  Average length: {avg_length:.1f} characters")
        print(f"  Max length: {max_length} characters")
        print(f"  Min length: {min_length} characters")
        
        chapters = {}
        for chunk in self.chunks:
            chapter = chunk['metadata']['chapter_title']
            chapters[chapter] = chapters.get(chapter, 0) + 1
            
        print(f"  Chapters: {len(chapters)}")
        for chapter, count in list(chapters.items())[:5]:
            print(f"    {chapter[:30]}...: {count} chunks")
        if len(chapters) > 5:
            print(f"    ... and {len(chapters) - 5} more chapters")
            
    def vectorize_chunks(self) -> bool:
        """向量化所有文档块"""
        if not self.chunks:
            print("No chunks loaded!")
            return False
            
        #if not self.model:
        #    print("Model not loaded!")
        #    return False
        
        #print(f"Starting vectorization of {len(self.chunks)} chunks...")
        
        #texts = [chunk['text'] for chunk in self.chunks]

        #if self.config.EMBEDDING_PROVIDER == "openai" and self.openai_client:
        #    return self._vectorize_with_openai(texts)
        #else:
        #    return self._vectorize_with_local_model(texts)

        print(f"Starting vectorization of {len(self.chunks)} chunks...")
        texts = [chunk['text'] for chunk in self.chunks]

        if self.config.EMBEDDING_PROVIDER.lower() == "openai":
            if not self.openai_client:
                print("OpenAI client not loaded!")
                return False
            return self._vectorize_with_openai(texts)
        
        if not self.model:
            print("Local model not loaded!")
            return False
        return self._vectorize_with_local_model(texts)

    def _vectorize_with_openai(self, texts: List[str]) -> bool:
        """使用OpenAI API进行向量化"""
        print(f"Using OpenAI API for vectorization")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        
        all_embeddings = []
        start_time = time.time()
        
        for i in tqdm(range(0, len(texts), self.config.BATCH_SIZE), 
                     desc="Vectorizing with OpenAI", disable=not self.config.SHOW_PROGRESS):
            
            batch_texts = texts[i:i + self.config.BATCH_SIZE]
            
            try:
                response = self.openai_client.embeddings.create(
                    input=batch_texts,
                    model=self.config.EMBEDDING_MODEL
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # 添加延迟以避免API限制
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing batch {i//self.config.BATCH_SIZE + 1}: {e}")
                # 尝试单个处理
                for text in batch_texts:
                    try:
                        response = self.openai_client.embeddings.create(
                            input=[text],
                            model=self.config.EMBEDDING_MODEL
                        )
                        all_embeddings.append(response.data[0].embedding)
                        time.sleep(0.1)
                    except:
                        print(f"Failed to encode text, adding zero vector")
                        # 获取模型维度（text-embedding-3-large是3072维）
                        all_embeddings.append([0.0] * 3072)
        
        self.embeddings = np.array(all_embeddings)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"OpenAI vectorization completed!")
        print(f"  Time taken: {processing_time:.2f} seconds")
        print(f"  Embeddings shape: {self.embeddings.shape}")
        print(f"  Average time per chunk: {processing_time/len(self.chunks):.3f} seconds")
        
        return True
    
    def _vectorize_with_local_model(self, texts: List[str]) -> bool:
        """使用本地模型进行向量化（备选方案）"""
        if not self.model:
            print("Local model not loaded!")
            return False
        
        print(f"Using local model for vectorization")

        all_embeddings = []
        start_time = time.time()
        
        for i in tqdm(range(0, len(texts), self.config.BATCH_SIZE), 
                     desc="Vectorizing", disable=not self.config.SHOW_PROGRESS):
            
            batch_texts = texts[i:i + self.config.BATCH_SIZE]
            
            try:
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    normalize_embeddings=self.config.NORMALIZE_EMBEDDINGS,
                    batch_size=len(batch_texts)
                )
                
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error processing batch {i//self.config.BATCH_SIZE + 1}: {e}")
                for text in batch_texts:
                    try:
                        embedding = self.model.encode(
                            text,
                            convert_to_tensor=False,
                            show_progress_bar=False,
                            normalize_embeddings=self.config.NORMALIZE_EMBEDDINGS
                        )
                        all_embeddings.append(embedding)
                    except:
                        print(f"Failed to encode text, adding zero vector")
                        dim = self.model.get_sentence_embedding_dimension()
                        all_embeddings.append(np.zeros(dim))
        
        self.embeddings = np.array(all_embeddings)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Vectorization completed!")
        print(f"  Time taken: {processing_time:.2f} seconds")
        print(f"  Embeddings shape: {self.embeddings.shape}")
        print(f"  Average time per chunk: {processing_time/len(self.chunks):.3f} seconds")
        
        return True
        
    def save_vectors(self, formats: List[str] = None):
        """保存向量到不同格式"""
        if self.embeddings is None:
            print("No embeddings to save!")
            return
            
        formats = formats or self.config.SAVE_FORMATS
        
        for format_type in formats:
            try:
                if format_type == "numpy":
                    self._save_numpy()
                elif format_type == "json":
                    self._save_json()
                elif format_type == "faiss":
                    self._save_faiss()
                elif format_type == "pickle":
                    self._save_pickle()
                else:
                    print(f"Unknown format: {format_type}")
                    
            except Exception as e:
                print(f"Error saving in {format_type} format: {e}")
                
    def _save_numpy(self):
        """保存为numpy格式"""
        embeddings_file = os.path.join(self.config.OUTPUT_DIR, "embeddings.npy")
        metadata_file = os.path.join(self.config.OUTPUT_DIR, "metadata.json")
        
        np.save(embeddings_file, self.embeddings)
        
        metadata = {
            "chunks": self.chunks,
            "model_name": self.config.EMBEDDING_MODEL,
            "embedding_dim": self.embeddings.shape[1],
            "num_chunks": len(self.chunks),
            "normalized": self.config.NORMALIZE_EMBEDDINGS
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        print(f"Saved numpy format to {embeddings_file} and {metadata_file}")
        
    def _save_json(self):
        """保存为JSON格式"""
        output_file = os.path.join(self.config.OUTPUT_DIR, "vectors.json")
        
        data = {
            "model_name": self.config.EMBEDDING_MODEL,
            "embedding_dim": self.embeddings.shape[1],
            "normalized": self.config.NORMALIZE_EMBEDDINGS,
            "chunks": []
        }
        
        for i, chunk in enumerate(self.chunks):
            chunk_data = {
                "id": i,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "embedding": self.embeddings[i].tolist()
            }
            data["chunks"].append(chunk_data)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"Saved JSON format to {output_file}")
        
    def _save_faiss(self):
        """保存为FAISS索引格式"""
        try:
            import faiss
        except ImportError:
            print("FAISS not installed. Install with: pip install faiss-cpu")
            return
            
        dimension = self.embeddings.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        
        if self.config.NORMALIZE_EMBEDDINGS:
            index = faiss.IndexFlatIP(dimension)
            
        embeddings_float32 = self.embeddings.astype('float32')
        index.add(embeddings_float32)
        
        index_file = os.path.join(self.config.OUTPUT_DIR, "faiss_index.index")
        faiss.write_index(index, index_file)
        
        id_mapping_file = os.path.join(self.config.OUTPUT_DIR, "id_mapping.json")
        id_mapping = {
            i: {
                "text": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"],
                "metadata": chunk["metadata"]
            }
            for i, chunk in enumerate(self.chunks)
        }
        
        with open(id_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(id_mapping, f, ensure_ascii=False, indent=2)
            
        print(f"Saved FAISS index to {index_file} and mapping to {id_mapping_file}")
        
    def _save_pickle(self):
        """保存为pickle格式"""
        output_file = os.path.join(self.config.OUTPUT_DIR, "vectorized_chunks.pkl")
        
        data = {
            "embeddings": self.embeddings,
            "chunks": self.chunks,
            "model_name": self.config.EMBEDDING_MODEL,
            "config": self.config
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Saved pickle format to {output_file}")