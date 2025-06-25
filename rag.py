# ================================
# rag.py - 增强RAG系统主入口（支持增量更新）
# ================================

import logging
import json
import numpy as np
import os
import time
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# 导入所有模块
from config import RAGConfig
from chunking.extractor import StructuredPDFExtractor 
#from chunking.extractor import PDFExtractor
from chunking.embedder import ChunkVectorizer
from vector_store.build_index import VectorIndexBuilder
from vector_store.retriever import DocumentRetriever
from llm.llm_client import LLMClient


class IncrementalVectorUpdater:
    """增量向量库更新器"""
    
    def __init__(self, rag_config: RAGConfig):
        self.config = rag_config
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.pdf_extractor = StructuredPDFExtractor(rag_config)
        self.vectorizer = ChunkVectorizer(rag_config)
        self.index_builder = VectorIndexBuilder(rag_config)
    
    def load_existing_vectors(self) -> tuple[np.ndarray, List[Dict], bool]:
        """加载现有的向量和chunks"""
        try:
            embeddings_file = os.path.join(self.config.embedding.OUTPUT_DIR, "embeddings.npy")
            metadata_file = os.path.join(self.config.embedding.OUTPUT_DIR, "metadata.json")
            
            if os.path.exists(embeddings_file) and os.path.exists(metadata_file):
                embeddings = np.load(embeddings_file)
                
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                chunks = metadata.get('chunks', [])
                self.logger.info(f"成功加载现有向量: {embeddings.shape}, 文档块: {len(chunks)}")
                return embeddings, chunks, True
            else:
                self.logger.info("未找到现有向量数据，将创建新的向量库")
                return np.array([]), [], False
                
        except Exception as e:
            self.logger.error(f"加载现有向量失败: {e}")
            return np.array([]), [], False
    
    def get_processed_files(self, chunks: List[Dict]) -> set:
        """获取已处理的文件列表"""
        processed_files = set()
        for chunk in chunks:
            if 'source_file' in chunk.get('metadata', {}):
                processed_files.add(chunk['metadata']['source_file'])
        return processed_files
    
    def find_new_pdf_files(self, pdf_dir: str = None, processed_files: set = None) -> List[str]:
        """找到需要处理的新PDF文件"""
        pdf_dir = pdf_dir or self.config.PDF_DIR
        processed_files = processed_files or set()
        
        all_pdf_files = list(Path(pdf_dir).glob("*.json"))
        new_files = []
        
        for pdf_path in all_pdf_files:
            filename = pdf_path.name
            if filename not in processed_files:
                new_files.append(str(pdf_path))
        
        self.logger.info(f"发现 {len(new_files)} 个新的PDF文件需要处理")
        return new_files
    
    def process_new_pdfs(self, new_pdf_files: List[str]) -> List[Dict[str, Any]]:
        """处理新的json文件"""
        all_new_chunks = []
        
        for i, pdf_path in enumerate(new_pdf_files, 1):
            self.logger.info(f"处理新文件 [{i}/{len(new_pdf_files)}]: {Path(pdf_path).name}")
            
            try:
                chunks = self.pdf_extractor.process_structured_pdf_file(pdf_path)
                
                # 添加源文件信息
                filename = Path(pdf_path).name
                for chunk in chunks:
                    chunk["metadata"]["source_file"] = filename
                
                all_new_chunks.extend(chunks)
                self.logger.info(f"成功提取 {len(chunks)} 个文档块")
                
            except Exception as e:
                self.logger.error(f"处理PDF文件失败 {pdf_path}: {e}")
                continue
        
        return all_new_chunks
    
    def vectorize_new_chunks(self, new_chunks: List[Dict[str, Any]]) -> np.ndarray:
        """对新文档块进行向量化"""
        if not new_chunks:
            return np.array([])
        
        self.logger.info(f"开始向量化 {len(new_chunks)} 个新文档块...")
        
        # 加载向量化模型
        self.vectorizer.load_model()
        
        # 设置新的chunks
        self.vectorizer.load_chunks_from_list(new_chunks)
        
        # 进行向量化
        if self.vectorizer.vectorize_chunks():
            self.logger.info(f"新文档块向量化完成: {self.vectorizer.embeddings.shape}")
            return self.vectorizer.embeddings
        else:
            raise RuntimeError("新文档块向量化失败")
    
    def merge_vectors(self, existing_embeddings: np.ndarray, new_embeddings: np.ndarray) -> np.ndarray:
        """合并现有向量和新向量"""
        if existing_embeddings.size == 0:
            return new_embeddings
        
        if new_embeddings.size == 0:
            return existing_embeddings
        
        # 检查维度是否匹配
        if existing_embeddings.shape[1] != new_embeddings.shape[1]:
            raise ValueError(f"向量维度不匹配: 现有 {existing_embeddings.shape[1]}, 新增 {new_embeddings.shape[1]}")
        
        merged_embeddings = np.vstack([existing_embeddings, new_embeddings])
        self.logger.info(f"向量合并完成: {existing_embeddings.shape} + {new_embeddings.shape} = {merged_embeddings.shape}")
        return merged_embeddings
    
    def merge_chunks(self, existing_chunks: List[Dict], new_chunks: List[Dict]) -> List[Dict]:
        """合并现有chunks和新chunks"""
        # 为新chunks添加全局索引
        start_index = len(existing_chunks)
        for i, chunk in enumerate(new_chunks):
            chunk["metadata"]["global_index"] = start_index + i
        
        merged_chunks = existing_chunks + new_chunks
        self.logger.info(f"文档块合并完成: {len(existing_chunks)} + {len(new_chunks)} = {len(merged_chunks)}")
        return merged_chunks
    
    def update_vector_store(self, merged_embeddings: np.ndarray, merged_chunks: List[Dict]):
        """更新向量存储"""
        try:
            # 保存合并后的向量
            embeddings_file = os.path.join(self.config.embedding.OUTPUT_DIR, "embeddings.npy")
            metadata_file = os.path.join(self.config.embedding.OUTPUT_DIR, "metadata.json")
            
            # 确保输出目录存在
            os.makedirs(self.config.embedding.OUTPUT_DIR, exist_ok=True)
            
            # 保存向量
            np.save(embeddings_file, merged_embeddings)
            
            # 保存元数据
            metadata = {
                "chunks": merged_chunks,
                "model_name": self.config.embedding.EMBEDDING_MODEL,
                "embedding_dim": merged_embeddings.shape[1],
                "num_chunks": len(merged_chunks),
                "normalized": self.config.embedding.NORMALIZE_EMBEDDINGS,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"向量数据已保存到 {embeddings_file} 和 {metadata_file}")
            
            # 重建FAISS索引
            self.rebuild_faiss_index(merged_embeddings, merged_chunks)
            
        except Exception as e:
            self.logger.error(f"更新向量存储失败: {e}")
            raise
    
    def rebuild_faiss_index(self, embeddings: np.ndarray, chunks: List[Dict]):
        """重建FAISS索引"""
        try:
            self.logger.info("重建FAISS索引...")
            
            # 使用现有的index_builder重建索引
            self.index_builder.build_faiss_index(embeddings, chunks)
            self.index_builder.save_index()
            
            self.logger.info("FAISS索引重建完成")
            
        except Exception as e:
            self.logger.error(f"重建FAISS索引失败: {e}")
            raise
    
    def incremental_update(self, new_pdf_paths: List[str] = None) -> Dict[str, Any]:
        """增量更新主函数"""
        start_time = time.time()
        
        try:
            # 1. 加载现有向量和chunks
            self.logger.info("=== 开始增量向量库更新 ===")
            existing_embeddings, existing_chunks, has_existing = self.load_existing_vectors()
            
            # 2. 确定需要处理的新文件
            if new_pdf_paths:
                # 使用指定的PDF文件
                new_pdf_files = [path for path in new_pdf_paths if Path(path).exists()]
                if len(new_pdf_files) != len(new_pdf_paths):
                    missing = set(new_pdf_paths) - set(new_pdf_files)
                    self.logger.warning(f"以下文件未找到: {missing}")
            else:
                # 自动发现新文件
                processed_files = self.get_processed_files(existing_chunks) if has_existing else set()
                new_pdf_files = self.find_new_pdf_files(processed_files=processed_files)
            
            if not new_pdf_files:
                self.logger.info("没有发现需要处理的新PDF文件")
                return {
                    "status": "success",
                    "message": "没有新文件需要处理",
                    "existing_chunks": len(existing_chunks),
                    "new_chunks": 0,
                    "total_chunks": len(existing_chunks),
                    "processing_time": time.time() - start_time
                }
            
            # 3. 处理新PDF文件
            self.logger.info(f"开始处理 {len(new_pdf_files)} 个新PDF文件...")
            new_chunks = self.process_new_pdfs(new_pdf_files)
            
            if not new_chunks:
                self.logger.warning("新PDF文件未能提取出任何文档块")
                return {
                    "status": "warning",
                    "message": "新PDF文件未能提取出文档块",
                    "existing_chunks": len(existing_chunks),
                    "new_chunks": 0,
                    "total_chunks": len(existing_chunks),
                    "processing_time": time.time() - start_time
                }
            
            # 4. 向量化新文档块
            new_embeddings = self.vectorize_new_chunks(new_chunks)
            
            # 5. 合并向量和chunks
            merged_embeddings = self.merge_vectors(existing_embeddings, new_embeddings)
            merged_chunks = self.merge_chunks(existing_chunks, new_chunks)
            
            # 6. 更新向量存储
            self.update_vector_store(merged_embeddings, merged_chunks)
            
            processing_time = time.time() - start_time
            
            result = {
                "status": "success",
                "message": "增量更新完成",
                "existing_chunks": len(existing_chunks),
                "new_chunks": len(new_chunks),
                "total_chunks": len(merged_chunks),
                "new_files_processed": [Path(f).name for f in new_pdf_files],
                "processing_time": processing_time
            }
            
            self.logger.info(f"=== 增量更新完成 ===")
            self.logger.info(f"处理时间: {processing_time:.2f}秒")
            self.logger.info(f"新增文档块: {len(new_chunks)}")
            self.logger.info(f"总文档块: {len(merged_chunks)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"增量更新失败: {e}")
            raise


class RAGSystem:
    """增强的RAG系统主类（支持增量更新）"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # 初始化组件
        self.pdf_extractor = StructuredPDFExtractor(config)
        #self.pdf_extractor = PDFExtractor(config)
        self.vectorizer = ChunkVectorizer(config)
        self.index_builder = VectorIndexBuilder(config)
        self.retriever = None
        self.llm_client = LLMClient(config)
        
        # 初始化增量更新器
        self.updater = IncrementalVectorUpdater(config)
        
        # 尝试加载现有索引
        if self.index_builder.load_index():
            self.retriever = DocumentRetriever(config, self.index_builder)
            self.vectorizer.load_model()
    
    def _setup_logging(self) -> logging.Logger: 
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('rag_system.log', encoding='utf-8')
            ]
        )
        return logging.getLogger(__name__)
    
    def build_knowledge_base(self, pdf_paths: List[str] = None):
        """构建知识库"""
        self.logger.info("开始构建知识库...")
        
        if pdf_paths is None:
            # 批量处理PDF目录中的所有文件
            all_json_chunks = self.pdf_extractor.batch_process_structured_pdfs()
            if all_json_chunks:
                all_chunks = all_json_chunks
            #else:
            #    all_chunks = self.pdf_extractor.batch_process_pdfs()
        else:
        # 处理指定的PDF文件
            all_chunks = []
            for pdf_path in pdf_paths:
                if not Path(pdf_path).exists():
                    self.logger.warning(f"PDF文件不存在: {pdf_path}")
                    continue
                
                try:
                    #if pdf_path.lower().endswith(".json"):
                    all_chunks += self.pdf_extractor.process_structured_pdf_file(pdf_path)
                    #else:
                    #    all_chunks += self.pdf_extractor.process_pdf_with_column_detection(pdf_path)
                except Exception as e:
                    self.logger.error(f"处理PDF失败 {pdf_path}: {e}")
        
        if not all_chunks:
            raise ValueError("没有成功提取任何文档块")
        
        # 生成嵌入
        self.logger.info("开始向量化...")
        self.vectorizer.load_model()
        self.vectorizer.load_chunks_from_list(all_chunks)
        
        if not self.vectorizer.vectorize_chunks():
            raise RuntimeError("向量化失败")
        
        # 保存向量
        self.vectorizer.save_vectors()
        
        # 构建索引
        self.logger.info("构建向量索引...")
        self.index_builder.build_faiss_index(self.vectorizer.embeddings, self.vectorizer.chunks)
        self.index_builder.save_index()
        
        # 初始化检索器
        self.retriever = DocumentRetriever(self.config, self.index_builder)
        
        self.logger.info(f"知识库构建完成,包含 {len(all_chunks)} 个文档块")
    
    def add_documents(self, pdf_paths: List[str] = None) -> Dict[str, Any]:
        """添加新文档到向量库"""
        try:
            # 执行增量更新
            result = self.updater.incremental_update(pdf_paths)
            
            # 重新初始化检索器以使用更新后的索引
            if result["status"] == "success" and result["new_chunks"] > 0:
                if self.index_builder.load_index():
                    self.retriever = DocumentRetriever(self.config, self.index_builder)
                    self.vectorizer.load_model()
                    self.logger.info("检索器已更新，可以查询新添加的文档")
                else:
                    self.logger.warning("索引加载失败，检索器未更新")
            
            return result
            
        except Exception as e:
            self.logger.error(f"添加文档失败: {e}")
            raise
    
    def get_document_stats(self) -> Dict[str, Any]:
        """获取文档库统计信息"""
        try:
            # 加载现有数据
            existing_embeddings, existing_chunks, has_existing = self.updater.load_existing_vectors()
            
            if not has_existing:
                return {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "processed_files": []
                }
            
            # 统计文件信息
            file_stats = {}
            for chunk in existing_chunks:
                source_file = chunk.get('metadata', {}).get('source_file', 'Unknown')
                if source_file not in file_stats:
                    file_stats[source_file] = {
                        "chunks": 0,
                        "chapters": set()
                    }
                file_stats[source_file]["chunks"] += 1
                chapter = chunk.get('metadata', {}).get('chapter_title', '')
                if chapter:
                    file_stats[source_file]["chapters"].add(chapter)
            
            # 转换为可序列化的格式
            processed_files = []
            for filename, stats in file_stats.items():
                processed_files.append({
                    "filename": filename,
                    "chunks": stats["chunks"],
                    "chapters": len(stats["chapters"])
                })
            
            return {
                "total_documents": len(file_stats),
                "total_chunks": len(existing_chunks),
                "embedding_dimension": existing_embeddings.shape[1] if existing_embeddings.size > 0 else 0,
                "processed_files": processed_files
            }
            
        except Exception as e:
            self.logger.error(f"获取文档统计失败: {e}")
            return {"error": str(e)}
    
    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """查询RAG系统"""
        if self.retriever is None:
            raise RuntimeError("检索器未初始化,请先构建知识库或添加文档")
        
        self.logger.info(f"处理查询: {question}")
        start_time = time.time()
        
        try:
            # 检索相关文档
            retrieved_docs = self.retriever.retrieve(question, top_k)
            
            retrieval_time = time.time() - start_time
            
            if not retrieved_docs:
                return {
                    "question": question,
                    "answer": "抱歉,我在知识库中没有找到相关信息来回答您的问题。",
                    "sources": [],
                    "retrieval_time": retrieval_time,
                    "generation_time": 0,
                    "total_time": retrieval_time
                }
            
            # 构建提示词
            context = self._build_context(retrieved_docs)
            full_prompt = self.config.llm.SYSTEM_PROMPT.format(context=context) + f"\n\n问题: {question}\n\n回答:"
            
            # 生成回答
            generation_start = time.time()
            messages = [{"role": "user", "content": full_prompt}]
            answer = self.llm_client.generate_response(messages)
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            result = {
                "question": question,
                "answer": answer,
                "sources": [
                    {
                        "chapter": doc["metadata"]["chapter_title"],
                        "page_range": f"{doc['metadata']['start_page']}-{doc['metadata']['end_page']}",
                        "similarity": doc.get("similarity_score", 0),
                        "rerank_score": doc.get("rerank_score", 0),
                        "text_preview": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"]
                    }
                    for doc in retrieved_docs
                ],
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time
            }
            
            self.logger.info("查询处理完成")
            return result
            
        except Exception as e:
            self.logger.error(f"查询处理失败: {e}")
            raise
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """构建上下文"""
        if not documents:
            return "没有找到相关信息。"
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents, 1):
            doc_info = f"""
文档块 {i}:
章节: {doc['metadata']['chapter_title']}
页码: {doc['metadata']['start_page']}-{doc['metadata']['end_page']}
相似度: {doc.get('similarity_score', 0):.3f}
内容: {doc['text']}
---
"""
            
            if current_length + len(doc_info) > self.config.retrieval.MAX_CONTEXT_LENGTH:
                break
                
            context_parts.append(doc_info)
            current_length += len(doc_info)
        
        return "\n".join(context_parts)
    
    def interactive_chat(self):
        """增强的交互式聊天界面"""
        print("=== 增强RAG文档问答系统 ===")
        print("可用命令:")
        print("  正常问题: 直接输入问题进行查询")
        print("  'add': 添加新的PDF文档")
        print("  'add [路径]': 添加指定路径的PDF文档")
        print("  'stats': 查看文档库统计信息")
        print("  'config': 查看当前配置")
        print("  'rebuild': 重新构建整个知识库")
        print("  'quit': 退出系统")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n您的输入: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("感谢使用!")
                    break
                
                elif user_input.lower().startswith('add'):
                    # 处理add命令
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        # 指定路径
                        pdf_path = parts[1].strip()
                        if not Path(pdf_path).exists():
                            print(f"文件不存在: {pdf_path}")
                            continue
                        result = self.add_documents([pdf_path])
                    else:
                        # 自动发现
                        result = self.add_documents()
                    
                    print(f"添加结果: {result['message']}")
                    if result['status'] == 'success':
                        print(f"新增文档块: {result['new_chunks']}")
                        print(f"总文档块: {result['total_chunks']}")
                        print(f"处理时间: {result['processing_time']:.2f}秒")
                        if result.get('new_files_processed'):
                            print(f"处理的文件: {', '.join(result['new_files_processed'])}")
                
                elif user_input.lower() == 'stats':
                    stats = self.get_document_stats()
                    if 'error' in stats:
                        print(f"获取统计信息失败: {stats['error']}")
                    else:
                        print(f"📊 文档库统计:")
                        print(f"  总文档数: {stats['total_documents']}")
                        print(f"  总文档块: {stats['total_chunks']}")
                        print(f"  向量维度: {stats['embedding_dimension']}")
                        print(f"  已处理文件:")
                        for file_info in stats['processed_files'][:10]:  # 显示前10个
                            print(f"    📄 {file_info['filename']}: {file_info['chunks']} 块, {file_info['chapters']} 章节")
                        if len(stats['processed_files']) > 10:
                            print(f"    ... 还有 {len(stats['processed_files']) - 10} 个文件")
                
                elif user_input.lower() == 'config':
                    print(f"⚙️ 当前配置:")
                    print(f"  LLM类型: {self.config.llm.LLM_TYPE}")
                    print(f"  LLM模型: {self.config.llm.MODEL_NAME}")
                    print(f"  嵌入模型: {self.config.embedding.EMBEDDING_MODEL}")
                    print(f"  嵌入提供商: {self.config.embedding.EMBEDDING_PROVIDER}")
                    print(f"  检索数量: {self.config.retrieval.FINAL_TOP_K}")
                    print(f"  相似度阈值: {self.config.retrieval.SIMILARITY_THRESHOLD}")
                    print(f"  PDF目录: {self.config.PDF_DIR}")
                
                elif user_input.lower() == 'rebuild':
                    confirm = input("⚠️ 这将重新构建整个知识库，继续吗? (y/N): ").strip().lower()
                    if confirm == 'y':
                        print("🔄 开始重新构建知识库...")
                        self.build_knowledge_base()
                        print("✅ 知识库重新构建完成!")
                    else:
                        print("操作已取消")
                
                elif user_input:
                    # 正常查询
                    if not self.retriever:
                        print("❌ 检索器未初始化，请先添加文档或检查是否存在现有向量库")
                        print("   可以使用 'add' 命令添加文档，或 'rebuild' 重新构建知识库")
                        continue
                    
                    result = self.query(user_input)
                    
                    print(f"\n🤖 回答: {result['answer']}")
                    print(f"\n⏱️ 用时: 检索 {result['retrieval_time']:.2f}s | 生成 {result['generation_time']:.2f}s | 总计 {result['total_time']:.2f}s")
                    
                    #if result.get("sources"):
                    #    print(f"\n📚 参考来源 ({len(result['sources'])} 个):")
                    #    for i, source in enumerate(result["sources"], 1):
                    #        print(f"  {i}. {source['chapter']} (页{source['page_range']}) - 相似度: {source['similarity']:.3f}")
                    #        print(f"     预览: {source['text_preview']}")
                
            except KeyboardInterrupt:
                print("\n\n感谢使用!")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
                self.logger.error(f"交互式聊天错误: {e}")


# ================================
# 使用示例和配置
# ================================

def get_openai_config() -> RAGConfig:
    """OpenAI配置示例"""
    config = RAGConfig()
    config.llm.LLM_TYPE = "openai"
    config.llm.MODEL_NAME = "deepseek-r1-0528"
    config.llm.API_KEY = ""
    config.llm.API_BASE = ''
    config.llm.TEMPERATURE = 0.1
    return config

#def get_ollama_config() -> RAGConfig:
#    """Ollama本地配置示例"""
#    config = RAGConfig()
#    config.llm.LLM_TYPE = "ollama"
#    config.llm.MODEL_NAME = "llama2"
#    config.llm.API_BASE = "http://localhost:11434"
#    config.llm.TEMPERATURE = 0.1
#    return config

def main():
    """主函数"""
    print("=== 增强PDF文档RAG系统 ===")
    #print("选择LLM后端:")
    #print("1. OpenAI API (需要API key)")
    #print("2. Ollama (本地模型)")
    
    #choice = input("请选择 (1-2): ").strip()
    
    #if choice == "1":
    config = get_openai_config()
    if not config.llm.API_KEY:
        api_key = input("请输入OpenAI API Key: ").strip()
        if api_key:
            config.llm.API_KEY = api_key
    #elif choice == "2":
    #    config = get_ollama_config()
    #    model = input("请输入Ollama模型名称 (默认: llama2): ").strip()
    #    if model:
    #        config.llm.MODEL_NAME = model
    #else:
    #    print("无效选择,使用默认OpenAI配置")
    #    config = get_openai_config()
    
    try:
        # 初始化RAG系统
        rag = RAGSystem(config)
        
        # 检查现有向量库状态
        stats = rag.get_document_stats()
        if stats.get('total_chunks', 0) > 0:
            print(f"\n✅ 发现现有向量库:")
            print(f"   📄 {stats['total_documents']} 个文档")
            print(f"   📦 {stats['total_chunks']} 个文档块")
            print(f"   📐 {stats['embedding_dimension']} 维向量")
        else:
            print("\n🆕 未发现现有向量库")
            
            # 检查文件目录
            doc_dir = Path(config.PDF_DIR)
            if not doc_dir.exists():
                doc_dir.mkdir(parents=True, exist_ok=True)
                print(f"📁 已创建PDF目录: {doc_dir}")
            
            doc_files = list(doc_dir.glob("*.json"))
            if doc_files:
                print(f"📋 发现 {len(doc_files)} 个json文件")
                build_choice = input("是否立即构建知识库? (Y/n): ").strip().lower()
                if build_choice != 'n':
                    print("🔄 开始构建知识库...")
                    rag.build_knowledge_base()
                    print("✅ 知识库构建完成!")
            else:
                print(f"📁 PDF目录 {doc_files} 中没有找到PDF文件")
                print("💡 可以使用 'add [文件路径]' 命令添加文档")
        
        # 启动交互式聊天
        rag.interactive_chat()
        
    except Exception as e:
        print(f"❌ 初始化RAG系统失败: {e}")
        print("请检查:")
        print("1. PDF文件是否存在")
        print("2. LLM配置是否正确") 
        print("3. 相关依赖是否已安装")

if __name__ == "__main__":
    main()