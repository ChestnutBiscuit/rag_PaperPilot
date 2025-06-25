import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import SentenceTransformerEmbeddings
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import os
import logging
import openai
import os
from config import RAGConfig
from dataclasses import dataclass
import json
from pathlib import Path

class OpenAIEmbeddings:
    """OpenAI嵌入模型封装"""
    def __init__(self, model_name: str = "BAAI/bge-m3", api_key: str = None, base_url: str = None, max_batch_size: int = 64):
        self.model_name = model_name
        self.max_batch_size = max_batch_size  # 添加最大批处理大小限制
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or None
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文档列表转为嵌入向量 - 支持批处理"""
        if not texts:
            return []
        
        all_embeddings = []
        
        # 将输入文本分批处理
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            
            try:
                print(f"处理批次 {i//self.max_batch_size + 1}: {len(batch)} 个文本")
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"批次 {i//self.max_batch_size + 1} 嵌入失败: {e}")
                # 如果批次仍然太大，尝试单个处理
                if "batch size" in str(e).lower() and len(batch) > 1:
                    print(f"尝试单个处理批次中的 {len(batch)} 个文本...")
                    for single_text in batch:
                        try:
                            single_response = self.client.embeddings.create(
                                input=[single_text],
                                model=self.model_name
                            )
                            all_embeddings.append(single_response.data[0].embedding)
                        except Exception as single_e:
                            print(f"单个文本嵌入失败: {single_e}")
                            # 如果单个文本也失败，添加零向量占位
                            all_embeddings.append([0.0] * 1024)  # 假设嵌入维度为1024
                else:
                    raise
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """将查询文本转为嵌入向量"""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model_name
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"查询嵌入失败: {e}")
            raise

class CustomEmbeddings:
    """封装sentence-transformers嵌入模型"""
    def __init__(self, model_name: str = None, device: str = None):
        model_name = RAGConfig.EmbeddingConfig.EMBEDDING_MODEL
        device = RAGConfig.EmbeddingConfig.DEVICE
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            print(f"Warning: Could not load model {model_name}, falling back to all-MiniLM-L6-v2")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文档列表转为嵌入向量"""
        return self.model.encode(
            texts, 
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """将查询文本转为嵌入向量"""
        return self.model.encode(
            text, 
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()


@dataclass
class StructuredElement:
    "结构化PDF元素"
    label: str
    bbox: list[float]
    text: str
    reading_order: int
    page_number: int

class StructuredPDFExtractor:
    """结构化PDF文档提取器"""

    def __init__(self, rag_config: 'RAGConfig'):
        self.config = rag_config
        self.chunk_config = rag_config.chunking
        self.embedding_config = rag_config.embedding
        self.pdf_dir = rag_config.PDF_DIR
        self.logger = logging.getLogger(__name__)

    def parse_structured_pdf_results(self, structured_data: Dict[str, Any]) -> List[StructuredElement]:
        """解析结构化PDF结果"""
        elements = []

        for page_data in structured_data.get("pages", []):
            page_number = page_data.get("page_number", 0)
            
            for element_data in page_data.get("elements", []):
                element = StructuredElement(
                    label=element_data.get("label", ""),
                    bbox=element_data.get("bbox", [0, 0, 0, 0]),
                    text=element_data.get("text", "").strip(),
                    reading_order=element_data.get("reading_order", 0),
                    page_number=page_number
                )
                
                if element.text:  # 只添加有文本内容的元素
                    elements.append(element)
        
        return elements
    def load_structured_pdf_data(self, json_file_path: str) -> Optional[Dict[str, Any]]:
        """从JSON文件加载结构化PDF数据"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载结构化PDF数据失败: {e}")
            return None

    def extract_sections_and_content(self, elements: List[StructuredElement]) -> List[Dict[str, Any]]:
        """从结构化元素中提取章节和内容"""
        sections = []
        current_section = None
        current_subsection = None
        content_buffer = []
        
        # 按页码和阅读顺序排序
        sorted_elements = sorted(elements, key=lambda x: (x.page_number, x.reading_order))
        
        for element in sorted_elements:
            # 跳过页眉页脚
            if element.label in ['header', 'foot']:
                continue
            
            # 处理标题（通常只有一个）
            if element.label == 'title':
                if current_section:
                    sections.append(self._create_section_chunk(current_section, content_buffer))
                current_section = {
                    'title': element.text,
                    'level': 0,
                    'start_page': element.page_number,
                    'type': 'title'
                }
                content_buffer = []
                continue
            
            # 处理作者信息
            if element.label == 'author':
                content_buffer.append(f"作者: {element.text}")
                continue
            
            # 处理章节标题
            if element.label == 'sec':
                # 保存之前的章节
                if current_section and content_buffer:
                    sections.append(self._create_section_chunk(current_section, content_buffer))
                
                current_section = {
                    'title': element.text,
                    'level': 1,
                    'start_page': element.page_number,
                    'type': 'section'
                }
                #current_subsection = None
                content_buffer = []
                continue
            
            # 处理子章节标题
            #if element.label == 'sub_sec':
            #    # 如果有当前子章节的内容，先保存
            #    if current_subsection and content_buffer:
            #        sections.append(self._create_section_chunk(current_section, current_subsection, content_buffer))
            #    
            #    current_subsection = {
            #        'title': element.text,
            #        'level': 2,
            #        'start_page': element.page_number,
            #        'type': 'subsection'
            #    }
            #    content_buffer = []
            #    continue
            
            # 处理段落内容
            if element.label == 'para':
                content_buffer.append(element.text)
                continue
            
            # 处理图片说明
            if element.label in ['fig', 'cap']:
                # 图片和图注作为特殊内容添加
                if element.label == 'fig':
                    content_buffer.append(f"[图片: {element.text}]")
                else:  # cap
                    content_buffer.append(f"图注: {element.text}")
                continue
        
        # 处理最后的章节
        if current_section and content_buffer:
            sections.append(self._create_section_chunk(current_section, content_buffer))
        with open('text.txt',  'w', encoding='utf-8') as out_file:
            print(sections, file=out_file)

        return sections

    def _create_section_chunk(self, section: Dict, content: List[str]) -> Dict[str, Any]:
        """创建章节块"""
        if not content:
            return None
        
        # 确定章节信息
        #if subsection:
        #    title = f"{section['title']} - {subsection['title']}"
        #    level = subsection['level']
        #    start_page = subsection['start_page']
        else:
            title = section['title']
            level = section['level']
            start_page = section['start_page']
        
        # 合并内容
        full_content = '\n\n'.join(content)
        
        return {
            'title': title,
            'content': full_content,
            'level': level,
            'start_page': start_page,
        }

    def process_structured_pdf_file(self, json_file_path: str) -> List[Dict[str, Any]]:
        """处理结构化json文件"""
        print(f"\n处理结构化json文件: {json_file_path}")
        
        # 加载结构化数据
        structured_data = self.load_structured_pdf_data(json_file_path)
        if not structured_data:
            return []
        
        source_file = structured_data.get("source_file", Path(json_file_path).stem)
        total_pages = structured_data.get("total_pages", 0)
        
        print(f"源文件: {source_file}, 总页数: {total_pages}")
        
        # 解析结构化元素
        elements = self.parse_structured_pdf_results(structured_data)
        print(f"解析到 {len(elements)} 个结构化元素")
        
        # 提取章节和内容
        sections = self.extract_sections_and_content(elements)
        sections = [s for s in sections if s is not None]  # 过滤空章节
        
        print(f"提取到 {len(sections)} 个章节")
        
        # 对每个章节进行语义分块
        all_chunks = []
        for i, section in enumerate(sections):
            print(f"处理章节 {i+1}/{len(sections)}: {section['title'][:50]}...")
            
            # 跳过参考文献章节
            if self.is_reference_section(section['title']):
                print(f"跳过参考文献章节: {section['title']}")
                continue
            
            # 进行语义分块
            semantic_chunks = self.semantic_chunking(section['content'])
            print(f"done")
            
            for chunk_idx, chunk_text in enumerate(semantic_chunks):
                chunk = {
                    "text": chunk_text,
                    "metadata": {
                        "chapter_title": section['title'],
                        "chapter_level": section['level'],
                        "start_page": section['start_page'],
                        "end_page": section['start_page'],  # 结构化数据中每个元素只有起始页
                        "chunk_index": chunk_idx,
                        "total_chunks": len(semantic_chunks),
                        #"section_type": section['type'],
                        "embedding_model": self.config.embedding.EMBEDDING_MODEL,
                        "extraction_method": "structured_pdf"
                    }
                }
                all_chunks.append(chunk)
        
        # 添加源文件信息
        #for chunk in all_chunks:
        #    chunk["metadata"]["source_file"] = Path(source_file).name
        
        self.print_chunk_stats(all_chunks)
        
        if all_chunks:
            self.save_chunks(all_chunks, output_format="csv")
            print(f"保存了 {len(all_chunks)} 个文档块")
        
        return all_chunks

    def batch_process_structured_pdfs(self, json_dir: str = None, suffix: str = ".json") -> List[Dict[str, Any]]:
        """批量处理结构化PDF JSON文件"""
        json_dir = json_dir or self.pdf_dir
        all_chunks = []
        
        # 查找JSON文件
        json_files = []
        if Path(json_dir).is_dir():
            json_files = [f for f in os.listdir(json_dir) if f.lower().endswith(suffix)]
        else:
            # 如果传入的是单个文件路径
            if json_dir.endswith(suffix) and os.path.exists(json_dir):
                json_files = [json_dir]
                json_dir = os.path.dirname(json_dir)

        print(f"在 {json_dir} 中找到 {len(json_files)} 个JSON文件")

        for i, filename in enumerate(json_files, start=1):
            if os.path.isabs(filename):
                json_path = filename
            else:
                json_path = os.path.join(json_dir, filename)
                
            print(f"\n[{i}/{len(json_files)}] 处理: {filename}")

            try:
                chunks = self.process_structured_pdf_file(json_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"处理文件失败 {filename}: {e}")
                continue

        return all_chunks

    def is_reference_section(self, title: str) -> bool:
        """检查是否是参考文献章节"""
        title_lower = title.lower()
        reference_keywords = [
            'references', 'reference', 'bibliography', 'citations',
            '参考文献', '引用', '文献'
        ]
        return any(keyword in title_lower for keyword in reference_keywords)

    def semantic_chunking(self, text: str) -> List[str]:
        """使用OpenAI嵌入进行语义分块"""
        if not text.strip():
            return []

        try:
            # 使用OpenAI嵌入模型
            if self.chunk_config.SEMANTIC_EMBEDDING_PROVIDER == "openai":
                embedder = OpenAIEmbeddings(
                    model_name=self.chunk_config.SEMANTIC_EMBEDDING_MODEL,
                    api_key=self.embedding_config.OPENAI_API_KEY,
                    base_url=self.embedding_config.OPENAI_BASE_URL
                )
            else:
                # 使用本地模型作为备选
                embedder = CustomEmbeddings(
                    self.config.embedding.EMBEDDING_MODEL, 
                    self.config.embedding.DEVICE
                )
            
            chunker = SemanticChunker(
                embeddings=embedder,
                breakpoint_threshold_amount=self.chunk_config.SEMANTIC_CHUNK_THRESHOLD,
            )
            return chunker.split_text(text)
            
        except Exception as e:
            print(f"Warning: 语义分块失败,回退到简单分割: {e}")
            # 简单分割作为备选方案
            sentences = re.split(r'[.!?]+', text)
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk + sentence) > self.chunk_config.SEMANTIC_MAX_CHUNK_SIZE:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += sentence + ". "

            if current_chunk:
                chunks.append(current_chunk.strip())

            return chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_format: str = "jsonl"):
        """保存分块结果"""
        if output_format == "jsonl":
            import json
            with open("chunks.jsonl", "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        elif output_format == "csv":
            import csv
            with open("chunks.csv", "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["text", "chapter_title", "chapter_level", "start_page", "end_page", 
                             "chunk_index", "total_chunks", "embedding_model", "extraction_method"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for chunk in chunks:
                    row = {"text": chunk["text"]}
                    row.update(chunk["metadata"])
                    writer.writerow(row)
        else:
            raise ValueError("Unsupported output format")

    def print_chunk_stats(self, chunks: List[Dict[str, Any]]):
        """打印分块统计信息"""
        if not chunks:
            print("No chunks generated.")
            return
            
        chapter_chunk_counts = {}
        #section_types = {}
        
        for chunk in chunks:
            title = chunk["metadata"]["chapter_title"]
            #section_type = chunk["metadata"].get("section_type", "unknown")
            
            chapter_chunk_counts[title] = chapter_chunk_counts.get(title, 0) + 1
            #section_types[section_type] = section_types.get(section_type, 0) + 1
        
        print(f"总文档块: {len(chunks)}")
        print("各章节文档块数量:")
        for title, count in list(chapter_chunk_counts.items())[:10]:
            print(f"  {title[:50]}...: {count} 块")
        if len(chapter_chunk_counts) > 10:
            print(f"  ... 还有 {len(chapter_chunk_counts) - 10} 个章节")
            
        #print("元素类型分布:")
        #for section_type, count in section_types.items():
        #    print(f"  {section_type}: {count} 块")

@dataclass
class TextBlock:
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    x0: float
    y0: float
    x1: float
    y1: float
    center_x: float
    center_y: float
    width: float
    height: float
    is_spanning: bool = False  # 是否跨栏
    row_group: int = -1  # 所属行组


class SmartPDFTextExtractor:
    def __init__(self):
        self.spanning_threshold = 0.7  # 宽度超过页面70%认为是跨栏元素
        self.y_tolerance = 15  # 行分组的y坐标容差
        self.column_gap_threshold = 0.1  # 栏间距阈值

    def extract_text_blocks_with_position(self, page: fitz.Page) -> List[TextBlock]:
        """提取文本块及其位置信息"""
        text_blocks = page.get_text("dict")["blocks"]
        blocks = []
        
        for block in text_blocks:
            if "lines" not in block:
                continue
                
            text_lines = []
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                if line_text.strip():
                    text_lines.append(line_text.strip())
                    
            if text_lines:
                bbox = block["bbox"]
                text_block = TextBlock(
                    text="\n".join(text_lines),
                    bbox=bbox,
                    x0=bbox[0],
                    y0=bbox[1],
                    x1=bbox[2],
                    y1=bbox[3],
                    center_x=(bbox[0] + bbox[2]) / 2,
                    center_y=(bbox[1] + bbox[3]) / 2,
                    width=bbox[2] - bbox[0],
                    height=bbox[3] - bbox[1]
                )
                blocks.append(text_block)
                
        return blocks

    def detect_page_layout(self, blocks: List[TextBlock], page_width: float) -> Dict:
        """分析页面布局特征"""
        if not blocks:
            return {"type": "single", "column_boundary": page_width / 2}
        
        # 识别跨栏元素
        for block in blocks:
            if block.width > page_width * self.spanning_threshold:
                block.is_spanning = True
        
        # 分析非跨栏元素的分布来确定栏边界
        non_spanning_blocks = [b for b in blocks if not b.is_spanning]
        
        if len(non_spanning_blocks) < 4:
            return {"type": "single", "column_boundary": page_width / 2}
        
        # 使用聚类方法找到栏边界
        x_centers = [b.center_x for b in non_spanning_blocks]
        x_centers.sort()
        
        # 寻找最大的间隙
        max_gap = 0
        best_boundary = page_width / 2
        
        for i in range(len(x_centers) - 1):
            gap = x_centers[i + 1] - x_centers[i]
            if gap > max_gap:
                max_gap = gap
                best_boundary = (x_centers[i] + x_centers[i + 1]) / 2
        
        # 如果最大间隙足够大，认为是双栏布局
        if max_gap > page_width * self.column_gap_threshold:
            return {"type": "double", "column_boundary": best_boundary}
        else:
            return {"type": "single", "column_boundary": page_width / 2}

    def group_blocks_by_rows(self, blocks: List[TextBlock]) -> List[List[TextBlock]]:
        """将文本块按行分组"""
        if not blocks:
            return []
        
        # 按y坐标排序
        sorted_blocks = sorted(blocks, key=lambda x: x.y0)
        
        rows = []
        current_row = [sorted_blocks[0]]
        current_row[0].row_group = 0
        
        for i, block in enumerate(sorted_blocks[1:], 1):
            # 检查是否与当前行重叠
            current_row_y_min = min(b.y0 for b in current_row)
            current_row_y_max = max(b.y1 for b in current_row)
            
            # 如果新块与当前行有垂直重叠，加入当前行
            if (block.y0 <= current_row_y_max and 
                block.y1 >= current_row_y_min) or \
               (abs(block.y0 - current_row_y_max) <= self.y_tolerance):
                current_row.append(block)
                block.row_group = len(rows)
            else:
                # 开始新行
                rows.append(current_row)
                current_row = [block]
                block.row_group = len(rows)
        
        rows.append(current_row)
        return rows

    def sort_row_blocks(self, row_blocks: List[TextBlock], layout_info: Dict) -> List[TextBlock]:
        """对单行内的文本块排序"""
        if len(row_blocks) <= 1:
            return row_blocks
        
        # 检查是否有跨栏元素
        spanning_blocks = [b for b in row_blocks if b.is_spanning]
        non_spanning_blocks = [b for b in row_blocks if not b.is_spanning]
        
        if spanning_blocks:
            # 有跨栏元素的情况：跨栏元素优先，然后按位置排序其他元素
            spanning_sorted = sorted(spanning_blocks, key=lambda x: (x.y0, x.x0))
            
            if non_spanning_blocks:
                column_boundary = layout_info["column_boundary"]
                left_blocks = [b for b in non_spanning_blocks if b.center_x < column_boundary]
                right_blocks = [b for b in non_spanning_blocks if b.center_x >= column_boundary]
                
                left_sorted = sorted(left_blocks, key=lambda x: (x.y0, x.x0))
                right_sorted = sorted(right_blocks, key=lambda x: (x.y0, x.x0))
                
                # 根据跨栏元素的位置决定顺序
                spanning_y = spanning_sorted[0].center_y if spanning_sorted else float('inf')
                non_spanning_y = min([b.center_y for b in non_spanning_blocks]) if non_spanning_blocks else float('inf')
                
                if spanning_y < non_spanning_y:
                    # 跨栏元素在上方：跨栏 → 左栏 → 右栏
                    return spanning_sorted + left_sorted + right_sorted
                else:
                    # 跨栏元素在下方：左栏 → 右栏 → 跨栏
                    return left_sorted + right_sorted + spanning_sorted
            else:
                return spanning_sorted
        else:
            # 没有跨栏元素：按列分组排序
            if layout_info["type"] == "double":
                column_boundary = layout_info["column_boundary"]
                left_blocks = [b for b in row_blocks if b.center_x < column_boundary]
                right_blocks = [b for b in row_blocks if b.center_x >= column_boundary]
                
                left_sorted = sorted(left_blocks, key=lambda x: (x.y0, x.x0))
                right_sorted = sorted(right_blocks, key=lambda x: (x.y0, x.x0))
                
                return left_sorted + right_sorted
            else:
                # 单栏布局：直接按x坐标排序
                return sorted(row_blocks, key=lambda x: x.x0)

    def smart_sort_blocks(self, blocks: List[TextBlock], page_width: float) -> List[TextBlock]:
        """智能文本块排序主函数"""
        if not blocks:
            return blocks
        
        # 分析页面布局
        layout_info = self.detect_page_layout(blocks, page_width)
        
        # 按行分组
        rows = self.group_blocks_by_rows(blocks)
        
        # 对每行内部排序
        sorted_blocks = []
        for row in rows:
            sorted_row = self.sort_row_blocks(row, layout_info)
            sorted_blocks.extend(sorted_row)
        
        return sorted_blocks

    def extract_ordered_text(self, page: fitz.Page) -> str:
        """提取按正确阅读顺序排列的文本"""
        # 提取文本块
        blocks = self.extract_text_blocks_with_position(page)
        
        # 获取页面宽度
        page_width = page.rect.width
        
        # 智能排序
        sorted_blocks = self.smart_sort_blocks(blocks, page_width)
        
        # 合并文本，并在行间添加适当的分隔
        result_text = []
        last_row_group = -1
        
        for block in sorted_blocks:
            # 如果是新的行组，添加段落分隔
            if block.row_group != last_row_group and last_row_group != -1:
                result_text.append("\n")
            
            result_text.append(block.text)
            last_row_group = block.row_group
        
        return "\n".join(result_text)
    
REFER_TOKENS = ["references", "reference", "bibliography", "参考文献"]

def is_reference_title(title: str) -> bool:
    title_l = title.lower()
    return any(tok in title_l for tok in REFER_TOKENS)

def looks_like_reference_line(line: str) -> bool:
    """
    用简单启发式：含 DOI / 年份(19xx|20xx) + 作者逗号 + 'et al.' 等关键词
    """
    import re
    line_l = line.lower()
    return ("doi" in line_l) or re.search(r'\b(19|20)\d{2}\b', line_l)