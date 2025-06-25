import openai
from typing import List, Dict, Any
import logging
import requests
import json
import os

class LLMClient:
    """LLM客户端"""
    
    def __init__(self, rag_config: 'RAGConfig'):
        self.config = rag_config.llm
        self.logger = logging.getLogger(__name__)
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化LLM客户端"""
        if self.config.LLM_TYPE == "openai":
            self._init_openai_client()
        elif self.config.LLM_TYPE == "ollama":
            self._init_ollama_client()
        else:
            self.logger.warning(f"不支持的LLM提供商: {self.config.LLM_TYPE}")
    
    def _init_openai_client(self):
        """初始化OpenAI客户端"""
        try:
            self.client = openai.OpenAI(
                api_key=self.config.API_KEY or os.getenv("OPENAI_API_KEY"),
                base_url=self.config.API_BASE or None
            )
            self.logger.info("OpenAI客户端初始化成功")
        except Exception as e:
            self.logger.error(f"OpenAI客户端初始化失败: {e}")
            raise
    
    #def _init_ollama_client(self):
    #    """初始化Ollama客户端"""
    #    self.base_url = self.config.API_BASE or "http://localhost:11434"
    #    self.logger.info("Ollama客户端初始化成功")
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """生成回复"""
        if self.config.LLM_TYPE == "openai":
            return self._openai_generate(messages)
        elif self.config.LLM_TYPE == "ollama":
            return self._ollama_generate(messages)
        else:
            raise ValueError(f"不支持的LLM提供商: {self.config.LLM_TYPE}")
    
    def _openai_generate(self, messages: List[Dict[str, str]]) -> str:
        """OpenAI生成"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=messages,
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS
            )
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI生成失败: {e}")
            raise
    
    def _ollama_generate(self, messages: List[Dict[str, str]]) -> str:
        """Ollama生成"""
        try:
            prompt = self._convert_messages_to_prompt(messages)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.TEMPERATURE,
                        "num_predict": self.config.MAX_TOKENS
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise Exception(f"Ollama API错误: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Ollama生成失败: {e}")
            raise
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """将消息转换为提示词"""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts) + "\n\nAssistant: "