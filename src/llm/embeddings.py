"""Qwen embeddings client for semantic search."""
from openai import OpenAI
from typing import List

from ..config import get_settings


class QwenEmbeddings:
    """Client for Qwen text embeddings.
    
    Used for:
    - Generating query embeddings for vector search
    - Both single and batch embedding generation
    """
    
    def __init__(self):
        settings = get_settings()
        api_key = settings.lapathon_api_key
        
        # Validate API key is set
        if not api_key or api_key == "":
            raise ValueError(
                "LAPATHON_API_KEY is not set. Please create a .env file with your API key. "
                "Example: LAPATHON_API_KEY=your_key_here"
            )
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=settings.llm_base_url
        )
        self.model = settings.embedding_model
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
            encoding_format="float"
        )
        return response.data[0].embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
            encoding_format="float"
        )
        return [item.embedding for item in response.data]
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension by running a test embed."""
        test_embed = self.embed("test")
        return len(test_embed)
