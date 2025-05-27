"""
Schema definitions for API requests and responses following OpenAI's API standard.
"""

import re
from typing_extensions import Literal
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Optional, Union, Any, ClassVar

# Precompile regex patterns for better performance
UNICODE_BOX_PATTERN = re.compile(r'\\u25[0-9a-fA-F]{2}')

# Configuration
class Config:
    """
    Configuration class holding the default model names for different types of requests.
    """
    TEXT_MODEL = "gpt-4-turbo"          # Default model for text-based chat completions
    VISION_MODEL = "gpt-4-vision-preview"  # Model used for vision-based requests
    EMBEDDING_MODEL = "text-embedding-ada-002"  # Model used for generating embeddings

# Common models used in both streaming and non-streaming contexts
class ImageUrl(BaseModel):
    """
    Represents an image URL in a message.
    """
    url: str = Field(..., description="URL of the image")

    @validator("url")
    def validate_url(cls, v: str) -> str:
        """Validate that the URL is properly formatted."""
        if not v.startswith(("http://", "https://", "data:")):
            raise ValueError("URL must start with http://, https://, or data:")
        return v

class VisionContentItem(BaseModel):
    """
    Represents a single content item in a message (text or image).
    """
    type: Literal["text", "image_url"] = Field(..., description="Type of content")
    text: Optional[str] = Field(None, description="Text content if type is text")
    image_url: Optional[ImageUrl] = Field(None, description="Image URL if type is image_url")
        

class FunctionCall(BaseModel):
    """
    Represents a function call in a message.
    """
    arguments: str = Field(..., description="JSON string of function arguments")
    name: str = Field(..., description="Name of the function to call")

    @validator("arguments")
    def validate_arguments(cls, v: str) -> str:
        """Validate that arguments is a valid JSON string."""
        try:
            import json
            json.loads(v)
            return v
        except json.JSONDecodeError:
            raise ValueError("arguments must be a valid JSON string")

class ChatCompletionMessageToolCall(BaseModel):
    """
    Represents a tool call in a message.
    """
    id: str = Field(..., description="Unique identifier for the tool call")
    function: FunctionCall = Field(..., description="Function call details")
    type: Literal["function"] = Field("function", description="Type of tool call")

class Message(BaseModel):
    """
    Represents a message in a chat completion.
    """
    content: Optional[Union[str, List[VisionContentItem]]] = Field(None, description="Message content")
    refusal: Optional[str] = Field(None, description="Refusal message if any")
    role: Literal["system", "user", "assistant", "tool"] = Field(..., description="Role of the message sender")
    function_call: Optional[FunctionCall] = Field(None, description="Function call if any")
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = Field(None, description="Tool calls if any")
    
# Common request base for both streaming and non-streaming
class ChatCompletionRequestBase(BaseModel):
    """
    Base model for chat completion requests.
    """
    model: str = Field(Config.TEXT_MODEL, description="Model to use for completion")
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Available tools for the model")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Tool choice configuration")
    max_tokens: Optional[int] = Field(4096, ge=1, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="Sampling temperature")
    top_p: Optional[float] = Field(0.8, ge=0, le=1, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(20, ge=1, le=100, description="Top-k sampling parameter")
    min_p: Optional[float] = Field(0.0, ge=0, le=1, description="Minimum probability parameter")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2, le=2, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(1.5, ge=-2, le=2, description="Presence penalty")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    seed: Optional[int] = Field(0, description="Random seed for generation")
    
    @validator("messages")
    def check_messages_not_empty(cls, v: List[Message]) -> List[Message]:
        """Ensure that the messages list is not empty and validate message structure."""
        if not v:
            raise ValueError("messages cannot be empty")
        
        if len(v) > 100:  # OpenAI's limit is typically around 100 messages
            raise ValueError("message history too long")
            
        valid_roles = {"user", "assistant", "system", "tool"}
        for msg in v:
            if msg.role not in valid_roles:
                raise ValueError(f"invalid role: {msg.role}")
                
        return v

    def is_vision_request(self) -> bool:
        """Check if the request includes image content, indicating a vision-based request."""
        import logging
        logger = logging.getLogger(__name__)
        
        for message in self.messages:
            if isinstance(message.content, list):
                for item in message.content:
                    if item.type == "image_url":
                        logger.debug(f"Detected vision request with image: {item.image_url.url[:30]}...")
                        return True
        
        logger.debug("No images detected, treating as text-only request")
        return False
    
class ChatTemplateKwargs(BaseModel):
    """
    Represents the arguments for a chat template.
    """
    enable_thinking: bool = Field(False, description="Whether to enable thinking mode")
    add_generation_prompt: bool = Field(True, description="Whether to add a generation prompt to the response")

# Non-streaming request and response
class ChatCompletionRequest(ChatCompletionRequestBase):
    """
    Model for non-streaming chat completion requests.
    """
    stream: bool = Field(False, description="Whether to stream the response")
    chat_template_kwargs: Optional[ChatTemplateKwargs] = Field(ChatTemplateKwargs(), description="Chat template kwargs")

    def fix_messages(self) -> None:
        """Fix the messages list to ensure proper formatting and ordering."""
        def clean_special_box_text(input_text: str) -> str:
            return UNICODE_BOX_PATTERN.sub('', input_text).strip()
        
        # Clean message contents
        for message in self.messages:
            if message.content is None:
                message.content = ""
            elif isinstance(message.content, str):
                message.content = clean_special_box_text(message.content)
            elif isinstance(message.content, list):
                for item in message.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        item["text"] = clean_special_box_text(item.get("text", ""))
        
        # Sort messages by role
        system_messages = []
        non_system_messages = []
        
        for message in self.messages:
            if message.role == "system":
                system_messages.append(message)
            else:
                non_system_messages.append(message)
            
        self.messages = system_messages + non_system_messages

class Choice(BaseModel):
    """
    Represents a choice in a chat completion response.
    """
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] = Field(..., description="Reason for completion")
    index: int = Field(..., ge=0, description="Index of the choice")
    message: Message = Field(..., description="Generated message")

class ChatCompletionResponse(BaseModel):
    """
    Represents a complete chat completion response.
    """
    id: str = Field(..., description="Unique identifier for the completion")
    object: Literal["chat.completion"] = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[Choice] = Field(..., description="Generated choices")

# Embedding models
class EmbeddingRequest(BaseModel):
    """
    Model for embedding requests.
    """
    model: str = Field(Config.EMBEDDING_MODEL, description="Model to use for embedding")
    input: List[str] = Field(..., min_items=1, description="List of text inputs for embedding")
    image_url: Optional[str] = Field(None, description="Image URL to embed")

    @validator("input")
    def validate_input(cls, v: List[str]) -> List[str]:
        """Validate that input texts are not empty."""
        if not all(text.strip() for text in v):
            raise ValueError("Input texts cannot be empty")
        return v

class Embedding(BaseModel):
    """
    Represents an embedding object in an embedding response.
    """
    embedding: List[float] = Field(..., description="The embedding vector")
    index: int = Field(..., ge=0, description="The index of the embedding in the list")
    object: str = Field("embedding", description="The object type")

class EmbeddingResponse(BaseModel):
    """
    Represents an embedding response.
    """
    object: str = Field("list", description="Object type")
    data: List[Embedding] = Field(..., description="List of embeddings")
    model: str = Field(..., description="Model used for embedding")