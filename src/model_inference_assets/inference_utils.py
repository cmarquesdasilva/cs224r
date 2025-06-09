"""
Model inference service for persona generation.

This module provides functions for making inference requests to language models
and handling responses.
"""

import os
from typing import Dict, List, Any, Optional, Union
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Import helper functions
from .helpers import extract_json_from_response

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def call_openai_direct(
    system_prompt: str, 
    user_prompt: str, 
    model_name: str = "gpt-4o"
) -> str:
    """
    Call OpenAI model directly using the OpenAI Python SDK.
    
    Args:
        system_prompt: The system prompt
        user_prompt: The user prompt
        model_name: OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")
        
    Returns:
        The model's response as a string
    """
    try:
        # Initialize the client
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Create the messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call the model
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,  # Use deterministic output for evaluation
            max_tokens=4096
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error calling OpenAI directly: {str(e)}")
        raise


def call_together_model(
    prompt: str, 
    temperature: float = 0.7, 
    max_tokens: int = 2048
) -> str:
    """
    Calls the Together AI model with the given prompt, using system/user roles.
    
    Args:
        prompt: The prompt to send to the model
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 2048)
        
    Returns:
        The model's response as a string
    """
    try:
        # Import here to avoid making it a required dependency
        from langchain_together import ChatTogether
        
        messages = [
            {
                "role": "system",
                "content": "You are a system that generates specifications for realistic simulations of people. You follow the generation rules and constraints carefully."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        formatted_prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
        
        chat = ChatTogether(
            model=os.getenv("MODEL"),
            together_api_key=os.getenv("TOGETHER_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens
        )
        response = chat.invoke(formatted_prompt)
        return response
    
    except Exception as e:
        logger.error(f"Error calling Together model: {str(e)}")
        raise


def invoke_chat_model(
    system_prompt: str,
    user_prompt: str,
    provider: str = "openai",
    model_name: Optional[str] = None,
    **kwargs
) -> Union[Dict[str, Any], str]:
    """
    Send a system / user prompt pair to a chat-completion model (OpenAI or Together AI)
    and return either the raw reply or the reply parsed as JSON.

    Parameters
    ----------
    system_prompt : str
        Content for the assistant’s “system” role.
    user_prompt : str
        Content for the “user” role (or full prompt for Together AI).
    provider : {"openai", "together"}, default "openai"
        Which backend to call.
    model_name : str, optional
        Concrete model or deployment name; falls back to sensible defaults per provider.
    extract_json : bool, default True
        If True, try to parse the reply as JSON via `extract_json_from_response`.
        When False, return the raw text.
    **kwargs
        Extra keyword arguments passed to the provider implementation
        (e.g. temperature, max_tokens for Together).

    Returns
    -------
    dict | str
        Parsed JSON (when extract_json=True and parsing succeeds) or raw reply string.
    """
    # Default model names
    default_models = {
        "openai": "gpt-4o",
        "together": os.getenv("MODEL", "Qwen/Qwen1.5-72B-Chat")
    }
    
    # Use default model if none specified
    if model_name is None:
        model_name = default_models.get(provider, default_models["openai"])
    
    # Call the appropriate model
    if provider.lower() == "openai":
        response = call_openai_direct(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_name=model_name
        )

    elif provider.lower() == "together":
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)
        response = call_together_model(
            prompt=user_prompt,  # Together model formats the prompt differently
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return response