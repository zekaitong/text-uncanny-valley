from abc import ABC, abstractmethod
import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def get_response(self, system_prompt: str, user_message: str, model: str) -> str:
        """Get a response from the LLM model"""
        pass
    
    @classmethod
    def get_provider(cls, provider_name: str) -> 'LLMProvider':
        """Factory method to get the appropriate provider"""
        providers = {
            'openai': OpenAIProvider,
            'togetherai': TogetherAIProvider,
            'xai': XAIProvider,
            'anthropic': AnthropicProvider,
            'cohere': CohereProvider,
            'google': GoogleProvider,
            'custom': CustomProvider
        }
        
        if provider_name not in providers:
            raise ValueError(f"Provider '{provider_name}' not supported. Available providers: {', '.join(providers.keys())}")
        
        return providers[provider_name]()
    

class CustomProvider(LLMProvider):
    """User-specific implementation"""
    
    def __init__(self):
        # IMPLEMENT YOUR OWN INIT FUNCTION
        pass
    
    def get_response(self, system_prompt: str, user_message: str, model: str, thinking:bool=False) -> str:
        # WRITE YOUR OWN get_response FUNCTION
        pass



class OpenAIProvider(LLMProvider):
    """OpenAI-specific implementation"""
    
    def __init__(self):
        # API key is loaded from environment variable OPENAI_API_KEY by the openai library
        pass
    
    # TODO: implement thinking
    def get_response(self, system_prompt: str, user_message: str, model: str, thinking:bool=False, reasoning_effort: str=None) -> str:
        import openai
        """Get a response from OpenAI model"""
        thinking_tokens = 0

        # Check if this is a GPT-5 model with reasoning effort set
        is_gpt5 = model.startswith("gpt-5")
        use_responses_api = is_gpt5 and reasoning_effort is not None

        if use_responses_api:
            # Use the new responses API for GPT-5 with reasoning
            client = openai.OpenAI()

            # Combine system prompt and user message into a single input
            combined_input = f"{system_prompt}\n\n{user_message}"

            response = client.responses.create(
                model=model,
                input=combined_input,
                reasoning={"effort": reasoning_effort, "summary": "auto"}
            )

            if thinking:
                thinking_tokens = response.usage.output_tokens_details.reasoning_tokens
                thinking_summary = response.reasoning.summary

            return [response.output_text, thinking_tokens, thinking_summary]
        else:
            # Use standard chat completions API
            call_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
            }

            response = openai.chat.completions.create(**call_params)

            if thinking:
                thinking_tokens = response.usage.completion_tokens_details.reasoning_tokens
            
            thinking_summary = ''

            return [response.choices[0].message.content, thinking_tokens, thinking_summary]
    

class XAIProvider(LLMProvider):
    """XAI implementation"""

    def __init__(self):
        import openai
        # initialize the client. api key is accessed through the environment variable XAI_API_KEY
        self.client = openai.OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )

    def get_response(self, system_prompt: str, user_message: str, model: str, thinking:bool=False) -> str:
        """Get a response through the XAI api"""
        thinking_tokens = 0
        response = self.client.chat.completions.create(
            model=model,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
        )
        if thinking:
            thinking_tokens = response.usage.completion_tokens_details.reasoning_tokens
            thinking_summary = response.reasoning.summary
        else:
            thinking_summary = ''

        return [response.choices[0].message.content, thinking_tokens, thinking_summary]


class TogetherAIProvider(LLMProvider):
    """TogetherAI-specific implementation"""

    def __init__(self):
        from together import Together
        # API key is loaded from environment variable TOGETHER_API_KEY by the togetherai library
        self.client = Together()

    def get_response(self, system_prompt: str, user_message:str, model: str, thinking: bool=False) -> str:
        """Get a response through TogetherAI api"""
        response = self.client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        )
        response_text = response.choices[0].message.content

        # Extract thinking/reasoning if present
        thinking_summary = ""
        thinking_tokens = 0

        if "<think>" in response_text and "</think>" in response_text:
            # Extract the thinking content between tags
            think_start = response_text.index("<think>") + 7
            think_end = response_text.index("</think>")
            thinking_summary = response_text[think_start:think_end]

            # Extract final response after thinking tags
            final_response = response_text[think_end + 8:]

            # Count thinking tokens
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4")
            thinking_tokens = len(encoding.encode(
                thinking_summary,
                disallowed_special=()
            ))
        else:
            # No thinking tags present, use full response
            final_response = response_text

        return [final_response, thinking_tokens, thinking_summary]


class AnthropicProvider(LLMProvider):
    """Anthropic-specific implementation"""
    
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    def get_response(self, system_prompt: str,
                     user_message: str, model: str,
                     thinking: bool=False) -> str:
        """Get a response from Anthropic model"""
        if thinking:
            response = self.client.messages.create(
                model=model,
                stream=True,
                max_tokens=64_000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 55_000,
                },
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                temperature=1
            )

            # Collect thinking and response from stream
            thinking_text = ""
            response_str = ""

            with response as stream:
                for event in stream:
                    if event.type == "content_block_start":
                        if hasattr(event.content_block, 'thinking'):
                            # This is a thinking block
                            pass
                    elif event.type == "content_block_delta":
                        if hasattr(event.delta, 'thinking'):
                            # Accumulate thinking text
                            thinking_text += event.delta.thinking
                        elif hasattr(event.delta, 'text'):
                            # Accumulate response text
                            response_str += event.delta.text

            # Count thinking tokens
            thinking_count = self.client.messages.count_tokens(
                model=model,
                messages=[{
                    "role": "user",
                    "content": thinking_text
                }],
            )
            num_thinking = thinking_count.input_tokens
        else:
            response = self.client.messages.create(
                model=model,
                max_tokens=5000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0
            )
            response_str = response.content[0].text
            num_thinking = 0
            thinking_text = ""

        return [response_str, num_thinking, thinking_text]
    

class GoogleProvider(LLMProvider):
    """Google-specific implementation"""

    def __init__(self):
        from google import genai
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    def get_response(self, system_prompt: str, user_message: str, 
                     model: str, thinking: bool=False) -> str:
        """Get a response from Gemini"""
        from google.genai import types

        if thinking:
            response = self.client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    thinking_config=types.ThinkingConfig(include_thoughts=True)
                ),
                contents=user_message
            )
            model_response = ''
            thinking_summary = ''
            for part in response.candidates[0].content.parts:
                if not part.text:
                    continue
                if part.thought:
                    thinking_summary += part.text
                else:
                    model_response += part.text
            return [model_response, response.usage_metadata.thoughts_token_count, thinking_summary]
        else:
            response = self.client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)),
                contents=user_message
            )
            return [response.text, 0, '']


class CohereProvider(LLMProvider):
    """Cohere-specific implementation"""
    
    def __init__(self):
        try:
            import cohere
            self.client = cohere.Client(os.environ.get("COHERE_API_KEY"))
        except ImportError:
            raise ImportError("The 'cohere' package is not installed. Please install it with 'pip install cohere'")
    
    def get_response(self, system_prompt: str, user_message: str, model: str) -> str:
        """Get a response from Cohere model"""
        # Use chat endpoint for command models, generate for other models
        if model.startswith("command"):
            response = self.client.chat(
                model=model,
                message=user_message,
                preamble=system_prompt,
                temperature=0.0,
            )
            return response.text
        else:
            # Legacy generate API for older models
            response = self.client.generate(
                model=model,
                prompt=f"{system_prompt}\n\n{user_message}",
                max_tokens=1000,
                temperature=0.0,
            )
            return response.generations[0].text