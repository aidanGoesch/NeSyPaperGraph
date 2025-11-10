# File for various LLM related services
import boto3
import json
import os
import re
import ast
from botocore.config import Config
from transformers import pipeline
from openai import OpenAI
from services.topic_extraction_prompts import build_topic_extraction_prompt, get_topic_reuse_additional_instructions

class LLMClient:
    def __init__(self, model_id, region='us-east-1'):
        self.model_id = model_id
        config = Config(retries={'max_attempts': 3, 'mode': 'adaptive'})
        
        # Use API key if available
        if os.getenv('AWS_BEARER_TOKEN_BEDROCK'):
            self.bedrock = boto3.client(
                'bedrock-runtime', 
                region_name=region, 
                config=config,
                aws_access_key_id='',
                aws_secret_access_key='',
                aws_session_token=os.getenv('AWS_BEARER_TOKEN_BEDROCK')
            )
        else:
            self.bedrock = boto3.client('bedrock-runtime', region_name=region, config=config)

    def generate(self, prompt):
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        })
        
        response = self.bedrock.invoke_model(
            body=body,
            modelId=self.model_id,
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response.get('body').read())
        return response_body['content'][0]['text']


class HuggingFaceLLMClient:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        self.generator = pipeline("text-generation", model=model_name, max_length=512)

    def generate(self, prompt):
        result = self.generator(prompt, max_new_tokens=2000, do_sample=True, temperature=0.7)
        return result[0]['generated_text'][len(prompt):].strip()


class OpenAILLMClient:
    def __init__(self, api_key=None, assistant_id=None):
        """
        Initialize OpenAI client with Assistant support only.
        
        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY environment variable.
            assistant_id: OpenAI Assistant ID. If None, will use OPENAI_ASSISTANT_ID environment variable.
        """
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.assistant_id = assistant_id or os.getenv('OPENAI_ASSISTANT_ID')
        if not self.assistant_id:
            raise ValueError("OpenAI Assistant ID is required. Set OPENAI_ASSISTANT_ID environment variable or pass assistant_id parameter.")
        
        # Use Assistants API v2 (v1 is deprecated)
        # Set default headers with v2 beta header
        self.client = OpenAI(
            api_key=api_key,
            default_headers={"OpenAI-Beta": "assistants=v2"}
        )

    def generate(self, prompt, additional_instructions=None):
        """
        Generate text using OpenAI Assistants API.
        
        Args:
            prompt: The prompt text
            additional_instructions: Optional additional instructions to pass to the assistant run
            
        Returns:
            str: The generated text response
        """
        return self._generate_with_assistant(prompt, additional_instructions)
    
    def _generate_with_assistant(self, prompt, additional_instructions=None):
        """
        Generate text using OpenAI Assistants API.
        
        Args:
            prompt: The prompt text
            
        Returns:
            str: The generated text response
        """
        import time
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Create a thread (v2 header is set in client default_headers)
        thread = self.client.beta.threads.create()
        
        try:
            # Add message to thread
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )
            
            # Run the assistant with additional instructions if provided
            run_params = {
                "thread_id": thread.id,
                "assistant_id": self.assistant_id
            }
            if additional_instructions:
                run_params["additional_instructions"] = additional_instructions
            
            run = self.client.beta.threads.runs.create(**run_params)
            
            # Poll for completion with timeout
            max_wait_time = 180  # 3 minutes max per request
            start_time = time.time()
            
            while run.status in ['queued', 'in_progress', 'cancelling']:
                elapsed = time.time() - start_time
                if elapsed > max_wait_time:
                    raise TimeoutError(f"Assistant run timed out after {max_wait_time} seconds")
                
                time.sleep(1.0)
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
            
            # Check if run requires action (tool calls, etc.)
            if run.status == 'requires_action':
                logger.warning(f"Assistant run requires action: {run.required_action}")
                raise RuntimeError(f"Assistant run requires action: {run.required_action}")
            
            if run.status == 'completed':
                # Retrieve the messages
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread.id,
                    order="asc"  # Get messages in chronological order
                )
                
                # Get the assistant's response (last message should be assistant's response)
                assistant_messages = [msg for msg in messages.data if msg.role == 'assistant']
                if not assistant_messages:
                    raise ValueError("No assistant response found in thread messages")
                
                # Get the most recent assistant message
                latest_message = assistant_messages[-1]
                
                # Extract text content from the message
                if latest_message.content:
                    text_parts = []
                    for content_item in latest_message.content:
                        if content_item.type == 'text':
                            text_parts.append(content_item.text.value)
                        elif content_item.type == 'tool_use':
                            # Assistant is using tools - this shouldn't happen for topic extraction
                            pass
                    
                    if text_parts:
                        return "\n".join(text_parts)
                
                raise ValueError(f"Assistant response has no text content. Content types: {[c.type for c in latest_message.content]}")
            elif run.status == 'failed':
                error_info = run.last_error if run.last_error else "Unknown error"
                raise RuntimeError(f"Assistant run failed: {error_info}")
            else:
                raise RuntimeError(f"Assistant run ended with unexpected status: {run.status}. Error: {run.last_error}")
        
        finally:
            # Clean up: delete the thread (optional, but good practice)
            try:
                self.client.beta.threads.delete(thread.id)
            except:
                pass  # Ignore cleanup errors


class TopicExtractor:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def extract_topics(self, text, current_topics=None):
        # Build user message prompt (simplified since assistant has main instructions)
        prompt = build_topic_extraction_prompt(text, current_topics)
        
        # Get additional instructions focused on topic reuse
        additional_instructions = get_topic_reuse_additional_instructions(current_topics)
        
        # Generate response with additional instructions
        response = self.llm_client.generate(prompt, additional_instructions=additional_instructions).strip()
        
        import logging
        logger = logging.getLogger(__name__)

        # Try to parse JSON response
        try:
            # First, try to extract JSON from the response (in case there's extra text)
            # Look for JSON object pattern
            json_match = re.search(r'\{[^{}]*"topics"[^{}]*\[[^\]]*\][^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Try to find any JSON object
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response
            
            # Parse the JSON
            data = json.loads(json_str)
            
            # Extract topics from the JSON
            if isinstance(data, dict) and "topics" in data:
                topics = data["topics"]
            elif isinstance(data, list):
                # If it's just a list, use it directly
                topics = data
            else:
                logger.warning(f"⚠️ Unexpected JSON structure: {data}")
                return []
            
            # Validate and clean topics
            if isinstance(topics, list):
                cleaned_topics = [str(t).strip() for t in topics if t]
                return cleaned_topics
            else:
                logger.warning(f"⚠️ Topics is not a list: {type(topics)}")
                return []
                
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Failed to parse JSON: {e}. Response: {response[:1000]}")
            return []
        except Exception as e:
            logger.warning(f"⚠️ Error parsing response: {e}. Response: {response[:1000]}")
            return []



if __name__ == "__main__":
    # Example usage with AWS Bedrock:
    # model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    # bedrock_client = LLMClient(model_id=model_id)
    
    # Example usage with HuggingFace:
    hf_client = HuggingFaceLLMClient(model_name="Qwen/Qwen2.5-0.5B")
    
    # Both clients work interchangeably with TopicExtractor
    topic_extractor = TopicExtractor(hf_client)  # or bedrock_client

    sample_text = "Artificial Intelligence and Machine Learning are transforming the tech industry. Cloud computing provides scalable resources for AI applications."
    extracted_topics = topic_extractor.extract_topics(sample_text, current_topics=["Artificial Intelligence", "Cloud Computing", "Data Science"])
    print("Extracted Topics:", extracted_topics)