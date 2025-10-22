# File for various LLM related services
import boto3
import json
import os
import re
import ast
from botocore.config import Config
from transformers import pipeline

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


class TopicExtractor:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def extract_topics(self, text, current_topics=None):
        prompt = f"""
    You are a precise text classifier. Your task is to extract the main topics from the given text.

    Follow these strict rules:
    1. Use topics from the existing list if they are relevant.
    2. Add new topics only if necessary, using 1–3 word noun phrases.
    3. Respond with **only** a valid Python list literal (e.g. ['LLMs', 'Reinforcement Learning']).
    4. Do **not** include any commentary, explanation, or text outside the list.
    5. The response must start with '[' and end with ']'. Nothing else is allowed.

    Example of a valid output:
    ['Machine Learning', 'Bayesian Models', 'Cognitive Science']

    Existing Topics: {current_topics}

    Text:
    {text}

    Output only the Python list:
    """

        response = self.llm_client.generate(prompt).strip()

        # Extract first list-looking structure
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if not match:
            # fallback: if model ignored format, return empty
            return []

        list_str = match.group(0)

        # Safely parse the Python list
        try:
            topics = ast.literal_eval(list_str)
            if isinstance(topics, list):
                return [t.strip() for t in topics if isinstance(t, str)]
        except Exception:
            pass

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