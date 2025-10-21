# File for various LLM related services
import boto3
import json
import os
from botocore.config import Config

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


class TopicExtractor:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def extract_topics(self, text, current_topics=None):
        prompt = f"Extract the main topics from the following text. If the topics already exist in the list of existing topics, please use those, rather than adding new topics. Please give the list of topics for the following text in the form of a python list (for example ['LLMs', 'Reinforcement Learning', 'Big Data']) \nExisting Topics: {current_topics}\n\nText:\n\n{text}\n\nTopics:"
        response = self.llm_client.generate(prompt)
        topics = response.strip().split('\n')
        return [topic.strip() for topic in topics if topic.strip()]


if __name__ == "__main__":
    # Example usage:
    model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    llm_client = LLMClient(model_id=model_id)
    topic_extractor = TopicExtractor(llm_client)

    sample_text = "Artificial Intelligence and Machine Learning are transforming the tech industry. Cloud computing provides scalable resources for AI applications."
    extracted_topics = topic_extractor.extract_topics(sample_text, current_topics=["Artificial Intelligence", "Cloud Computing", "Data Science"])
    print("Extracted Topics:", extracted_topics)