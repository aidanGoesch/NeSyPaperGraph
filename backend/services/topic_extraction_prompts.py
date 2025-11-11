"""
Topic extraction prompt templates for LLM-based topic extraction.
"""


def get_topic_reuse_additional_instructions(current_topics=None):
    """
    Get additional instructions focused on topic reuse for the assistant.
    This is used as additional_instructions when creating a run.
    
    Args:
        current_topics: Set or list of existing topics, or None
        
    Returns:
        str: Additional instructions emphasizing topic reuse
    """
    if current_topics:
        topics_list = sorted(list(current_topics)) if isinstance(current_topics, set) else current_topics
        topics_count = len(topics_list)
        topics_formatted = "\n".join([f"  {i+1}. \"{topic}\"" for i, topic in enumerate(topics_list)])
        
        return f"""🚨 CRITICAL: TOPIC REUSE IS MANDATORY 🚨

You have access to {topics_count} existing topics from previous papers. YOU MUST REUSE THESE TOPICS AGGRESSIVELY.

Existing Topics Database:
{topics_formatted}

**MANDATORY TOPIC REUSE PROCESS:**

1. **FIRST STEP - CHECK EXISTING TOPICS**: Before creating ANY new topics, review the entire existing topics list above. For EACH concept in the paper, check if ANY existing topic matches or is related.

2. **REUSE REQUIREMENT**: 
   - You MUST reuse at least 5-6 existing topics if they are available and relevant (out of 8 total)
   - Only create 1-2 NEW topics if absolutely necessary for concepts not covered by existing topics
   - If an existing topic is even remotely related, REUSE IT instead of creating a new one

3. **FLEXIBILITY IN MATCHING**:
   - Prefer reusing a broader existing topic over creating a narrow new one
   - "Convolutional Neural Networks" → If "Neural Networks" exists, REUSE "Neural Networks"
   - "Deep Reinforcement Learning" → If "Reinforcement Learning" exists, REUSE "Reinforcement Learning"
   - "Image Recognition" → If "Computer Vision" exists, REUSE "Computer Vision"

4. **EXACT NAME MATCHING**: When reusing topics, use the EXACT name from the existing topics list (case-sensitive).

5. **CREATE NEW ONLY AS LAST RESORT**: Only create a new topic if NO existing topic is relevant, even loosely. Before creating new, ask: "Could any existing topic represent this concept?" If yes, reuse it.

**REMEMBER: Consistency across papers is MORE IMPORTANT than perfect topic specificity. Reuse existing topics aggressively - aim to reuse 5-6 out of 8 topics when possible.**"""
    else:
        return """No existing topics available. You may create new topics as needed."""


def build_topic_extraction_prompt(text, current_topics=None):
    """
    Build the user message prompt for topic extraction.
    Since the assistant already has the main instructions, this just provides
    the existing topics list and paper text.
    
    Args:
        text: The paper text to extract topics from
        current_topics: Set or list of existing topics to reuse, or None
        
    Returns:
        str: User message prompt with existing topics and paper text
    """
    if current_topics:
        topics_list = sorted(list(current_topics)) if isinstance(current_topics, set) else current_topics
        topics_count = len(topics_list)
        topics_formatted = "\n".join([f"  {i+1}. \"{topic}\"" for i, topic in enumerate(topics_list)])
        
        return f"""## Existing Topics Database

You have access to a database of {topics_count} previously identified topics from other papers:

{topics_formatted}

**CRITICAL: You MUST prioritize reusing these topics whenever they are relevant to the current paper.**

## Paper Text

{text}

Extract topics following your instructions, prioritizing reuse of the existing topics above."""
    else:
        return f"""## Paper Text

{text}

Extract topics following your instructions."""

