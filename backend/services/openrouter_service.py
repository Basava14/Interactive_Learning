import os
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


class OpenRouterService:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.model = "mistralai/mistral-small-3.2-24b-instruct:free"
    
    def generate_image_summary(self, image_name: str, image_type: str = "general") -> str:
        """
        Generate an educational summary about the uploaded image.
        Since we're using a text-only model, we base the summary on image filename/context.
        """
        prompt = f"""You are an educational assistant helping students learn through interactive 3D visualization.

A student has uploaded an image named "{image_name}" for 3D conversion and learning.

Provide a concise, educational summary (2-5 sentences) about this subject that would help a student understand the key concepts. Focus on:
- Main purpose/function
- Key structural features
- Educational significance

Keep it clear, engaging, and educational. Do not mention the image itself, just provide factual educational content about the subject."""

        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://2d-to-3d-converter.app",
                    "X-Title": "2D-to-3D Interactive Learning",
                },
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Unable to generate summary: {str(e)}"
    
    def chat_about_image(self, image_name: str, conversation_history: List[Dict[str, str]], user_message: str) -> str:
        """
        Handle chat conversation about the uploaded image with context awareness.
        """
        # Build conversation context
        system_prompt = f"""You are an educational AI assistant helping a student learn about "{image_name}".
The student is viewing a 3D model of this subject and wants to learn more through conversation.

Provide clear, concise answers (2-10 sentences max) that are:
- Educational and factually accurate
- Appropriate for learning purposes
- Engaging and easy to understand
- Focused on the subject matter

Always stay on topic and help the student learn effectively."""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in conversation_history:
            messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://2d-to-3d-converter.app",
                    "X-Title": "2D-to-3D Interactive Learning",
                },
                model=self.model,
                messages=messages,
                max_tokens=300,
                temperature=0.7,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Unable to process your question: {str(e)}"

# Singleton instance
openrouter_service = OpenRouterService()
