import os
import base64
from openai import OpenAI
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class LLMService:
    def __init__(self, provider: str = "auto"):
        """
        Initialize LLM service with provider selection.
        
        Args:
            provider: "github", "openrouter", or "auto" (tries github first, falls back to openrouter)
        """
        self.provider = provider
        self.client = None
        self.model = None
        self.is_multimodal = False
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider and availability."""
        if self.provider == "auto":
            # Try GitHub first, then OpenRouter
            if self._try_github():
                return
            elif self._try_openrouter():
                return
            else:
                raise Exception("No valid LLM provider available")
        elif self.provider == "github":
            if not self._try_github():
                raise Exception("GitHub provider not available")
        elif self.provider == "openrouter":
            if not self._try_openrouter():
                raise Exception("OpenRouter provider not available")
        else:
            raise ValueError("Invalid provider. Use 'github', 'openrouter', or 'auto'")
    
    def _try_github(self) -> bool:
        """Try to initialize GitHub client."""
        try:
            github_token = os.environ.get("GITHUB_TOKEN")
            if not github_token:
                return False
            
            self.client = OpenAI(
                base_url="https://models.github.ai/inference",
                api_key=github_token,
            )
            self.model = "openai/gpt-4o-mini"
            self.is_multimodal = True
            self.provider = "github"
            print("Using GitHub GPT-4o-mini (multimodal)")
            return True
        except Exception:
            return False
    
    def _try_openrouter(self) -> bool:
        """Try to initialize OpenRouter client."""
        try:
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if not openrouter_key:
                return False
            
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
            )
            self.model = "mistralai/mistral-small-3.2-24b-instruct:free"
            self.is_multimodal = False
            self.provider = "openrouter"
            print("Using OpenRouter Mistral (text-only)")
            return True
        except Exception:
            return False
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for multimodal models."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def generate_image_summary(self, image_name: str, image_path: Optional[str] = None, image_type: str = "general") -> str:
        """
        Generate an educational summary about the uploaded image.
        
        Args:
            image_name: Name/description of the image
            image_path: Path to the actual image file (for multimodal models)
            image_type: Type of image for context
        """
        if self.is_multimodal and image_path and os.path.exists(image_path):
            # Use multimodal capabilities with actual image
            return self._generate_multimodal_summary(image_path, image_name, image_type)
        else:
            # Fallback to text-based summary
            return self._generate_text_based_summary(image_name, image_type)
    
    def _generate_multimodal_summary(self, image_path: str, image_name: str, image_type: str) -> str:
        """Generate summary using multimodal model with actual image."""
        prompt = f"""You are an educational assistant helping students learn through interactive 3D visualization.

A student has uploaded this image (named "{image_name}") for 3D conversion and learning.

Analyze the image and provide a concise, educational summary (2-5 sentences) that would help a student understand the key concepts. Focus on:
- What you can see in the image
- Main purpose/function of the subject
- Key structural features visible
- Educational significance

Keep it clear, engaging, and educational."""

        try:
            # Get image extension for proper MIME type
            file_extension = os.path.splitext(image_path)[1].lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }.get(file_extension, 'image/jpeg')
            
            base64_image = self._encode_image(image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.7,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Multimodal summary failed: {e}")
            # Fallback to text-based summary
            return self._generate_text_based_summary(image_name, image_type)
    
    def _generate_text_based_summary(self, image_name: str, image_type: str) -> str:
        """Generate summary using text-only model based on image name/context."""
        prompt = f"""You are an educational assistant helping students learn through interactive 3D visualization.

A student has uploaded an image named "{image_name}" for 3D conversion and learning.

Provide a concise, educational summary (2-5 sentences) about this subject that would help a student understand the key concepts. Focus on:
- Main purpose/function
- Key structural features
- Educational significance

Keep it clear, engaging, and educational. Do not mention the image itself, just provide factual educational content about the subject."""

        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Add provider-specific headers if using OpenRouter
            extra_headers = {}
            if self.provider == "openrouter":
                extra_headers = {
                    "HTTP-Referer": "https://2d-to-3d-converter.app",
                    "X-Title": "2D-to-3D Interactive Learning",
                }
            
            completion = self.client.chat.completions.create(
                extra_headers=extra_headers,
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.7,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Unable to generate summary: {str(e)}"
    
    def chat_about_image(self, image_name: str, conversation_history: List[Dict[str, str]], 
                        user_message: str, image_path: Optional[str] = None) -> str:
        """
        Handle chat conversation about the uploaded image with context awareness.
        
        Args:
            image_name: Name/description of the image
            conversation_history: Previous conversation messages
            user_message: Current user question
            image_path: Path to the actual image file (for multimodal models)
        """
        if self.is_multimodal and image_path and os.path.exists(image_path):
            return self._chat_multimodal(image_name, conversation_history, user_message, image_path)
        else:
            return self._chat_text_based(image_name, conversation_history, user_message)
    
    def _chat_multimodal(self, image_name: str, conversation_history: List[Dict[str, str]], 
                        user_message: str, image_path: str) -> str:
        """Handle chat using multimodal model with image context."""
        system_prompt = f"""You are an educational AI assistant helping a student learn about the subject shown in the image.
The student is viewing a 3D model of this subject and wants to learn more through conversation.

Provide clear, concise answers (2-10 sentences max) that are:
- Educational and factually accurate
- Appropriate for learning purposes
- Engaging and easy to understand
- Based on what you can see in the image and your knowledge

Always stay on topic and help the student learn effectively."""

        try:
            # Get image extension for proper MIME type
            file_extension = os.path.splitext(image_path)[1].lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }.get(file_extension, 'image/jpeg')
            
            base64_image = self._encode_image(image_path)
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history
            for msg in conversation_history:
                messages.append(msg)
            
            # Add current user message with image
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            })
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=300,
                temperature=0.7,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Multimodal chat failed: {e}")
            # Fallback to text-based chat
            return self._chat_text_based(image_name, conversation_history, user_message)
    
    def _chat_text_based(self, image_name: str, conversation_history: List[Dict[str, str]], 
                        user_message: str) -> str:
        """Handle chat using text-only model."""
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
            # Add provider-specific headers if using OpenRouter
            extra_headers = {}
            if self.provider == "openrouter":
                extra_headers = {
                    "HTTP-Referer": "https://2d-to-3d-converter.app",
                    "X-Title": "2D-to-3D Interactive Learning",
                }
            
            completion = self.client.chat.completions.create(
                extra_headers=extra_headers,
                model=self.model,
                messages=messages,
                max_tokens=300,
                temperature=0.7,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Unable to process your question: {str(e)}"
    
    def get_provider_info(self) -> Dict[str, any]:
        """Get information about the current provider and capabilities."""
        return {
            "provider": self.provider,
            "model": self.model,
            "is_multimodal": self.is_multimodal,
            "client_initialized": self.client is not None
        }


# Factory function for easy instantiation
def create_llm_service(provider: str = "auto") -> LLMService:
    """
    Factory function to create LLM service with specified provider.
    
    Args:
        provider: "github", "openrouter", or "auto"
    
    Returns:
        LLMService instance
    """
    return LLMService(provider=provider)

# Singleton instances for backward compatibility
llm_service = create_llm_service("openrouter")  # Auto-select best available
openrouter_service = llm_service  # Backward compatibility alias


# # Usage examples:
# if __name__ == "__main__":    
#    # Force specific provider
#     try:
#         github_service = create_llm_service("github")
#         print("GitHub service initialized successfully")
#     except Exception as e:
#         print(f"GitHub service failed: {e}")
    
    # Example 3: Generate summary with image (if multimodal)
    # summary = service.generate_image_summary(
    #     image_name="human heart",
    #     image_path="/path/to/heart.jpg"  # Optional for multimodal
    # )
    # print(summary)
    
    # Example 4: Chat about image
    # response = service.chat_about_image(
    #     image_name="human heart",
    #     conversation_history=[],
    #     user_message="How does blood flow through the heart?",
    #     image_path="/path/to/heart.jpg"  # Optional for multimodal
    # )
    # print(response)