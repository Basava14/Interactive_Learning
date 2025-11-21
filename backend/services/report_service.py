from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from datetime import datetime
from typing import List, Dict, Optional
import io
import os
from PIL import Image
import tempfile
import re

class ReportService:
    @staticmethod
    def _format_text_for_pdf(text: str) -> str:
        """
        Convert Markdown-style formatting to HTML for ReportLab.
        Handles **bold** and basic text formatting.
        """
        if not text:
            return text
            
        # Convert **bold** to <b>bold</b>
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        
        # Convert *italic* to <i>italic</i>
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', text)
        
        # Convert line breaks to <br/> tags
        text = text.replace('\n', '<br/>')
        
        # Escape any remaining special characters that might break XML
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        
        # Restore our formatting tags
        text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
        text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
        text = text.replace('&lt;br/&gt;', '<br/>')
        
        return text
    
    @staticmethod
    def _prepare_image_for_pdf(image_path: str, max_width: float = 4.5 * inch, max_height: float = 3.5 * inch) -> Optional[str]:
        """
        Prepare image for PDF inclusion by resizing if necessary and converting to RGB.
        Returns path to temporary processed image or None if processing fails.
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (handles RGBA, P mode, etc.)
                if img.mode != 'RGB':
                    # Create white background for transparency
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        rgb_img.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    else:
                        rgb_img.paste(img)
                    img = rgb_img
                
                # Calculate resize dimensions while maintaining aspect ratio
                img_width, img_height = img.size
                aspect_ratio = img_width / img_height
                
                if img_width > max_width or img_height > max_height:
                    if aspect_ratio > 1:  # Landscape
                        new_width = min(max_width, img_width)
                        new_height = new_width / aspect_ratio
                    else:  # Portrait or square
                        new_height = min(max_height, img_height)
                        new_width = new_height * aspect_ratio
                    
                    img = img.resize((int(new_width), int(new_height)), Image.Resampling.LANCZOS)
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', prefix='report_img_')
                img.save(temp_file.name, 'JPEG', quality=85, optimize=True)
                return temp_file.name
                
        except Exception as e:
            print(f"Error processing image for PDF: {e}")
            return None
    
    @staticmethod
    def generate_conversation_report(
        image_name: str,
        summary: str,
        conversation_history: List[Dict[str, str]],
        image_path: Optional[str] = None
    ) -> bytes:
        """
        Generate a PDF report containing the original image, summary and chat conversation.
        
        Args:
            image_name: Name of the image
            summary: AI-generated summary
            conversation_history: Chat messages
            image_path: Optional path to the original image file
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            rightMargin=72, 
            leftMargin=72, 
            topMargin=72, 
            bottomMargin=18
        )

        # Container for the 'Flowable' objects
        elements = []
        temp_image_path = None

        try:
            # Define styles
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor='#1e40af',
                spaceAfter=30,
                alignment=TA_CENTER
            ))
            styles.add(ParagraphStyle(
                name='CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor='#1e40af',
                spaceAfter=12,
                spaceBefore=12
            ))
            styles.add(ParagraphStyle(
                name='ChatUser',
                parent=styles['Normal'],
                fontSize=11,
                textColor='#1f2937',
                leftIndent=20,
                spaceAfter=6
            ))
            styles.add(ParagraphStyle(
                name='ChatAssistant',
                parent=styles['Normal'],
                fontSize=11,
                textColor='#374151',
                leftIndent=20,
                spaceAfter=12
            ))

            # Title
            title = Paragraph("Interactive 3D Learning Report", styles['CustomTitle'])
            elements.append(title)
            elements.append(Spacer(1, 12))

            # Metadata
            date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            metadata = Paragraph(
                f"<b>Image:</b> {image_name}<br/><b>Generated:</b> {date_str}",
                styles['Normal']
            )
            elements.append(metadata)
            elements.append(Spacer(1, 20))

            # Original Image Section (NEW FEATURE)
            if image_path and os.path.exists(image_path):
                try:
                    image_title = Paragraph("Original Image", styles['CustomHeading'])
                    elements.append(image_title)
                    
                    # Process image for PDF inclusion
                    temp_image_path = ReportService._prepare_image_for_pdf(image_path)
                    
                    if temp_image_path:
                        # Add image to PDF
                        pdf_image = RLImage(
                            temp_image_path,
                            width=4.5 * inch,
                            height=3.5 * inch,
                            kind='proportional'  # Maintains aspect ratio
                        )
                        elements.append(pdf_image)
                        elements.append(Spacer(1, 20))
                    else:
                        # Fallback if image processing fails
                        error_msg = Paragraph(
                            "<i>Original image could not be included in the report.</i>",
                            styles['Normal']
                        )
                        elements.append(error_msg)
                        elements.append(Spacer(1, 20))
                        
                except Exception as e:
                    print(f"Error adding image to PDF: {e}")
                    # Continue without image - don't break the report generation
                    error_msg = Paragraph(
                        "<i>Original image could not be included in the report.</i>",
                        styles['Normal']
                    )
                    elements.append(error_msg)
                    elements.append(Spacer(1, 20))

            # Summary Section
            summary_title = Paragraph("Summary", styles['CustomHeading'])
            elements.append(summary_title)
            
            # Format summary text for proper display
            formatted_summary = ReportService._format_text_for_pdf(summary)
            summary_content = Paragraph(formatted_summary, styles['Normal'])
            elements.append(summary_content)
            elements.append(Spacer(1, 20))

            # Conversation Section
            if conversation_history:
                chat_title = Paragraph("Conversation History", styles['CustomHeading'])
                elements.append(chat_title)
                elements.append(Spacer(1, 12))

                for i, msg in enumerate(conversation_history):
                    if msg['role'] == 'user':
                        # Format user message content
                        formatted_content = ReportService._format_text_for_pdf(msg['content'])
                        user_msg = Paragraph(
                            f"<b>You:</b> {formatted_content}",
                            styles['ChatUser']
                        )
                        elements.append(user_msg)
                    elif msg['role'] == 'assistant':
                        # Format assistant message content
                        formatted_content = ReportService._format_text_for_pdf(msg['content'])
                        assistant_msg = Paragraph(
                            f"<b>AI Assistant:</b> {formatted_content}",
                            styles['ChatAssistant']
                        )
                        elements.append(assistant_msg)

            # Build PDF
            doc.build(elements)

            # Get PDF bytes
            pdf_bytes = buffer.getvalue()
            buffer.close()
            
            return pdf_bytes
            
        finally:
            # Clean up temporary image file
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.unlink(temp_image_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary image file: {e}")

# Singleton instance
report_service = ReportService()