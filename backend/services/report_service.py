from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from datetime import datetime
from typing import List, Dict
import io

class ReportService:
    @staticmethod
    def generate_conversation_report(
        image_name: str,
        summary: str,
        conversation_history: List[Dict[str, str]]
    ) -> bytes:
        """
        Generate a PDF report containing the summary and chat conversation.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='CustomTitle', 
                                  parent=styles['Heading1'],
                                  fontSize=24,
                                  textColor='#1e40af',
                                  spaceAfter=30,
                                  alignment=TA_CENTER))
        
        styles.add(ParagraphStyle(name='CustomHeading', 
                                  parent=styles['Heading2'],
                                  fontSize=14,
                                  textColor='#1e40af',
                                  spaceAfter=12,
                                  spaceBefore=12))
        
        styles.add(ParagraphStyle(name='ChatUser', 
                                  parent=styles['Normal'],
                                  fontSize=11,
                                  textColor='#1f2937',
                                  leftIndent=20,
                                  spaceAfter=6))
        
        styles.add(ParagraphStyle(name='ChatAssistant', 
                                  parent=styles['Normal'],
                                  fontSize=11,
                                  textColor='#374151',
                                  leftIndent=20,
                                  spaceAfter=12))
        
        # Title
        title = Paragraph("Interactive 3D Learning Report", styles['CustomTitle'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Metadata
        date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        metadata = Paragraph(f"<b>Image:</b> {image_name}<br/><b>Generated:</b> {date_str}", 
                            styles['Normal'])
        elements.append(metadata)
        elements.append(Spacer(1, 20))
        
        # Summary Section
        summary_title = Paragraph("Summary", styles['CustomHeading'])
        elements.append(summary_title)
        summary_content = Paragraph(summary, styles['Normal'])
        elements.append(summary_content)
        elements.append(Spacer(1, 20))
        
        # Conversation Section
        if conversation_history:
            chat_title = Paragraph("Conversation History", styles['CustomHeading'])
            elements.append(chat_title)
            elements.append(Spacer(1, 12))
            
            for i, msg in enumerate(conversation_history):
                if msg['role'] == 'user':
                    user_msg = Paragraph(f"<b>You:</b> {msg['content']}", styles['ChatUser'])
                    elements.append(user_msg)
                elif msg['role'] == 'assistant':
                    assistant_msg = Paragraph(f"<b>AI Assistant:</b> {msg['content']}", 
                                             styles['ChatAssistant'])
                    elements.append(assistant_msg)
        
        # Build PDF
        doc.build(elements)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes

# Singleton instance
report_service = ReportService()
