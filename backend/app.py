from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict
import io
import os
import cv2
import numpy as np
import torch
from pathlib import Path
import base64
from PIL import Image
import trimesh

# Import your services
try:
    from services.openrouter_service import openrouter_service
    from services.report_service import report_service
except ImportError:
    print("Warning: Services not found. Creating mock services.")
    openrouter_service = None
    report_service = None

# Create FastAPI app instance
app = FastAPI(
    title="2D to 3D Converter API",
    description="Convert 2D images to 3D models with AI-powered learning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directories
OUTPUT_DIR = Path("outputs")
MODELS_DIR = OUTPUT_DIR / "models"
DEPTH_DIR = OUTPUT_DIR / "depth"
POINTCLOUD_DIR = OUTPUT_DIR / "pointcloud"

for dir_path in [OUTPUT_DIR, MODELS_DIR, DEPTH_DIR, POINTCLOUD_DIR]:
    dir_path.mkdir(exist_ok=True)

# Mount static files with CORS support
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Add CORS headers for static files
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


# Global variable to cache the MiDaS model
midas_model = None
midas_transform = None
device = None

def load_midas_model():
    """Load MiDaS model for depth estimation"""
    global midas_model, midas_transform, device
    
    if midas_model is None:
        print("Loading MiDaS model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load MiDaS model from torch hub
        midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        # midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")  # Faster, still good quality
        midas_model.to(device)
        midas_model.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        midas_transform = midas_transforms.dpt_transform
        
        print(f"MiDaS model loaded on device: {device}")
    
    return midas_model, midas_transform, device

def estimate_depth(image_rgb):
    """
    Estimate depth map from RGB image using MiDaS
    """
    model, transform, device = load_midas_model()
    
    # Prepare image for MiDaS
    input_batch = transform(image_rgb).to(device)
    
    # Predict depth
    with torch.no_grad():
        prediction = model(input_batch)
        
        # Resize to original resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    
    # Normalize depth map to 0-255 range
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
    depth_image = (depth_normalized * 255).astype(np.uint8)
    
    return depth_normalized, depth_image

def create_3d_mesh(image_rgb, depth_map, output_path):
    """ Create 3D mesh from RGB image and depth map using trimesh """
    h, w = depth_map.shape

    # Create vertex grid (image space: x rightwards, y downwards)
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    x_grid, y_grid = np.meshgrid(x, y)

    # Flip Y to convert to a Y-up world (renderer) convention
    y_world = (h - 1) - y_grid

    # Flatten arrays
    x_flat = x_grid.flatten()
    y_flat = y_world.flatten()
    z_flat = depth_map.flatten()

    # Scale depth for better visualization
    z_scaled = z_flat * 100.0
    
    # Create front surface vertices
    front_vertices = np.column_stack([x_flat, y_flat, z_scaled])
    
    # Create back surface vertices (slightly behind front surface)
    back_offset = 2.0  # Small thickness to avoid z-fighting
    back_vertices = np.column_stack([x_flat, y_flat, z_scaled - back_offset])
    
    # Combine all vertices
    vertices = np.vstack([front_vertices, back_vertices])
    
    # Create vertex colors (duplicate for front and back)
    colors = image_rgb.reshape(-1, 3)
    all_colors = np.vstack([colors, colors])

    # Create faces with proper winding order
    faces = []
    
    # Front faces (counter-clockwise when viewed from front)
    for i in range(h - 1):
        for j in range(w - 1):
            idx = i * w + j
            # Triangle 1: top-left, bottom-left, top-right
            faces.append([idx, idx + w, idx + 1])
            # Triangle 2: top-right, bottom-left, bottom-right  
            faces.append([idx + 1, idx + w, idx + w + 1])
    
    # Back faces (clockwise when viewed from front = counter-clockwise from back)
    vertex_offset = h * w  # Offset to back vertices
    for i in range(h - 1):
        for j in range(w - 1):
            idx = i * w + j + vertex_offset
            # Triangle 1: top-left, top-right, bottom-left (reversed winding)
            faces.append([idx, idx + 1, idx + w])
            # Triangle 2: top-right, bottom-right, bottom-left (reversed winding)
            faces.append([idx + 1, idx + w + 1, idx + w])

    faces = np.array(faces)

    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=all_colors)
    
    # Ensure proper normals
    mesh.fix_normals()
    
    # Remove any degenerate faces
    mesh.remove_degenerate_faces()
    
    # Export mesh
    mesh.export(output_path)
    return str(output_path)

def create_point_cloud(image_rgb, depth_map, output_path):
    """ Create 3D point cloud from RGB image and depth map """
    h, w = depth_map.shape

    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    x_grid, y_grid = np.meshgrid(x, y)

    # Flip Y for Y-up convention
    y_world = (h - 1) - y_grid

    # Points: [X, Y, Z]
    points = np.column_stack([
        x_grid.flatten(),
        y_world.flatten(),          # flipped Y
        depth_map.flatten() * 100.0
    ])

    colors = image_rgb.reshape(-1, 3)

    point_cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
    point_cloud.export(output_path)
    return str(output_path)

# Pydantic models
class ChatRequest(BaseModel):
    image_name: str
    conversation_history: List[Dict[str, str]]
    user_message: str

class SummaryRequest(BaseModel):
    image_name: str
    image_type: str = "general"

class ReportRequest(BaseModel):
    image_name: str
    summary: str
    conversation_history: List[Dict[str, str]]

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "2D to 3D Converter API is running",
        "status": "ok",
        "version": "1.0.0"
    }

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# 3D Conversion Endpoint (Complete Implementation)
@app.post("/api/convert")
async def convert_image_to_3d(file: UploadFile = File(...)):
    """
    Convert 2D image to 3D model using MiDaS depth estimation
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        print(f"Processing image: {file.filename}, shape: {img_rgb.shape}")
        
        # Generate unique filename
        base_name = Path(file.filename).stem
        timestamp = int(os.path.getmtime(__file__) * 1000) if os.path.exists(__file__) else 0
        unique_name = f"{base_name}_{timestamp}"
        
        # 1. Estimate depth map
        print("Estimating depth map...")
        depth_normalized, depth_image = estimate_depth(img_rgb)
        
        # Save depth map
        depth_path = DEPTH_DIR / f"{unique_name}_depth.png"
        cv2.imwrite(str(depth_path), depth_image)
        print(f"Depth map saved to: {depth_path}")
        
        # 2. Create 3D mesh (GLB format)
        print("Creating 3D mesh...")
        mesh_path = MODELS_DIR / f"{unique_name}.glb"
        create_3d_mesh(img_rgb, depth_normalized, mesh_path)
        print(f"3D mesh saved to: {mesh_path}")
        
        # 3. Create point cloud
        print("Creating point cloud...")
        pointcloud_path = POINTCLOUD_DIR / f"{unique_name}.ply"
        create_point_cloud(img_rgb, depth_normalized, pointcloud_path)
        print(f"Point cloud saved to: {pointcloud_path}")
        
        # Return URLs to the generated files
        response_data = {
            "model_url": f"http://localhost:8000/outputs/models/{mesh_path.name}",
            "depth_map_url": f"http://localhost:8000/outputs/depth/{depth_path.name}",
            "point_cloud_url": f"http://localhost:8000/outputs/pointcloud/{pointcloud_path.name}",
            "format": "glb",
            "success": True,
            "message": "3D model generated successfully"
        }
        
        print("Conversion completed successfully!")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"Error in conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

# Generate Summary Endpoint
@app.post("/api/generate-summary")
async def generate_summary(request: SummaryRequest):
    """Generate an educational summary about the uploaded image."""
    try:
        if openrouter_service:
            summary = openrouter_service.generate_image_summary(
                request.image_name, 
                request.image_type
            )
        else:
            summary = f"This is an educational overview of {request.image_name}. The 3D model allows you to explore the structure and features in detail."
        
        return JSONResponse(content={"summary": summary})
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return JSONResponse(
            content={"summary": "Unable to generate summary at this time."},
            status_code=200
        )

# Chat Endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handle chat conversation about the image."""
    try:
        if openrouter_service:
            response = openrouter_service.chat_about_image(
                request.image_name,
                request.conversation_history,
                request.user_message
            )
        else:
            response = f"I'm here to help you learn about {request.image_name}. Unfortunately, the AI chat service is not configured yet."
        
        return JSONResponse(content={"response": response})
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return JSONResponse(
            content={"response": "Sorry, I'm having trouble processing your question right now."},
            status_code=200
        )

# Export Report Endpoint
@app.post("/api/export-report")
async def export_report(request: ReportRequest):
    """Export summary and conversation as a PDF report."""
    try:
        if report_service:
            pdf_bytes = report_service.generate_conversation_report(
                request.image_name,
                request.summary,
                request.conversation_history
            )
            
            filename = f"learning_report_{request.image_name.replace(' ', '_')}.pdf"
            
            return StreamingResponse(
                io.BytesIO(pdf_bytes),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        else:
            raise HTTPException(status_code=503, detail="Report service not available")
    except Exception as e:
        print(f"Error exporting report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# List all routes (for debugging)
@app.get("/api/routes")
async def list_routes():
    """Debug endpoint to see all registered routes"""
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": list(route.methods) if hasattr(route, 'methods') else []
        })
    return {"routes": routes}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
