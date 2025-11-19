# Interactive_Learning

This project enables learners to upload an image and:

- Generate a depth map using MiDaS (DPT_Large) via Torch Hub.
- Create a 3D mesh (GLB) and a point cloud (PLY) from the image and depth.
- Obtain an educational summary and chat context about the image.
- Export a learning report (summary + chat) as a PDF.

## Features

#### 2D → 3D conversion:
- Depth estimation (MiDaS) with GPU/CPU auto-detection.
- Mesh export as GLB and point cloud export as PLY.
- Depth map saved as PNG.
#### AI-assisted learning:
- Image summary generation (optional integration).
- Chat endpoint for Q&A on the image (optional integration).
#### Reporting:
- Export session data as a PDF (optional integration).
- Static serving of generated assets via FastAPI.

## Tech Stack

### Frontend
- React.js with Next.js App Router
- Three.js and React Three Fiber for 3D rendering
- Tailwind CSS for styling
- TypeScript for type safety
- Radix UI components

### Backend
- Python with FastAPI
- PyTorch for ML models
- MiDaS for depth estimation
- OpenCV for image processing
- Trimesh for 3D mesh generation
- PyRender for 3D rendering

## Installation

### Prerequisites

- Python 3.9.7 with pip
- Node.js V24.11.1 with npm 11.6
- (Optional) CUDA-capable GPU for faster depth estimation.
- Internet access for first run (Torch Hub downloads MiDaS weights).
- Git

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Basava14/Interactive_Learning.git
   cd interactive_learning
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   

3. **Install Python dependencies**
   ```bash
   pip install -r req_full.txt
   ```
4. **Install Python dependencies**
    install additional liberies like Torch, timm

5. **Start the backend server**
   ```
   cd backend
   python run.py
   ```
   The backend API will be available at `http://localhost:8000`


### Frontend Setup

1. **Navigate to the frontend directory**
   ```bash
   cd 2d-to-3d-converter
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:3000`

