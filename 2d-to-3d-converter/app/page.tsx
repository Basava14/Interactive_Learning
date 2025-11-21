'use client';

import React, { useState, useCallback } from 'react';
import { Upload, Loader2, RefreshCw, Download, Eye, Box, Cloud } from 'lucide-react';
import ChatPanel from '@/components/ChatPanel';
import dynamic from 'next/dynamic';

// Dynamically import Three.js components (client-side only)
const ModelViewer = dynamic(() => import('@/components/ModelViewer'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full bg-gray-900 rounded-lg">
      <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
    </div>
  ),
});

// Types
interface ConvertedModel {
  model_url: string;
  depth_map_url: string;
  point_cloud_url?: string;
  format: string;
  original_image_path?: string;
  unique_name?: string;
}

type ViewMode = '3d-model' | 'textured' | 'point-cloud';

export default function Home() {
  // State Management
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string>('');
  const [imageName, setImageName] = useState<string>('');
  const [generatedModel, setGeneratedModel] = useState<ConvertedModel | null>(null);
  const [summary, setSummary] = useState<string>('');
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false);
  const [isGeneratingModel, setIsGeneratingModel] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('3d-model');
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string>('');
  const [imageMetadata, setImageMetadata] = useState<{
    uniqueName?: string;
    originalImagePath?: string;
  }>({});

  // Handle Image Upload and Processing
  const handleImageUpload = useCallback(async (file: File) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }

    setError('');
    setUploadedImage(file);
    setImageName(file.name.replace(/\.[^/.]+$/, ''));

    // Create preview URL
    const previewUrl = URL.createObjectURL(file);
    setImagePreviewUrl(previewUrl);

    // Start 3D model generation
    setIsGeneratingModel(true);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append('file', file);

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      const response = await fetch('http://localhost:8000/api/convert', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        throw new Error('Failed to generate 3D model');
      }

      const data: ConvertedModel = await response.json();
      setGeneratedModel(data);

      // After successful 3D generation, generate summary automatically
      await generateSummary(file.name);
    } catch (err) {
      console.error('Error generating 3D model:', err);
      setError('Failed to generate 3D model. Please try again.');
      setGeneratedModel(null);
    } finally {
      setIsGeneratingModel(false);
      setUploadProgress(0);
    }
  }, []);

  // Generate AI Summary
  const generateSummary = async (fileName: string) => {
    setIsGeneratingSummary(true);
    try {
      const cleanName = fileName.replace(/\.[^/.]+$/, '');
      const response = await fetch('http://localhost:8000/api/generate-summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_name: cleanName,
          image_type: 'general',
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate summary');
      }

      const data = await response.json();
      setSummary(data.summary);
    } catch (error) {
      console.error('Error generating summary:', error);
      setSummary('Unable to generate summary at this time. You can still use the chat to ask questions!');
    } finally {
      setIsGeneratingSummary(false);
    }
  };

  // Drag and Drop Handlers
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleImageUpload(e.target.files[0]);
    }
  };

  // Download Model Handler
  const handleDownloadModel = async () => {
    if (!generatedModel) return;

    try {
      const link = document.createElement('a');
      link.href = generatedModel.model_url;
      link.download = `${imageName}_3d_model.${generatedModel.format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Error downloading model:', error);
      alert('Failed to download model. Please try again.');
    }
  };

  // Reset to Upload New Image
  const handleReset = () => {
    setUploadedImage(null);
    setImagePreviewUrl('');
    setImageName('');
    setGeneratedModel(null);
    setSummary('');
    setError('');
    setViewMode('3d-model');
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-gray-100">
      {/* Header */}
      <header className="bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 shadow-2xl border-b border-gray-700 sticky top-0 z-50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                <div className="p-2 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl shadow-lg">
                  <Box className="w-8 h-8 text-white" />
                </div>
                Interactive 3D Learning Platform
              </h1>
              <p className="text-gray-300 mt-2 ml-14">
                Convert 2D images to 3D models and explore with AI-powered learning
              </p>
            </div>
            {generatedModel && (
              <button
                onClick={handleReset}
                className="flex items-center gap-2 px-5 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 transition-all duration-300 font-semibold shadow-lg hover:shadow-xl hover:scale-105 active:scale-95"
              >
                <RefreshCw className="w-5 h-5" />
                Upload New
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!generatedModel ? (
          /* Upload Section */
          <div className="flex items-center justify-center min-h-[calc(100vh-12rem)]">
            <div className="w-full max-w-2xl">
              <div
                className={`relative border-3 border-dashed rounded-2xl p-12 text-center transition-all duration-300 ${dragActive
                    ? 'border-blue-500 bg-blue-50 scale-105'
                    : 'border-gray-300 bg-white hover:border-blue-400 hover:bg-blue-50'
                  }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                {isGeneratingModel ? (
                  /* Loading State */
                  <div className="space-y-6">
                    <Loader2 className="w-16 h-16 animate-spin text-blue-600 mx-auto" />
                    <div>
                      <h3 className="text-xl font-semibold text-gray-900 mb-2">
                        Generating 3D Model...
                      </h3>
                      <p className="text-gray-600">
                        Processing your image with AI depth estimation
                      </p>
                    </div>
                    {uploadProgress > 0 && (
                      <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                        <div
                          className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                          style={{ width: `${uploadProgress}%` }}
                        />
                      </div>
                    )}
                  </div>
                ) : (
                  /* Upload UI */
                  <>
                    <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-2xl font-semibold text-gray-900 mb-2">
                      Upload Your Image
                    </h3>
                    <p className="text-gray-600 mb-6">
                      Drag and drop your image here, or click to browse
                    </p>

                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileInput}
                      className="hidden"
                      id="file-input"
                      disabled={isGeneratingModel}
                    />

                    <label
                      htmlFor="file-input"
                      className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors cursor-pointer font-medium"
                    >
                      <Upload className="w-5 h-5" />
                      Choose Image
                    </label>

                    <p className="text-sm text-gray-500 mt-6">
                      Supported formats: JPG, PNG, GIF • Max size: 10MB
                    </p>

                    {error && (
                      <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                        <p className="text-red-600 text-sm">{error}</p>
                      </div>
                    )}
                  </>
                )}
              </div>

              {/* Feature Highlights */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
                <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
                  <Box className="w-8 h-8 text-blue-600 mb-2" />
                  <h4 className="font-semibold text-gray-900 mb-1">3D Conversion</h4>
                  <p className="text-sm text-gray-600">
                    AI-powered depth estimation creates realistic 3D models
                  </p>
                </div>
                <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
                  <Eye className="w-8 h-8 text-blue-600 mb-2" />
                  <h4 className="font-semibold text-gray-900 mb-1">Multiple Views</h4>
                  <p className="text-sm text-gray-600">
                    Explore with 3D mesh, textured, and point cloud modes
                  </p>
                </div>
                <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
                  <Cloud className="w-8 h-8 text-blue-600 mb-2" />
                  <h4 className="font-semibold text-gray-900 mb-1">AI Learning</h4>
                  <p className="text-sm text-gray-600">
                    Get instant summaries and ask questions about your image
                  </p>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Side-by-side Layout: 3D Viewer + Chat Panel */
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8" style={{ height: 'calc(100vh - 12rem)' }}>
            {/* Left: 3D Viewer with Enhanced Design */}
            <div className="bg-gradient-to-br from-white to-gray-50 rounded-2xl shadow-2xl border border-gray-200/50 flex flex-col h-full overflow-hidden backdrop-blur-sm">
              {/* Viewer Header with Gradient */}
              <div className="px-6 py-5 border-b border-gray-200/50 bg-gradient-to-r from-gray-50 to-white flex-shrink-0 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <div className="p-1.5 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-lg">
                        <Box className="w-5 h-5 text-white" />
                      </div>
                      <h2 className="text-xl font-bold text-gray-900">3D Model Viewer</h2>
                    </div>
                    <p className="text-sm text-gray-600 font-medium ml-8">{imageName}</p>
                  </div>
                  <button
                    onClick={handleDownloadModel}
                    className="group flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 transition-all duration-300 text-sm font-semibold shadow-lg hover:shadow-xl hover:scale-105 active:scale-95"
                  >
                    <Download className="w-4 h-4 group-hover:animate-bounce" />
                    Download
                  </button>
                </div>

                {/* View Mode Selector with Modern Design */}
                <div className="flex gap-2 p-1.5 bg-gray-100/80 rounded-xl backdrop-blur-sm">
                  <button
                    onClick={() => setViewMode('3d-model')}
                    className={`flex-1 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all duration-300 ${viewMode === '3d-model'
                        ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg scale-105'
                        : 'text-gray-700 hover:bg-white hover:shadow-md'
                      }`}
                  >
                    3D Model
                  </button>
                  <button
                    onClick={() => setViewMode('textured')}
                    className={`flex-1 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all duration-300 ${viewMode === 'textured'
                        ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg scale-105'
                        : 'text-gray-700 hover:bg-white hover:shadow-md'
                      }`}
                  >
                    Textured 3D
                  </button>
                  <button
                    onClick={() => setViewMode('point-cloud')}
                    className={`flex-1 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all duration-300 ${viewMode === 'point-cloud'
                        ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg scale-105'
                        : 'text-gray-700 hover:bg-white hover:shadow-md'
                      }`}
                  >
                    Point Cloud
                  </button>
                </div>
              </div>

              {/* 3D Viewer Canvas */}
              <div className="flex-1 bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 rounded-b-2xl overflow-hidden min-h-0 relative">
                <div className="absolute inset-0 bg-gradient-to-t from-blue-900/10 to-transparent pointer-events-none"></div>
                <ModelViewer
                  modelUrl={generatedModel.model_url}
                  depthMapUrl={generatedModel.depth_map_url}
                  pointCloudUrl={generatedModel.point_cloud_url}
                  viewMode={viewMode}
                  imageName={imageName}
                />
              </div>
            </div>

            {/* Right: Chat Panel */}
            <div className="h-full overflow-hidden">
              <ChatPanel
                imageName={imageName}
                summary={summary}
                isGeneratingSummary={isGeneratingSummary}
                imageMetadata={imageMetadata} 
              />
            </div>
          </div>

        )}
      </main>
    </div>
  );
}
