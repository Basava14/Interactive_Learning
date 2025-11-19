'use client';

import React, { Suspense, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment, useGLTF, Points, PointMaterial } from '@react-three/drei';
import { Loader2 } from 'lucide-react';
import * as THREE from 'three';

interface ModelViewerProps {
  modelUrl: string;
  depthMapUrl: string;
  pointCloudUrl?: string;
  viewMode: '3d-model' | 'textured' | 'point-cloud';
  imageName: string;
}

// Component to load and display GLB model
function GLBModel({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  const meshRef = useRef<THREE.Group>(null);

  // Auto-rotate the model
  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.005;
    }
  });

  return <primitive ref={meshRef} object={scene} scale={0.01} />;
}

// Component to display depth map as textured plane
function DepthMapPlane({ url }: { url: string }) {
  const texture = React.useMemo(() => {
    const loader = new THREE.TextureLoader();
    return loader.load(url);
  }, [url]);

  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[5, 5, 100, 100]} />
      <meshStandardMaterial map={texture} side={THREE.DoubleSide} />
    </mesh>
  );
}

// Component for point cloud visualization
function PointCloud({ url }: { url: string }) {
  const pointsRef = useRef<THREE.Points>(null);
  
  // Auto-rotate
  useFrame(() => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += 0.005;
    }
  });

  // For now, create a simple point cloud visualization
  // In production, you'd load the actual PLY file
  const particles = React.useMemo(() => {
    const count = 5000;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      positions[i3] = (Math.random() - 0.5) * 5;
      positions[i3 + 1] = (Math.random() - 0.5) * 5;
      positions[i3 + 2] = (Math.random() - 0.5) * 5;
      
      colors[i3] = Math.random();
      colors[i3 + 1] = Math.random();
      colors[i3 + 2] = Math.random();
    }

    return { positions, colors };
  }, []);

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particles.positions.length / 3}
          array={particles.positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={particles.colors.length / 3}
          array={particles.colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.05} vertexColors />
    </points>
  );
}

export default function ModelViewer({
  modelUrl,
  depthMapUrl,
  pointCloudUrl,
  viewMode,
  imageName,
}: ModelViewerProps) {
  const [error, setError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    setLoading(true);
    setError(null);
    
    // Check if model URL is accessible
    fetch(modelUrl, { method: 'HEAD' })
      .then(response => {
        if (!response.ok) {
          throw new Error('Model not found');
        }
        setLoading(false);
      })
      .catch(err => {
        console.error('Error loading model:', err);
        setError('Unable to load 3D model. Please try again.');
        setLoading(false);
      });
  }, [modelUrl]);

  return (
    <div className="w-full h-full relative">
      <Canvas shadows camera={{ position: [0, 2, 5], fov: 50 }}>
        <PerspectiveCamera makeDefault position={[0, 2, 5]} />
        
        <OrbitControls
          enableZoom={true}
          enablePan={true}
          enableRotate={true}
          minDistance={2}
          maxDistance={15}
          autoRotate={false}
        />
        
        {/* Lighting */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
        <pointLight position={[-10, -10, -5]} intensity={0.5} />
        
        <Suspense fallback={null}>
          {/* Render based on view mode */}
          {viewMode === '3d-model' && (
            <GLBModel url={modelUrl} />
          )}
          
          {viewMode === 'textured' && (
            <DepthMapPlane url={depthMapUrl} />
          )}
          
          {viewMode === 'point-cloud' && (
            <PointCloud url={pointCloudUrl || modelUrl} />
          )}
          
          {/* Environment for better reflections */}
          <Environment preset="sunset" />
        </Suspense>
        
        {/* Grid helper */}
        <gridHelper args={[10, 10]} />
      </Canvas>

      {/* Loading Overlay */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-80">
          <div className="text-center">
            <Loader2 className="w-12 h-12 animate-spin text-blue-500 mx-auto mb-3" />
            <p className="text-white text-sm">Loading 3D Model...</p>
          </div>
        </div>
      )}

      {/* Error Overlay */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-red-900 bg-opacity-80">
          <div className="text-center p-6">
            <p className="text-white text-lg mb-2">⚠️ {error}</p>
            <p className="text-gray-300 text-sm">Model URL: {modelUrl}</p>
          </div>
        </div>
      )}

      {/* Instructions Overlay */}
      {!loading && !error && (
        <div className="absolute bottom-4 left-4 bg-black bg-opacity-70 text-white px-4 py-2 rounded-lg text-sm pointer-events-none">
          <p>🖱️ Drag to rotate • Scroll to zoom • Right-click to pan</p>
        </div>
      )}
    </div>
  );
}

// Preload GLB models
useGLTF.preload = () => {};
