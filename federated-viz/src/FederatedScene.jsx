/**
 * FederatedScene.jsx — React Three Fiber 3D Visualization
 * ========================================================
 * Central server node + 51 client nodes arranged in orbital rings,
 * with animated model broadcast, client training pulses,
 * update particle streams, and aggregation flashes.
 */
import React, { useRef, useMemo, useState, useEffect, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Stars, Float, Text, Html } from '@react-three/drei';
import * as THREE from 'three';

// ─── Constants ─────────────────────────────────────────────────
const GROUP_COLORS = {
  'Locomotion': '#FF6B6B',
  'Sedentary': '#4ECDC4',
  'Hand-Oriented': '#45B7D1',
  'Eating': '#96CEB4',
  'Sports': '#FFEAA7',
};

// ─── Server Node (center) ──────────────────────────────────────
function ServerNode({ phase, dpEnabled }) {
  const meshRef = useRef();
  const glowRef = useRef();
  const ringRef = useRef();

  useFrame((state) => {
    const t = state.clock.elapsedTime;

    if (meshRef.current) {
      meshRef.current.rotation.y = t * 0.3;
      meshRef.current.rotation.x = Math.sin(t * 0.2) * 0.1;

      // Phase-based animation
      const isAggregating = phase === 'aggregate';
      const targetScale = isAggregating ? 1.3 : 1.0;
      meshRef.current.scale.lerp(
        new THREE.Vector3(targetScale, targetScale, targetScale), 0.05
      );
    }

    if (glowRef.current) {
      const intensity = phase === 'aggregate' ? 2.5 : phase === 'broadcast' ? 1.8 : 1.0;
      glowRef.current.material.opacity = 0.15 + Math.sin(t * 3) * 0.05 + (intensity - 1) * 0.1;
      glowRef.current.scale.setScalar(1.8 + Math.sin(t * 2) * 0.2);
    }

    if (ringRef.current) {
      ringRef.current.rotation.z = t * 0.5;
      ringRef.current.rotation.x = Math.PI / 2 + Math.sin(t * 0.3) * 0.1;
    }
  });

  const serverColor = dpEnabled ? '#8b5cf6' : '#06b6d4';

  return (
    <group position={[0, 0, 0]}>
      {/* Core icosahedron */}
      <mesh ref={meshRef}>
        <icosahedronGeometry args={[0.8, 2]} />
        <meshStandardMaterial
          color={serverColor}
          emissive={serverColor}
          emissiveIntensity={0.4}
          metalness={0.8}
          roughness={0.2}
          wireframe={false}
        />
      </mesh>

      {/* Inner wireframe */}
      <mesh rotation={[0.3, 0.5, 0]}>
        <icosahedronGeometry args={[0.85, 1]} />
        <meshStandardMaterial
          color={serverColor}
          wireframe
          transparent
          opacity={0.3}
        />
      </mesh>

      {/* Glow sphere */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[1.2, 32, 32]} />
        <meshBasicMaterial
          color={serverColor}
          transparent
          opacity={0.1}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Orbital ring */}
      <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[1.2, 0.02, 16, 64]} />
        <meshStandardMaterial
          color={serverColor}
          emissive={serverColor}
          emissiveIntensity={0.5}
          transparent
          opacity={0.6}
        />
      </mesh>

      {/* Label */}
      <Text
        position={[0, -1.5, 0]}
        fontSize={0.25}
        color="#94a3b8"
        anchorX="center"
        anchorY="middle"
        font="https://fonts.gstatic.com/s/inter/v13/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfMZhrib2Bg-4.ttf"
      >
        GLOBAL SERVER
      </Text>
    </group>
  );
}

// ─── Client Node ───────────────────────────────────────────────
function ClientNode({ node, index, total, phase, activeClients, onHover, onUnhover }) {
  const meshRef = useRef();
  const pulseRef = useRef();

  // Position clients in concentric rings
  const position = useMemo(() => {
    const rings = total <= 8 ? 1 : total <= 24 ? 2 : 3;
    let ring, angleIndex, ringCount;

    if (rings === 1) {
      ring = 0;
      angleIndex = index;
      ringCount = total;
    } else if (rings === 2) {
      const firstRing = Math.min(12, Math.ceil(total / 2));
      if (index < firstRing) {
        ring = 0;
        angleIndex = index;
        ringCount = firstRing;
      } else {
        ring = 1;
        angleIndex = index - firstRing;
        ringCount = total - firstRing;
      }
    } else {
      const firstRing = 12;
      const secondRing = 18;
      if (index < firstRing) {
        ring = 0;
        angleIndex = index;
        ringCount = firstRing;
      } else if (index < firstRing + secondRing) {
        ring = 1;
        angleIndex = index - firstRing;
        ringCount = secondRing;
      } else {
        ring = 2;
        angleIndex = index - firstRing - secondRing;
        ringCount = total - firstRing - secondRing;
      }
    }

    const radius = 4 + ring * 2.5;
    const angle = (angleIndex / ringCount) * Math.PI * 2 - Math.PI / 2;
    const yOffset = ring * 0.8 - (rings - 1) * 0.4;

    return [
      Math.cos(angle) * radius,
      yOffset + Math.sin(angle * 3) * 0.3,
      Math.sin(angle) * radius,
    ];
  }, [index, total]);

  const isActive = activeClients.includes(node.id);
  const isTraining = phase === 'training' && isActive;
  const nodeColor = new THREE.Color(node.color || '#45B7D1');

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (!meshRef.current) return;

    // Idle floating
    meshRef.current.position.y = position[1] + Math.sin(t * 0.8 + index * 0.5) * 0.1;
    meshRef.current.rotation.y = t * 0.2 + index;

    // Training pulse
    if (isTraining) {
      const pulse = 1 + Math.sin(t * 8) * 0.15;
      meshRef.current.scale.setScalar(pulse);
    } else {
      meshRef.current.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);
    }

    if (pulseRef.current) {
      if (isTraining) {
        pulseRef.current.visible = true;
        pulseRef.current.material.opacity = 0.2 + Math.sin(t * 6) * 0.15;
        pulseRef.current.scale.setScalar(1.5 + Math.sin(t * 4) * 0.3);
      } else {
        pulseRef.current.visible = false;
      }
    }
  });

  const nodeSize = 0.3 + (node.totalSamples ? Math.min(node.totalSamples / 500000, 0.3) : 0);

  return (
    <group position={position}>
      {/* Training pulse aura */}
      <mesh ref={pulseRef} visible={false}>
        <sphereGeometry args={[nodeSize * 2, 16, 16]} />
        <meshBasicMaterial
          color={nodeColor}
          transparent
          opacity={0.15}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Main node */}
      <mesh
        ref={meshRef}
        onPointerEnter={(e) => {
          e.stopPropagation();
          document.body.style.cursor = 'pointer';
          onHover && onHover(node, e);
        }}
        onPointerLeave={(e) => {
          e.stopPropagation();
          document.body.style.cursor = 'default';
          onUnhover && onUnhover();
        }}
      >
        <dodecahedronGeometry args={[nodeSize, 0]} />
        <meshStandardMaterial
          color={nodeColor}
          emissive={nodeColor}
          emissiveIntensity={isTraining ? 0.6 : isActive ? 0.3 : 0.15}
          metalness={0.6}
          roughness={0.3}
        />
      </mesh>

      {/* Device indicator ring */}
      {node.devices && node.devices.length > 1 && (
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[nodeSize + 0.12, 0.015, 8, 32]} />
          <meshStandardMaterial
            color="#f59e0b"
            emissive="#f59e0b"
            emissiveIntensity={0.3}
          />
        </mesh>
      )}
    </group>
  );
}

// ─── Particle System (for model updates) ───────────────────────
function Particles({ phase, activeClients, clientPositions }) {
  const particlesRef = useRef();
  const count = 200;

  const { positions, velocities, colors, lifetimes } = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const vel = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const life = new Float32Array(count);
    return { positions: pos, velocities: vel, colors: col, lifetimes: life };
  }, []);

  useFrame((state) => {
    if (!particlesRef.current) return;
    const geo = particlesRef.current.geometry;
    const posAttr = geo.attributes.position;
    const colAttr = geo.attributes.color;
    const t = state.clock.elapsedTime;

    const isBroadcast = phase === 'broadcast';
    const isUpload = phase === 'upload';

    if (!isBroadcast && !isUpload) {
      // Reset particles off-screen
      for (let i = 0; i < count; i++) {
        posAttr.setXYZ(i, 0, -100, 0);
      }
      posAttr.needsUpdate = true;
      return;
    }

    for (let i = 0; i < count; i++) {
      lifetimes[i] -= 0.02;

      if (lifetimes[i] <= 0) {
        lifetimes[i] = 0.5 + Math.random() * 0.5;

        if (isBroadcast) {
          // From center outward
          const angle = Math.random() * Math.PI * 2;
          const elevation = (Math.random() - 0.5) * 1;
          posAttr.setXYZ(i, 0, 0, 0);
          velocities[i * 3] = Math.cos(angle) * (3 + Math.random() * 4);
          velocities[i * 3 + 1] = elevation;
          velocities[i * 3 + 2] = Math.sin(angle) * (3 + Math.random() * 4);
          colAttr.setXYZ(i, 0.024, 0.714, 0.831); // cyan
        } else {
          // From edges inward
          const angle = Math.random() * Math.PI * 2;
          const r = 5 + Math.random() * 4;
          posAttr.setXYZ(i,
            Math.cos(angle) * r,
            (Math.random() - 0.5) * 2,
            Math.sin(angle) * r
          );
          velocities[i * 3] = -Math.cos(angle) * (3 + Math.random() * 3);
          velocities[i * 3 + 1] = -posAttr.getY(i) * 0.5;
          velocities[i * 3 + 2] = -Math.sin(angle) * (3 + Math.random() * 3);
          colAttr.setXYZ(i, 0.063, 0.522, 0.957); // blue
        }
      }

      const x = posAttr.getX(i) + velocities[i * 3] * 0.016;
      const y = posAttr.getY(i) + velocities[i * 3 + 1] * 0.016;
      const z = posAttr.getZ(i) + velocities[i * 3 + 2] * 0.016;
      posAttr.setXYZ(i, x, y, z);
    }

    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
  });

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={count}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.08}
        vertexColors
        transparent
        opacity={0.8}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

// ─── Connection Lines ──────────────────────────────────────────
function ConnectionLines({ nodes, phase, activeClients }) {
  const linesRef = useRef();

  const getNodePosition = useCallback((idx, total) => {
    const rings = total <= 8 ? 1 : total <= 24 ? 2 : 3;
    let ring, angleIndex, ringCount;

    if (rings === 1) {
      ring = 0; angleIndex = idx; ringCount = total;
    } else if (rings === 2) {
      const firstRing = Math.min(12, Math.ceil(total / 2));
      if (idx < firstRing) {
        ring = 0; angleIndex = idx; ringCount = firstRing;
      } else {
        ring = 1; angleIndex = idx - firstRing; ringCount = total - firstRing;
      }
    } else {
      const firstRing = 12; const secondRing = 18;
      if (idx < firstRing) {
        ring = 0; angleIndex = idx; ringCount = firstRing;
      } else if (idx < firstRing + secondRing) {
        ring = 1; angleIndex = idx - firstRing; ringCount = secondRing;
      } else {
        ring = 2; angleIndex = idx - firstRing - secondRing;
        ringCount = total - firstRing - secondRing;
      }
    }

    const radius = 4 + ring * 2.5;
    const angle = (angleIndex / ringCount) * Math.PI * 2 - Math.PI / 2;
    return new THREE.Vector3(
      Math.cos(angle) * radius,
      ring * 0.8 - (rings - 1) * 0.4,
      Math.sin(angle) * radius
    );
  }, []);

  useFrame((state) => {
    if (!linesRef.current) return;
    const t = state.clock.elapsedTime;
    const showLines = phase === 'broadcast' || phase === 'upload' || phase === 'training';
    linesRef.current.material.opacity = showLines ? 0.15 + Math.sin(t * 4) * 0.05 : 0.03;
  });

  const lineGeometry = useMemo(() => {
    const points = [];
    const center = new THREE.Vector3(0, 0, 0);
    nodes.forEach((_, idx) => {
      const pos = getNodePosition(idx, nodes.length);
      points.push(center.clone(), pos);
    });
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    return geo;
  }, [nodes, getNodePosition]);

  return (
    <lineSegments ref={linesRef} geometry={lineGeometry}>
      <lineBasicMaterial
        color="#1e3a5f"
        transparent
        opacity={0.05}
        depthWrite={false}
      />
    </lineSegments>
  );
}

// ─── Grid Floor ────────────────────────────────────────────────
function GridFloor() {
  return (
    <group position={[0, -3, 0]}>
      <gridHelper args={[30, 30, '#1a2744', '#0f1b2d']} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
        <planeGeometry args={[30, 30]} />
        <meshStandardMaterial
          color="#050a18"
          transparent
          opacity={0.5}
        />
      </mesh>
    </group>
  );
}

// ─── Camera Controller ─────────────────────────────────────────
function CameraController() {
  const { camera } = useThree();

  useEffect(() => {
    camera.position.set(8, 5, 8);
    camera.lookAt(0, 0, 0);
  }, [camera]);

  return (
    <OrbitControls
      enableDamping
      dampingFactor={0.05}
      minDistance={5}
      maxDistance={25}
      maxPolarAngle={Math.PI * 0.85}
      minPolarAngle={Math.PI * 0.1}
      enablePan={false}
    />
  );
}

// ─── Main Scene Component ──────────────────────────────────────
export default function FederatedScene({
  nodes = [],
  phase = 'idle',
  activeClients = [],
  dpEnabled = false,
  currentRound = 0,
  onNodeHover,
  onNodeUnhover,
}) {
  const displayNodes = nodes.length > 0 ? nodes : generateDefaultNodes();

  return (
    <Canvas
      camera={{ position: [8, 5, 8], fov: 50, near: 0.1, far: 100 }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: false }}
      style={{ background: '#0a0e1a' }}
    >
      {/* Lighting */}
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={0.6} color="#e2e8f0" />
      <pointLight position={[-10, -5, -10]} intensity={0.3} color="#3b82f6" />
      <pointLight position={[0, 0, 0]} intensity={phase === 'aggregate' ? 2 : 0.5} color="#06b6d4" />
      <directionalLight position={[5, 8, 3]} intensity={0.4} color="#f1f5f9" />

      {/* Background */}
      <color attach="background" args={['#0a0e1a']} />
      <fog attach="fog" args={['#0a0e1a', 15, 35]} />
      <Stars radius={50} depth={50} count={2000} factor={3} saturation={0.2} fade speed={0.5} />

      {/* Scene */}
      <CameraController />
      <GridFloor />

      {/* Server */}
      <ServerNode phase={phase} dpEnabled={dpEnabled} />

      {/* Client Nodes */}
      {displayNodes.map((node, index) => (
        <ClientNode
          key={node.id}
          node={node}
          index={index}
          total={displayNodes.length}
          phase={phase}
          activeClients={activeClients}
          onHover={onNodeHover}
          onUnhover={onNodeUnhover}
        />
      ))}

      {/* Connection Lines */}
      <ConnectionLines
        nodes={displayNodes}
        phase={phase}
        activeClients={activeClients}
      />

      {/* Particles */}
      <Particles
        phase={phase}
        activeClients={activeClients}
        clientPositions={[]}
      />
    </Canvas>
  );
}

// ─── Default nodes for demo mode ───────────────────────────────
function generateDefaultNodes() {
  const groups = Object.keys(GROUP_COLORS);
  return Array.from({ length: 8 }, (_, i) => ({
    id: i,
    label: `Client ${i}`,
    totalSamples: 150000 + Math.random() * 200000,
    dominantGroup: groups[i % groups.length],
    dominantActivity: ['Walking', 'Sitting', 'Jogging', 'Typing', 'Kicking', 'Eating', 'Standing', 'Stairs'][i % 8],
    color: GROUP_COLORS[groups[i % groups.length]],
    devices: i % 3 === 0 ? ['phone', 'watch'] : ['phone'],
    numActivities: 12 + Math.floor(Math.random() * 7),
    entropy: 3.5 + Math.random(),
  }));
}
