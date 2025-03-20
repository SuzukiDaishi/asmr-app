"use client";

import React, {
  useMemo,
  useRef,
  useEffect,
  forwardRef,
  useCallback,
  JSX
} from 'react';
import { Canvas, useFrame, extend } from '@react-three/fiber';
import * as THREE from 'three';
import { EffectComposer, Bloom, ChromaticAberration } from '@react-three/postprocessing';
import { BlendFunction } from 'postprocessing';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';

//────────────────────────────
// 1. Gem (Jewel) Shader Material
//────────────────────────────

const gemVertexShader = /* glsl */ `
  precision mediump float;
  varying vec3 vNormal;
  varying vec3 vWorldPosition;
  varying vec2 vUv;
  void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

const gemFragmentShader = /* glsl */ `
  precision mediump float;
  uniform float uTime;
  uniform vec3 uColor;
  uniform float uFresnelPower;
  uniform float uChromaticAberration;
  uniform float uAlpha;
  varying vec3 vNormal;
  varying vec3 vWorldPosition;
  varying vec2 vUv;
  void main() {
    float fresnel = pow(1.0 - dot(normalize(vNormal), normalize(cameraPosition - vWorldPosition)), uFresnelPower);
    vec3 color = mix(uColor, vec3(1.0), fresnel * uChromaticAberration);
    gl_FragColor = vec4(color, uAlpha);
  }
`;

class GemMaterialImpl extends THREE.ShaderMaterial {
  constructor() {
    super({
      vertexShader: gemVertexShader,
      fragmentShader: gemFragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uColor: { value: new THREE.Color('#ffffff') },
        uFresnelPower: { value: 3.0 },
        uChromaticAberration: { value: 0.1 },
        uAlpha: { value: 0.9 },
      },
      transparent: true,
    });
  }
}
extend({ GemMaterialImpl });

//────────────────────────────
// 2. DistortedSphere Component
//────────────────────────────

export type DistortedSphereProps = JSX.IntrinsicElements['mesh'] & {
  radius?: number;
  color?: string;
  fresnelPower?: number;
  chromaticAberration?: number;
  alpha?: number;
  stopRotation?: boolean;
};

export interface DistortedSphereHandle {
  mesh: THREE.Mesh | null;
  geometry: THREE.BufferGeometry | undefined;
}

export const DistortedSphere = forwardRef<DistortedSphereHandle, DistortedSphereProps>(({
  radius = 1,
  color = '#ffffff',
  fresnelPower = 3.0,
  chromaticAberration = 0.1,
  alpha = 0.9,
  stopRotation = false,
  ...props
}, ref) => {
  const meshRef = useRef<THREE.Mesh>(null);
  
  // Imperative handle に mesh と追加情報 (geometry) を渡す
  React.useImperativeHandle(ref, () => ({
    mesh: meshRef.current,
    geometry: meshRef.current ? (meshRef.current.geometry as THREE.BufferGeometry) : undefined,
  }));
  
  const geometry = useMemo(() => {
    const geom = new THREE.IcosahedronGeometry(radius, 3);
    const pos = geom.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const z = pos.getZ(i);
      const factor = 0.1;
      pos.setXYZ(i, x + (Math.random() - 0.5) * factor, y + (Math.random() - 0.5) * factor, z + (Math.random() - 0.5) * factor);
    }
    geom.computeVertexNormals();
    return geom;
  }, [radius]);
  
  useFrame((state, delta) => {
    if (!meshRef.current) return;
    const material = meshRef.current.material as THREE.ShaderMaterial;
    material.uniforms.uTime.value = state.clock.getElapsedTime();
    if (!stopRotation) {
      const t = state.clock.getElapsedTime();
      meshRef.current.rotation.y += delta * (0.5 + Math.sin(t * 0.5) * 0.3);
      meshRef.current.rotation.x += delta * (0.2 + Math.cos(t * 0.5) * 0.1);
    }
  });
  
  useEffect(() => {
    if (meshRef.current) {
      const material = meshRef.current.material as THREE.ShaderMaterial;
      material.uniforms.uColor.value = new THREE.Color(color);
      material.uniforms.uFresnelPower.value = fresnelPower;
      material.uniforms.uChromaticAberration.value = chromaticAberration;
      material.uniforms.uAlpha.value = alpha;
    }
  }, [color, fresnelPower, chromaticAberration, alpha]);
  
  return (
    <mesh ref={meshRef} geometry={geometry} {...props} castShadow receiveShadow>
      <primitive object={new GemMaterialImpl()} attach="material" />
    </mesh>
  );
});

//────────────────────────────
// 3. SatelliteSphere Component
//────────────────────────────

export type SatelliteSphereProps = DistortedSphereProps & {
  initialPosition: THREE.Vector3;
  targetPosition: THREE.Vector3;
};

export const SatelliteSphere: React.FC<SatelliteSphereProps> = ({
  initialPosition,
  targetPosition,
  ...sphereProps
}) => {
  // Ref は DistortedSphereHandle 型で管理する
  const sphereHandleRef = useRef<DistortedSphereHandle>(null);

  useEffect(() => {
    if (sphereHandleRef.current && sphereHandleRef.current.mesh) {
      sphereHandleRef.current.mesh.position.copy(initialPosition);
    }
  }, [initialPosition]);

  useFrame((state, delta) => {
    if (!sphereHandleRef.current || !sphereHandleRef.current.mesh) return;
    sphereHandleRef.current.mesh.position.lerp(targetPosition, delta * 0.5);
  });

  // handleRef は DistortedSphereHandle を受け取る
  const handleRef: React.RefCallback<DistortedSphereHandle> = useCallback((instance) => {
    sphereHandleRef.current = instance;
  }, []);

  return (
    // @ts-ignore
    <DistortedSphere ref={handleRef} {...sphereProps} />
  );
};

//────────────────────────────
// 4. DustParticles Component
//────────────────────────────

export const DustParticles: React.FC = () => {
  const count = 5000;
  const positions = useMemo(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 50;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 50;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 50;
    }
    return pos;
  }, [count]);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [positions]);

  const ref = useRef<THREE.Points>(null);
  useFrame((state, delta) => {
    if(ref.current){
      ref.current.rotation.y += delta * 0.02;
    }
  });
  
  return (
    <points ref={ref} geometry={geometry}>
      <pointsMaterial 
         color="#ffffff" 
         size={0.05} 
         transparent 
         opacity={0.3} 
         depthWrite={false}
      />
    </points>
  );
};

//────────────────────────────
// 5. Custom Post-Effect: WaterFlowPass (強い水面の揺らぎ＋虹色収差)
//────────────────────────────

const WaterFlowShader = {
  uniforms: {
    tDiffuse: { value: null },
    uTime: { value: 0 },
    uFrequency: { value: 20.0 },
    uAmplitude: { value: 0.05 },
    uSpeed: { value: 3.0 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main(){
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    precision mediump float;
    uniform sampler2D tDiffuse;
    uniform float uTime;
    uniform float uFrequency;
    uniform float uAmplitude;
    uniform float uSpeed;
    varying vec2 vUv;
    void main(){
      vec2 uv = vUv;
      uv += uAmplitude * vec2(
        sin(uv.y * uFrequency + uTime * uSpeed),
        cos(uv.x * uFrequency + uTime * uSpeed)
      );
      vec2 redOffset = vec2(0.005, 0.0);
      vec2 greenOffset = vec2(0.0, 0.005);
      vec2 blueOffset = vec2(-0.005, 0.0);
      float r = texture2D(tDiffuse, uv + redOffset).r;
      float g = texture2D(tDiffuse, uv + greenOffset).g;
      float b = texture2D(tDiffuse, uv + blueOffset).b;
      gl_FragColor = vec4(r, g, b, 1.0);
    }
  `
};

const WaterFlowPass = forwardRef((props, ref) => {
  const pass = useMemo(() => new ShaderPass(WaterFlowShader), []);
  useFrame((state) => {
    pass.uniforms.uTime.value = state.clock.getElapsedTime();
  });
  pass.renderToScreen = true;
  return <primitive ref={ref} object={pass} dispose={null} />;
});

//────────────────────────────
// 6. ASMR3DModel Component
//────────────────────────────

export type ASMR3DModelProps = {
  azimuth: number;
  elevation: number;
  filePath: string;
  getCloser: boolean;
  whisper: boolean;
  sourceNoise: string;
};

export const ASMR3DModel: React.FC<ASMR3DModelProps> = ({
  azimuth,
  elevation,
  filePath,
  getCloser,
  whisper,
  sourceNoise
}) => {
  const mainSphereColor = "yellowgreen";
  const baseDistance = 3;
  const radAzim = THREE.MathUtils.degToRad(azimuth);
  const radElev = THREE.MathUtils.degToRad(elevation);
  
  let targetSatellitePos: THREE.Vector3;
  if (getCloser) {
    targetSatellitePos = new THREE.Vector3(
      baseDistance * Math.cos(radElev) * Math.sin(radAzim),
      baseDistance * Math.sin(radElev),
      baseDistance * Math.cos(radElev) * Math.cos(radAzim)
    );
  } else {
    targetSatellitePos = new THREE.Vector3(
      baseDistance * 1.2 * Math.cos(radElev) * Math.sin(radAzim),
      baseDistance * 1.2 * Math.sin(radElev),
      baseDistance * 1.2 * Math.cos(radElev) * Math.cos(radAzim)
    );
  }
  
  const initialSatellitePos = new THREE.Vector3().copy(targetSatellitePos);
  
  let satelliteSphereColor = "";
  if (filePath.trim().length === 0) {
    satelliteSphereColor = "#808080";
  } else {
    const noiseLower = sourceNoise.toLowerCase();
    if (noiseLower === "white") {
      satelliteSphereColor = "#ffffff";
    } else if (noiseLower === "pink") {
      satelliteSphereColor = "#ffc0cb";
    } else if (noiseLower === "velvet") {
      satelliteSphereColor = "#800080";
    } else {
      satelliteSphereColor = "#800080";
    }
  }
  
  return (
    <div className="nm-flat-gray-200-lg" style={{ width: '100%', height: '100%', borderRadius: '20px' }}>
      <Canvas
        shadows
        camera={{ position: [0, 0, 15], fov: 40 }}
        style={{
          width: '100%',
          height: '100%',
          borderRadius: '20px',
          background: '#edf2f7'
        }}
      >
        <ambientLight intensity={0.3} />
        <directionalLight 
          castShadow 
          position={[10, 10, 10]} 
          intensity={0.8}
          shadow-mapSize-width={1024}
          shadow-mapSize-height={1024}
        />
        <DustParticles />
        <mesh 
          receiveShadow 
          rotation={[-Math.PI / 2, 0, 0]} 
          position={[0, -2.5, 0]}
        >
          <planeGeometry args={[200, 200]} />
          <meshStandardMaterial color="#edf2f7" />
        </mesh>
        <DistortedSphere
          radius={2}
          color={mainSphereColor}
          fresnelPower={4.0}
          chromaticAberration={0.15}
          stopRotation={!whisper}
          position={[0, 0, 0]}
        />
        <SatelliteSphere
          radius={0.7}
          color={satelliteSphereColor}
          fresnelPower={3.5}
          chromaticAberration={0.2}
          alpha={0.85}
          stopRotation={!whisper}
          initialPosition={initialSatellitePos}
          targetPosition={targetSatellitePos}
        />
        <EffectComposer>
          <Bloom luminanceThreshold={0.2} luminanceSmoothing={0.5} intensity={1.5} />
          <ChromaticAberration blendFunction={BlendFunction.ADD} offset={[0.0001, 0.0001]} />
          <WaterFlowPass />
        </EffectComposer>
      </Canvas>
    </div>
  );
};

export default ASMR3DModel;
