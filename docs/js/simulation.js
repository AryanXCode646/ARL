/**
 * ARL - Adaptive Reinforcement Learning
 * Interactive 3D Drone Flight Arena Simulation (Three.js)
 *
 * NOTE: This is a client-side browser demonstration of the 3D translational
 * kinematics and 16-ray LiDAR geometry within the 30m x 30m x 15m arena.
 * It is labeled honestly as an interactive conceptual visualization.
 */

(function () {
  'use strict';

  // Arena physical constants matching src/adaptive_rl/environments/drone.py
  const ARENA_X = 30.0;
  const ARENA_Y = 30.0;
  const ARENA_Z = 15.0;
  const DT = 0.1;
  const MAX_SPEED = 8.0;
  const MAX_ACCEL = 4.0;
  const DRAG_COEFF = 0.05;
  const TARGET_RADIUS = 1.5;
  const COLLISION_RADIUS = 0.8;
  const LIDAR_MAX_RANGE = 20.0;
  const MAX_STEPS = 200;

  // 16 spherical unit vectors for simulated LiDAR rays
  const LIDAR_RAYS = [];
  const NUM_RAYS = 16;
  for (let i = 0; i < NUM_RAYS; i++) {
    const phi = Math.acos(1 - 2 * (i + 0.5) / NUM_RAYS);
    const theta = Math.PI * (1 + Math.sqrt(5)) * (i + 0.5);
    LIDAR_RAYS.push(new THREE.Vector3(
      Math.sin(phi) * Math.cos(theta),
      Math.sin(phi) * Math.sin(theta),
      Math.cos(phi)
    ).normalize());
  }

  // Pre-configured obstacles (radius 2.0m)
  const OBSTACLES = [
    { x: 10.0, y: 10.0, z: 5.0, r: 2.0 },
    { x: 18.0, y: 12.0, z: 8.0, r: 2.0 },
    { x: 12.0, y: 20.0, z: 7.0, r: 2.0 },
    { x: 20.0, y: 22.0, z: 9.0, r: 2.0 }
  ];

  // Simulation State
  let simState = {
    isRunning: false,
    step: 0,
    speedMultiplier: 1.0,
    pos: new THREE.Vector3(5.0, 5.0, 3.0),
    vel: new THREE.Vector3(0.0, 0.0, 0.0),
    target: new THREE.Vector3(25.0, 25.0, 10.0),
    cumulativeReward: 0.0,
    status: 'READY',
    trajectory: []
  };

  let scene, camera, renderer, container;
  let droneMesh, targetMesh, obstacleMeshes = [], lidarLineSegments;
  let trajectoryLine;
  let animFrameId = null;

  function init() {
    container = document.getElementById('hero-sim-canvas-wrap');
    if (!container || typeof THREE === 'undefined') return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 380;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060a14);
    scene.fog = new THREE.FogExp2(0x060a14, 0.015);

    // Camera
    camera = new THREE.PerspectiveCamera(45, width / height, 0.5, 150);
    camera.position.set(38, -25, 28);
    camera.up.set(0, 0, 1);
    camera.lookAt(new THREE.Vector3(ARENA_X / 2, ARENA_Y / 2, ARENA_Z / 2));

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0x00f2fe, 0.9);
    dirLight.position.set(15, 15, 30);
    scene.add(dirLight);

    const blueLight = new THREE.PointLight(0x3b82f6, 1.2, 50);
    blueLight.position.set(5, 5, 15);
    scene.add(blueLight);

    buildArena();
    buildObstacles();
    buildTarget();
    buildDrone();
    buildLidarLines();
    buildTrajectory();
    bindEvents();
    updateHUD();
    render();
  }

  function buildArena() {
    // Ground Grid
    const grid = new THREE.GridHelper(ARENA_X, 30, 0x00f2fe, 0x1e293b);
    grid.rotation.x = Math.PI / 2;
    grid.position.set(ARENA_X / 2, ARENA_Y / 2, 0);
    scene.add(grid);

    // Bounding Box
    const boxGeom = new THREE.BoxGeometry(ARENA_X, ARENA_Y, ARENA_Z);
    const boxEdges = new THREE.EdgesGeometry(boxGeom);
    const boxMat = new THREE.LineBasicMaterial({ color: 0x1e293b, transparent: true, opacity: 0.6 });
    const box = new THREE.LineSegments(boxEdges, boxMat);
    box.position.set(ARENA_X / 2, ARENA_Y / 2, ARENA_Z / 2);
    scene.add(box);
  }

  function buildObstacles() {
    OBSTACLES.forEach(obs => {
      const geom = new THREE.SphereGeometry(obs.r, 24, 24);
      const mat = new THREE.MeshStandardMaterial({
        color: 0xef4444,
        roughness: 0.4,
        metalness: 0.3,
        transparent: true,
        opacity: 0.82,
        wireframe: false
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(obs.x, obs.y, obs.z);
      scene.add(mesh);

      // Wireframe aura
      const wireGeom = new THREE.WireframeGeometry(geom);
      const wireMat = new THREE.LineBasicMaterial({ color: 0xff7b7b, transparent: true, opacity: 0.3 });
      const wireMesh = new THREE.LineSegments(wireGeom, wireMat);
      mesh.add(wireMesh);

      // Safety buffer ring
      const ringGeom = new THREE.RingGeometry(obs.r + 0.1, obs.r + COLLISION_RADIUS, 32);
      const ringMat = new THREE.MeshBasicMaterial({ color: 0xff3333, side: THREE.DoubleSide, transparent: true, opacity: 0.15 });
      const ring = new THREE.Mesh(ringGeom, ringMat);
      mesh.add(ring);

      obstacleMeshes.push(mesh);
    });
  }

  function buildTarget() {
    const group = new THREE.Group();
    const geom = new THREE.OctahedronGeometry(TARGET_RADIUS * 0.8, 0);
    const mat = new THREE.MeshStandardMaterial({
      color: 0x10b981,
      emissive: 0x10b981,
      emissiveIntensity: 0.6,
      wireframe: false
    });
    targetMesh = new THREE.Mesh(geom, mat);
    group.add(targetMesh);

    // Vertical beacon line to ground
    const lineGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, -simState.target.z)
    ]);
    const lineMat = new THREE.LineDashedMaterial({ color: 0x10b981, dashSize: 0.5, gapSize: 0.5, transparent: true, opacity: 0.4 });
    const beaconLine = new THREE.Line(lineGeom, lineMat);
    beaconLine.computeLineDistances();
    group.add(beaconLine);

    group.position.copy(simState.target);
    scene.add(group);
  }

  function buildDrone() {
    droneMesh = new THREE.Group();

    // Central Carbon Body
    const bodyGeom = new THREE.CylinderGeometry(0.4, 0.4, 0.2, 16);
    bodyGeom.rotateX(Math.PI / 2);
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.8, roughness: 0.2 });
    const body = new THREE.Mesh(bodyGeom, bodyMat);
    droneMesh.add(body);

    // Quad Arms
    const armGeom = new THREE.BoxGeometry(1.6, 0.08, 0.08);
    const armMat = new THREE.MeshStandardMaterial({ color: 0x334155 });
    const arm1 = new THREE.Mesh(armGeom, armMat);
    arm1.rotation.z = Math.PI / 4;
    const arm2 = new THREE.Mesh(armGeom, armMat);
    arm2.rotation.z = -Math.PI / 4;
    droneMesh.add(arm1);
    droneMesh.add(arm2);

    // 4 Rotors
    const offsets = [
      { x: 0.56, y: 0.56 },
      { x: -0.56, y: 0.56 },
      { x: 0.56, y: -0.56 },
      { x: -0.56, y: -0.56 }
    ];

    offsets.forEach(off => {
      const rotorGeom = new THREE.CylinderGeometry(0.3, 0.3, 0.02, 16);
      rotorGeom.rotateX(Math.PI / 2);
      const rotorMat = new THREE.MeshBasicMaterial({ color: 0x00f2fe, transparent: true, opacity: 0.5 });
      const rotor = new THREE.Mesh(rotorGeom, rotorMat);
      rotor.position.set(off.x, off.y, 0.12);
      droneMesh.add(rotor);
    });

    // Nose indicator
    const noseGeom = new THREE.ConeGeometry(0.15, 0.3, 8);
    noseGeom.rotateZ(-Math.PI / 2);
    const noseMat = new THREE.MeshBasicMaterial({ color: 0x3b82f6 });
    const nose = new THREE.Mesh(noseGeom, noseMat);
    nose.position.set(0.45, 0, 0);
    droneMesh.add(nose);

    droneMesh.position.copy(simState.pos);
    scene.add(droneMesh);
  }

  function buildLidarLines() {
    const positions = new Float32Array(NUM_RAYS * 2 * 3);
    const colors = new Float32Array(NUM_RAYS * 2 * 3);

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.7,
      linewidth: 1
    });

    lidarLineSegments = new THREE.LineSegments(geom, mat);
    scene.add(lidarLineSegments);
  }

  function buildTrajectory() {
    const maxPoints = MAX_STEPS + 10;
    const positions = new Float32Array(maxPoints * 3);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setDrawRange(0, 0);

    const mat = new THREE.LineBasicMaterial({
      color: 0x00f2fe,
      transparent: true,
      opacity: 0.85,
      linewidth: 2
    });

    trajectoryLine = new THREE.Line(geom, mat);
    scene.add(trajectoryLine);
  }

  function computeLidar(pos) {
    const readings = [];
    const positions = lidarLineSegments.geometry.attributes.position.array;
    const colors = lidarLineSegments.geometry.attributes.color.array;

    for (let i = 0; i < NUM_RAYS; i++) {
      const rayDir = LIDAR_RAYS[i];
      let minDist = LIDAR_MAX_RANGE;

      // Obstacle intersections
      for (let j = 0; j < OBSTACLES.length; j++) {
        const obs = OBSTACLES[j];
        const oc = new THREE.Vector3(pos.x - obs.x, pos.y - obs.y, pos.z - obs.z);
        const b = 2.0 * oc.dot(rayDir);
        const c = oc.lengthSq() - obs.r * obs.r;
        const disc = b * b - 4 * c;
        if (disc >= 0) {
          const t1 = (-b - Math.sqrt(disc)) / 2.0;
          if (t1 > 0 && t1 < minDist) {
            minDist = t1;
          }
        }
      }

      // Arena boundary limits
      if (rayDir.x > 0) minDist = Math.min(minDist, (ARENA_X - pos.x) / rayDir.x);
      else if (rayDir.x < 0) minDist = Math.min(minDist, -pos.x / rayDir.x);

      if (rayDir.y > 0) minDist = Math.min(minDist, (ARENA_Y - pos.y) / rayDir.y);
      else if (rayDir.y < 0) minDist = Math.min(minDist, -pos.y / rayDir.y);

      if (rayDir.z > 0) minDist = Math.min(minDist, (ARENA_Z - pos.z) / rayDir.z);
      else if (rayDir.z < 0) minDist = Math.min(minDist, -pos.z / rayDir.z);

      readings.push(minDist);

      // Update lines
      const idx = i * 6;
      positions[idx] = pos.x;
      positions[idx + 1] = pos.y;
      positions[idx + 2] = pos.z;

      const hitX = pos.x + rayDir.x * minDist;
      const hitY = pos.y + rayDir.y * minDist;
      const hitZ = pos.z + rayDir.z * minDist;

      positions[idx + 3] = hitX;
      positions[idx + 4] = hitY;
      positions[idx + 5] = hitZ;

      // Color coding: Red if close, Cyan if open
      const normDist = minDist / LIDAR_MAX_RANGE;
      if (normDist < 0.2) {
        colors[idx + 3] = 1.0; colors[idx + 4] = 0.2; colors[idx + 5] = 0.2;
      } else if (normDist < 0.45) {
        colors[idx + 3] = 0.95; colors[idx + 4] = 0.7; colors[idx + 5] = 0.1;
      } else {
        colors[idx + 3] = 0.0; colors[idx + 4] = 0.95; colors[idx + 5] = 1.0;
      }
    }

    lidarLineSegments.geometry.attributes.position.needsUpdate = true;
    lidarLineSegments.geometry.attributes.color.needsUpdate = true;
    return readings;
  }

  function stepSim() {
    if (!simState.isRunning) return;

    simState.step += 1;

    // Guidance law: Attractive target field + repulsive obstacle avoidance
    const toGoal = new THREE.Vector3().subVectors(simState.target, simState.pos);
    const distToGoal = toGoal.length();

    // Goal attraction
    const desiredVel = toGoal.clone().normalize().multiplyScalar(Math.min(distToGoal, MAX_SPEED));
    const accel = new THREE.Vector3().subVectors(desiredVel, simState.vel).multiplyScalar(2.0);

    // Obstacle repulsion
    OBSTACLES.forEach(obs => {
      const obsPos = new THREE.Vector3(obs.x, obs.y, obs.z);
      const toObs = new THREE.Vector3().subVectors(simState.pos, obsPos);
      const dist = toObs.length();
      const safeDist = obs.r + 2.5;
      if (dist < safeDist) {
        const repForce = toObs.normalize().multiplyScalar((safeDist - dist) * 12.0);
        accel.add(repForce);
      }
    });

    // Boundary repulsion
    if (simState.pos.x < 3.0) accel.x += (3.0 - simState.pos.x) * 6.0;
    if (simState.pos.x > ARENA_X - 3.0) accel.x -= (simState.pos.x - (ARENA_X - 3.0)) * 6.0;
    if (simState.pos.y < 3.0) accel.y += (3.0 - simState.pos.y) * 6.0;
    if (simState.pos.y > ARENA_Y - 3.0) accel.y -= (simState.pos.y - (ARENA_Y - 3.0)) * 6.0;
    if (simState.pos.z < 2.0) accel.z += (2.0 - simState.pos.z) * 8.0;
    if (simState.pos.z > ARENA_Z - 2.0) accel.z -= (simState.pos.z - (ARENA_Z - 2.0)) * 8.0;

    // Clamp acceleration
    if (accel.length() > MAX_ACCEL) {
      accel.normalize().multiplyScalar(MAX_ACCEL);
    }

    // Kinematic integration with linear drag damping
    simState.vel.addScaledVector(accel, DT);
    simState.vel.addScaledVector(simState.vel, -DRAG_COEFF);
    if (simState.vel.length() > MAX_SPEED) {
      simState.vel.normalize().multiplyScalar(MAX_SPEED);
    }
    simState.pos.addScaledVector(simState.vel, DT);

    // Tilt drone mesh according to acceleration
    droneMesh.position.copy(simState.pos);
    droneMesh.rotation.z = Math.atan2(simState.vel.y, simState.vel.x);
    droneMesh.rotation.x = -accel.y * 0.08;
    droneMesh.rotation.y = accel.x * 0.08;

    // Record trajectory
    simState.trajectory.push(simState.pos.clone());
    updateTrajectoryLine();

    // Compute LiDAR
    const lidarReadings = computeLidar(simState.pos);
    const nearestObstacleDist = Math.min(...lidarReadings);

    // Reward calculation
    const progress = (distToGoal - toGoal.length()) * 10.0;
    const stepPenalty = -0.05;
    const stepReward = progress + stepPenalty;
    simState.cumulativeReward += stepReward;

    // Collision check
    let collided = false;
    for (let i = 0; i < OBSTACLES.length; i++) {
      const obs = OBSTACLES[i];
      const d = simState.pos.distanceTo(new THREE.Vector3(obs.x, obs.y, obs.z));
      if (d <= obs.r + COLLISION_RADIUS) {
        collided = true;
        break;
      }
    }
    if (
      simState.pos.x <= COLLISION_RADIUS || simState.pos.x >= ARENA_X - COLLISION_RADIUS ||
      simState.pos.y <= COLLISION_RADIUS || simState.pos.y >= ARENA_Y - COLLISION_RADIUS ||
      simState.pos.z <= COLLISION_RADIUS || simState.pos.z >= ARENA_Z - COLLISION_RADIUS
    ) {
      collided = true;
    }

    // Terminal Conditions
    if (distToGoal <= TARGET_RADIUS) {
      simState.isRunning = false;
      simState.status = 'SUCCESS (GOAL REACHED)';
      simState.cumulativeReward += 100.0;
    } else if (collided) {
      simState.isRunning = false;
      simState.status = 'COLLISION DETECTED';
      simState.cumulativeReward -= 100.0;
    } else if (simState.step >= MAX_STEPS) {
      simState.isRunning = false;
      simState.status = 'TRUNCATED (MAX STEPS)';
    } else {
      simState.status = 'NAVIGATING';
    }

    updateHUD();
  }

  function updateTrajectoryLine() {
    const positions = trajectoryLine.geometry.attributes.position.array;
    const count = simState.trajectory.length;
    for (let i = 0; i < count; i++) {
      const p = simState.trajectory[i];
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
    }
    trajectoryLine.geometry.setDrawRange(0, count);
    trajectoryLine.geometry.attributes.position.needsUpdate = true;
  }

  function updateHUD() {
    const stepEl = document.getElementById('hud-step');
    const posEl = document.getElementById('hud-pos');
    const velEl = document.getElementById('hud-vel');
    const distEl = document.getElementById('hud-dist');
    const rewEl = document.getElementById('hud-reward');
    const statusEl = document.getElementById('sim-status-text');
    const scrubEl = document.getElementById('sim-scrubber');

    const dist = simState.pos.distanceTo(simState.target).toFixed(1);
    const speed = simState.vel.length().toFixed(1);

    if (stepEl) stepEl.textContent = `Step ${simState.step} / ${MAX_STEPS}`;
    if (posEl) posEl.textContent = `(${simState.pos.x.toFixed(1)}, ${simState.pos.y.toFixed(1)}, ${simState.pos.z.toFixed(1)})`;
    if (velEl) velEl.textContent = `${speed} m/s`;
    if (distEl) distEl.textContent = `${dist} m`;
    if (rewEl) rewEl.textContent = simState.cumulativeReward.toFixed(1);
    if (statusEl) statusEl.textContent = simState.status;
    if (scrubEl) scrubEl.value = simState.step;
  }

  function resetSim() {
    simState.isRunning = false;
    simState.step = 0;
    simState.pos.set(5.0, 5.0, 3.0);
    simState.vel.set(0.0, 0.0, 0.0);
    simState.cumulativeReward = 0.0;
    simState.status = 'READY';
    simState.trajectory = [simState.pos.clone()];

    droneMesh.position.copy(simState.pos);
    droneMesh.rotation.set(0, 0, 0);

    updateTrajectoryLine();
    computeLidar(simState.pos);
    updateHUD();

    const startBtn = document.getElementById('btn-sim-start');
    if (startBtn) startBtn.textContent = 'START';
  }

  function bindEvents() {
    const startBtn = document.getElementById('btn-sim-start');
    const pauseBtn = document.getElementById('btn-sim-pause');
    const resetBtn = document.getElementById('btn-sim-reset');
    const scrub = document.getElementById('sim-scrubber');

    if (startBtn) {
      startBtn.addEventListener('click', () => {
        if (simState.status.includes('SUCCESS') || simState.status.includes('COLLISION') || simState.step >= MAX_STEPS) {
          resetSim();
        }
        simState.isRunning = true;
        simState.status = 'NAVIGATING';
        startBtn.textContent = 'RESUME';
      });
    }

    if (pauseBtn) {
      pauseBtn.addEventListener('click', () => {
        simState.isRunning = false;
        simState.status = 'PAUSED';
        updateHUD();
      });
    }

    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        resetSim();
      });
    }

    // Orbit Camera Drag
    let isDragging = false;
    let prevMousePos = { x: 0, y: 0 };
    let spherical = { radius: 52, theta: 0.8, phi: 1.1 };

    renderer.domElement.addEventListener('mousedown', (e) => {
      isDragging = true;
      prevMousePos = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener('mouseup', () => { isDragging = false; });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      const deltaX = e.clientX - prevMousePos.x;
      const deltaY = e.clientY - prevMousePos.y;
      prevMousePos = { x: e.clientX, y: e.clientY };

      spherical.theta -= deltaX * 0.008;
      spherical.phi = Math.max(0.2, Math.min(Math.PI / 2 - 0.05, spherical.phi - deltaY * 0.008));

      updateCameraPosition();
    });

    function updateCameraPosition() {
      const cx = ARENA_X / 2 + spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
      const cy = ARENA_Y / 2 + spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
      const cz = ARENA_Z / 2 + spherical.radius * Math.cos(spherical.phi);
      camera.position.set(cx, cy, cz);
      camera.lookAt(new THREE.Vector3(ARENA_X / 2, ARENA_Y / 2, ARENA_Z / 2));
    }

    window.addEventListener('resize', () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
  }

  let lastStepTime = 0;
  function render(time) {
    animFrameId = requestAnimationFrame(render);

    // Pulsing target beacon animation
    if (targetMesh) {
      targetMesh.rotation.y += 0.02;
      targetMesh.rotation.x += 0.01;
      const scale = 1.0 + Math.sin(time * 0.005) * 0.15;
      targetMesh.scale.set(scale, scale, scale);
    }

    // Run physics steps based on timer
    if (time - lastStepTime > (100 / simState.speedMultiplier)) {
      if (simState.isRunning) {
        stepSim();
      }
      lastStepTime = time;
    }

    renderer.render(scene, camera);
  }

  // Self-initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
