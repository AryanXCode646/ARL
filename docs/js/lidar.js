/**
 * ARL - Adaptive Reinforcement Learning
 * Interactive 2D Sensor / LiDAR Inspector
 */

(function () {
  'use strict';

  const NUM_RAYS = 16;
  const MAX_RANGE = 20.0;
  const DRONE_RADIUS = 0.8;
  const ARENA_SIZE = 30.0;

  const obstacles = [
    { x: 10.0, y: 10.0, r: 2.0 },
    { x: 18.0, y: 12.0, r: 2.0 },
    { x: 12.0, y: 20.0, r: 2.0 },
    { x: 22.0, y: 22.0, r: 2.0 }
  ];

  let drone = { x: 8.0, y: 16.0 };
  let target = { x: 25.0, y: 25.0 };
  let canvas, ctx;
  let isDragging = false;

  function init() {
    canvas = document.getElementById('lidar-2d-canvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    canvas.addEventListener('mousedown', (e) => {
      const pos = getCanvasPos(e);
      const d = Math.hypot(pos.x - drone.x, pos.y - drone.y);
      if (d < 3.0) isDragging = true;
    });

    window.addEventListener('mouseup', () => { isDragging = false; });

    canvas.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      const pos = getCanvasPos(e);
      drone.x = Math.max(DRONE_RADIUS, Math.min(ARENA_SIZE - DRONE_RADIUS, pos.x));
      drone.y = Math.max(DRONE_RADIUS, Math.min(ARENA_SIZE - DRONE_RADIUS, pos.y));
      draw();
    });

    // Touch support for mobile
    canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length > 0) {
        const pos = getTouchPos(e.touches[0]);
        drone.x = Math.max(DRONE_RADIUS, Math.min(ARENA_SIZE - DRONE_RADIUS, pos.x));
        drone.y = Math.max(DRONE_RADIUS, Math.min(ARENA_SIZE - DRONE_RADIUS, pos.y));
        draw();
      }
    });

    canvas.addEventListener('touchmove', (e) => {
      if (e.touches.length > 0) {
        e.preventDefault();
        const pos = getTouchPos(e.touches[0]);
        drone.x = Math.max(DRONE_RADIUS, Math.min(ARENA_SIZE - DRONE_RADIUS, pos.x));
        drone.y = Math.max(DRONE_RADIUS, Math.min(ARENA_SIZE - DRONE_RADIUS, pos.y));
        draw();
      }
    });

    draw();
  }

  function resizeCanvas() {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * (window.devicePixelRatio || 1);
    canvas.height = rect.height * (window.devicePixelRatio || 1);
    draw();
  }

  function toScreen(valX, valY) {
    const scale = Math.min(canvas.width, canvas.height) / ARENA_SIZE;
    const offsetX = (canvas.width - ARENA_SIZE * scale) / 2;
    const offsetY = (canvas.height - ARENA_SIZE * scale) / 2;
    return {
      x: offsetX + valX * scale,
      y: offsetY + (ARENA_SIZE - valY) * scale,
      scale: scale
    };
  }

  function toWorld(screenX, screenY) {
    const scale = Math.min(canvas.width, canvas.height) / ARENA_SIZE;
    const offsetX = (canvas.width - ARENA_SIZE * scale) / 2;
    const offsetY = (canvas.height - ARENA_SIZE * scale) / 2;
    return {
      x: (screenX - offsetX) / scale,
      y: ARENA_SIZE - (screenY - offsetY) / scale
    };
  }

  function getCanvasPos(e) {
    const rect = canvas.getBoundingClientRect();
    const sx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const sy = (e.clientY - rect.top) * (canvas.height / rect.height);
    return toWorld(sx, sy);
  }

  function getTouchPos(t) {
    const rect = canvas.getBoundingClientRect();
    const sx = (t.clientX - rect.left) * (canvas.width / rect.width);
    const sy = (t.clientY - rect.top) * (canvas.height / rect.height);
    return toWorld(sx, sy);
  }

  function computeRays() {
    const readings = [];
    for (let i = 0; i < NUM_RAYS; i++) {
      const angle = (i * 2 * Math.PI) / NUM_RAYS;
      const dx = Math.cos(angle);
      const dy = Math.sin(angle);
      let dist = MAX_RANGE;

      // Obstacles raycast (ray-circle intersection)
      obstacles.forEach(obs => {
        const ox = drone.x - obs.x;
        const oy = drone.y - obs.y;
        const b = 2 * (ox * dx + oy * dy);
        const c = ox * ox + oy * oy - obs.r * obs.r;
        const disc = b * b - 4 * c;
        if (disc >= 0) {
          const t = (-b - Math.sqrt(disc)) / 2;
          if (t > 0 && t < dist) dist = t;
        }
      });

      // Arena walls
      if (dx > 0) dist = Math.min(dist, (ARENA_SIZE - drone.x) / dx);
      else if (dx < 0) dist = Math.min(dist, -drone.x / dx);

      if (dy > 0) dist = Math.min(dist, (ARENA_SIZE - drone.y) / dy);
      else if (dy < 0) dist = Math.min(dist, -drone.y / dy);

      readings.push({ dist: Math.min(dist, MAX_RANGE), angle, dx, dy });
    }
    return readings;
  }

  function draw() {
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw arena background grid
    const sOrigin = toScreen(0, 0);
    const sEnd = toScreen(ARENA_SIZE, ARENA_SIZE);
    const w = sEnd.x - sOrigin.x;
    const h = sOrigin.y - sEnd.y;

    ctx.fillStyle = '#060a14';
    ctx.fillRect(sOrigin.x, sEnd.y, w, h);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= ARENA_SIZE; i += 5) {
      const p1 = toScreen(i, 0);
      const p2 = toScreen(i, ARENA_SIZE);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();

      const q1 = toScreen(0, i);
      const q2 = toScreen(ARENA_SIZE, i);
      ctx.beginPath();
      ctx.moveTo(q1.x, q1.y);
      ctx.lineTo(q2.x, q2.y);
      ctx.stroke();
    }

    // Draw Arena Boundary Line
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.3)';
    ctx.lineWidth = 2;
    ctx.strokeRect(sOrigin.x, sEnd.y, w, h);

    // Draw Target Beacon
    const sTarget = toScreen(target.x, target.y);
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(sTarget.x, sTarget.y, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(sTarget.x, sTarget.y, 14, 0, Math.PI * 2);
    ctx.stroke();

    // Draw Obstacles
    obstacles.forEach(obs => {
      const sObs = toScreen(obs.x, obs.y);
      const rPix = obs.r * sObs.scale;

      // Hazard clearance buffer
      ctx.fillStyle = 'rgba(239, 68, 68, 0.12)';
      ctx.beginPath();
      ctx.arc(sObs.x, sObs.y, (obs.r + DRONE_RADIUS) * sObs.scale, 0, Math.PI * 2);
      ctx.fill();

      // Obstacle body
      ctx.fillStyle = 'rgba(239, 68, 68, 0.65)';
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sObs.x, sObs.y, rPix, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });

    // Compute LiDAR rays
    const rays = computeRays();
    const sDrone = toScreen(drone.x, drone.y);

    // Draw LiDAR Rays
    rays.forEach((r, idx) => {
      const hitX = drone.x + r.dx * r.dist;
      const hitY = drone.y + r.dy * r.dist;
      const sHit = toScreen(hitX, hitY);

      const normDist = r.dist / MAX_RANGE;
      let strokeColor = 'rgba(0, 242, 254, 0.65)';
      if (normDist < 0.2) strokeColor = 'rgba(239, 68, 68, 0.9)';
      else if (normDist < 0.45) strokeColor = 'rgba(245, 158, 11, 0.8)';

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(sDrone.x, sDrone.y);
      ctx.lineTo(sHit.x, sHit.y);
      ctx.stroke();

      // Hit Point Dot
      ctx.fillStyle = strokeColor;
      ctx.beginPath();
      ctx.arc(sHit.x, sHit.y, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw Drone Body
    ctx.fillStyle = '#0f172a';
    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(sDrone.x, sDrone.y, DRONE_RADIUS * sDrone.scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Cross arms
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sDrone.x - 12, sDrone.y - 12);
    ctx.lineTo(sDrone.x + 12, sDrone.y + 12);
    ctx.moveTo(sDrone.x - 12, sDrone.y + 12);
    ctx.lineTo(sDrone.x + 12, sDrone.y - 12);
    ctx.stroke();

    // Status text inside canvas
    ctx.font = '12px "JetBrains Mono", monospace';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Drag drone to test 16-ray LiDAR rangefinder in real-time', 15, 25);

    // Update Telemetry Panel
    updateTelemetry(rays);
  }

  function updateTelemetry(rays) {
    const minRay = Math.min(...rays.map(r => r.dist));
    const targetDist = Math.hypot(target.x - drone.x, target.y - drone.y);

    const nearestObsEl = document.getElementById('lidar-nearest-obs');
    const targetDistEl = document.getElementById('lidar-target-dist');
    const statusBadgeEl = document.getElementById('lidar-status-badge');

    if (nearestObsEl) nearestObsEl.textContent = `${minRay.toFixed(2)} m`;
    if (targetDistEl) targetDistEl.textContent = `${targetDist.toFixed(2)} m`;

    if (statusBadgeEl) {
      if (minRay <= DRONE_RADIUS) {
        statusBadgeEl.textContent = 'COLLISION DETECTED';
        statusBadgeEl.style.color = '#ef4444';
        statusBadgeEl.style.borderColor = 'rgba(239, 68, 68, 0.4)';
      } else if (minRay < 3.0) {
        statusBadgeEl.textContent = 'PROXIMITY WARNING (<3m)';
        statusBadgeEl.style.color = '#f59e0b';
        statusBadgeEl.style.borderColor = 'rgba(245, 158, 11, 0.4)';
      } else {
        statusBadgeEl.textContent = 'CLEAR TRAJECTORY';
        statusBadgeEl.style.color = '#10b981';
        statusBadgeEl.style.borderColor = 'rgba(16, 185, 129, 0.4)';
      }
    }

    // Update 16 Ray Badges
    const grid = document.getElementById('lidar-rays-grid');
    if (grid && grid.children.length === 0) {
      for (let i = 0; i < NUM_RAYS; i++) {
        const badge = document.createElement('div');
        badge.className = 'ray-badge';
        badge.id = `ray-badge-${i}`;
        badge.innerHTML = `<span class="ray-label">R${i}</span><span class="ray-val" id="ray-val-${i}">--</span>`;
        grid.appendChild(badge);
      }
    }

    for (let i = 0; i < NUM_RAYS; i++) {
      const valEl = document.getElementById(`ray-val-${i}`);
      if (valEl) {
        valEl.textContent = `${rays[i].dist.toFixed(1)}m`;
        const norm = rays[i].dist / MAX_RANGE;
        if (norm < 0.2) valEl.style.color = '#ef4444';
        else if (norm < 0.45) valEl.style.color = '#f59e0b';
        else valEl.style.color = '#00f2fe';
      }
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
