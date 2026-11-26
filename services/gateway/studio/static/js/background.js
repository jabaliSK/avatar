// The animated backdrop: a slowly drifting network of points that reach towards the pointer. It is decoration, so it stops when the page is
// hidden, when the person prefers reduced motion (one still frame is drawn), and it draws fewer points on the conversation page.

const LINK_DISTANCE = 130;
const POINTER_DISTANCE = 190;

export function startBackground(canvas, { density = 1, maxFps = 30 } = {}) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return () => {};
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
  const pointer = { x: -9999, y: -9999 };
  let width = 0;
  let height = 0;
  let points = [];
  let frame = 0;
  let last = 0;

  function resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    const count = Math.min(90, Math.round((width * height) / 20000 * density));
    points = Array.from({ length: count }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.28,
      vy: (Math.random() - 0.5) * 0.28,
      r: Math.random() * 1.3 + 0.7,
      accent: Math.random() < 0.12,
    }));
    paint();
  }

  function step() {
    for (const p of points) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < -20) p.x = width + 20;
      else if (p.x > width + 20) p.x = -20;
      if (p.y < -20) p.y = height + 20;
      else if (p.y > height + 20) p.y = -20;
    }
  }

  function paint() {
    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = 1;
    for (let i = 0; i < points.length; i++) {
      const a = points[i];
      for (let j = i + 1; j < points.length; j++) {
        const b = points[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const distance = Math.hypot(dx, dy);
        if (distance > LINK_DISTANCE) continue;
        ctx.strokeStyle = `rgba(1, 169, 130, ${((1 - distance / LINK_DISTANCE) * 0.32).toFixed(3)})`;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
      const toPointer = Math.hypot(a.x - pointer.x, a.y - pointer.y);
      if (toPointer < POINTER_DISTANCE) {
        ctx.strokeStyle = `rgba(23, 235, 160, ${((1 - toPointer / POINTER_DISTANCE) * 0.6).toFixed(3)})`;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(pointer.x, pointer.y);
        ctx.stroke();
      }
    }
    for (const p of points) {
      ctx.fillStyle = p.accent ? "rgba(150, 102, 240, 0.85)" : "rgba(23, 235, 160, 0.8)";
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function loop(now) {
    frame = requestAnimationFrame(loop);
    if (now - last < 1000 / maxFps) return;
    last = now;
    step();
    paint();
  }

  function run() {
    cancelAnimationFrame(frame);
    if (!reduced.matches && !document.hidden) frame = requestAnimationFrame(loop);
  }

  const onPointer = (event) => { pointer.x = event.clientX; pointer.y = event.clientY; };
  const onLeave = () => { pointer.x = pointer.y = -9999; };
  window.addEventListener("resize", resize);
  window.addEventListener("pointermove", onPointer, { passive: true });
  document.addEventListener("pointerleave", onLeave);
  document.addEventListener("visibilitychange", run);
  reduced.addEventListener("change", run);
  resize();
  run();

  return () => {
    cancelAnimationFrame(frame);
    window.removeEventListener("resize", resize);
    window.removeEventListener("pointermove", onPointer);
    document.removeEventListener("pointerleave", onLeave);
    document.removeEventListener("visibilitychange", run);
    reduced.removeEventListener("change", run);
  };
}
