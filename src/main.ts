import { PoseTracker } from "./tracking";
import { AvatarViewer } from "./avatar";

const video = document.getElementById("webcam") as HTMLVideoElement;
const overlay = document.getElementById("overlay") as HTMLCanvasElement;
const avatarCanvas = document.getElementById("avatar") as HTMLCanvasElement;
const statusEl = document.getElementById("status") as HTMLDivElement;
const avatarStatusEl = document.getElementById("avatar-status") as HTMLDivElement;
const fpsEl = document.getElementById("fps") as HTMLSpanElement;
const mirrorEl = document.getElementById("mirror") as HTMLInputElement;
const stage = document.getElementById("stage") as HTMLDivElement;
const expressionsEl = document.getElementById("expressions") as HTMLDivElement;

const DEFAULT_VRM = "/avatar/test-avatar-talk.vrm";

// Expressions worth exposing as buttons, if the loaded avatar defines them.
const EXPRESSION_PRESETS = ["blink", "happy", "angry", "sad", "relaxed", "aa"];

const tracker = new PoseTracker(video, overlay);
const viewer = new AvatarViewer(avatarCanvas);

let frameCount = 0;
let fpsWindowStart = performance.now();
let lastFrame = performance.now();

mirrorEl.addEventListener("change", () => {
  stage.classList.toggle("mirrored", mirrorEl.checked);
});
stage.classList.toggle("mirrored", mirrorEl.checked);

window.addEventListener("resize", () => viewer.resize());

function buildExpressionButtons(): void {
  expressionsEl.innerHTML = "";
  const available = viewer.expressionNames;
  for (const name of EXPRESSION_PRESETS) {
    if (!available.includes(name)) continue;
    const btn = document.createElement("button");
    btn.textContent = name;
    // Hold to apply, release to relax — easier to demo than a toggle.
    const on = () => viewer.setExpression(name, 1);
    const off = () => viewer.setExpression(name, 0);
    btn.addEventListener("pointerdown", on);
    btn.addEventListener("pointerup", off);
    btn.addEventListener("pointerleave", off);
    expressionsEl.appendChild(btn);
  }
}

async function loadAvatar(source: string | File): Promise<void> {
  avatarStatusEl.textContent =
    typeof source === "string" ? "Loading avatar…" : `Loading ${source.name}…`;
  avatarStatusEl.classList.remove("hidden");
  try {
    await viewer.load(source);
    buildExpressionButtons();
    avatarStatusEl.classList.add("hidden");
  } catch (err) {
    // The bundled avatar is optional (it is not committed), so a missing
    // default is a prompt to bring your own rather than an error.
    avatarStatusEl.textContent =
      typeof source === "string"
        ? "No avatar loaded — drop a .vrm file here to begin."
        : `Could not load ${source.name}: ${err instanceof Error ? err.message : err}`;
  }
}

// Drag and drop any .vrm file onto the 3D pane to swap avatars.
const dropZone = document.getElementById("avatar-pane") as HTMLDivElement;
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragging");
  const file = e.dataTransfer?.files[0];
  if (file?.name.toLowerCase().endsWith(".vrm")) loadAvatar(file);
});

function renderLoop(): void {
  const now = performance.now();
  const delta = (now - lastFrame) / 1000;
  lastFrame = now;

  // One loop drives both halves: tracking only advances on new webcam frames,
  // while the 3D scene redraws every frame so the idle motion stays smooth.
  const landmarks = tracker.update();
  viewer.update(delta);

  if (landmarks) {
    frameCount++;
    const elapsed = now - fpsWindowStart;
    if (elapsed >= 1000) {
      fpsEl.textContent = `${((frameCount * 1000) / elapsed).toFixed(0)} fps`;
      frameCount = 0;
      fpsWindowStart = now;
    }
  }

  requestAnimationFrame(renderLoop);
}

async function main(): Promise<void> {
  // The avatar does not depend on the webcam, so load it and start drawing
  // before touching the camera. Both are awaited later than the render loop
  // on purpose: a slow permission prompt must not freeze the 3D pane.
  loadAvatar(DEFAULT_VRM);
  renderLoop();

  try {
    statusEl.textContent = "Loading pose model…";
    await tracker.init();

    statusEl.textContent = "Requesting webcam…";
    await tracker.startWebcam();

    statusEl.classList.add("hidden");
  } catch (err) {
    statusEl.textContent =
      err instanceof DOMException && err.name === "NotAllowedError"
        ? "Webcam access was denied. Allow camera access and reload."
        : `Failed to start: ${err instanceof Error ? err.message : err}`;
  }
}

main();
