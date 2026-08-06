import { PoseTracker } from "./tracking";
import { AvatarViewer } from "./avatar";
import { PoseRetargeter } from "./retarget";
import { FaceTracker } from "./face";
import { ExpressionDriver } from "./expression";
import { HandTracker, FingerRetargeter } from "./hands";
import { LandmarkFilter, OneEuroFilter } from "./filter";

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
const faceTracker = new FaceTracker(video);
const handTracker = new HandTracker(video);
const viewer = new AvatarViewer(avatarCanvas);
const retargeter = new PoseRetargeter();
const expressions = new ExpressionDriver();
const fingers = new FingerRetargeter();

// Jitter belongs to the measurement, so it is filtered at the source: every
// angle derived downstream inherits the fix, and the retargeting stays a pure
// function of the landmarks it is handed.
const poseFilter = new LandmarkFilter(() => new OneEuroFilter(1.2, 0.02));

/** Seconds without a usable pose before the avatar returns to its rest pose. */
const TRACKING_GRACE = 0.5;
let sinceTracked = Infinity;
let lastWorld: import("./tracking").TrackedPose["world"] | null = null;

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
    if (viewer.vrm) {
      retargeter.bind(viewer.vrm);
      expressions.bind(viewer.vrm);
      fingers.bind(viewer.vrm);
    }
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
// Optional readout: what the face model reports and what it maps to. A neutral
// face still produces numbers, so it shows the pipeline is alive even when the
// avatar's expression barely moves.
const readoutEl = document.getElementById("readout") as HTMLDivElement;
const showReadoutEl = document.getElementById("show-readout") as HTMLInputElement;
const WATCHED_SHAPES = ["jawOpen", "mouthSmileLeft", "eyeBlinkLeft", "browDownLeft"];
let readoutDue = 0;

showReadoutEl.addEventListener("change", () => {
  readoutEl.classList.toggle("hidden", !showReadoutEl.checked);
});

function bar(value: number): string {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
  return `<span class="bar"><span style="width:${pct}%"></span></span>`;
}

function updateReadout(face: import("./face").FaceResult | null, hands: number, now: number): void {
  if (!showReadoutEl.checked || now < readoutDue) return;
  readoutDue = now + 100; // 10 Hz is plenty, and keeps DOM work off the frame budget

  const rows: string[] = ['<div class="head">face model → blendshape</div>'];
  for (const key of WATCHED_SHAPES) {
    const v = face?.shapes.get(key) ?? 0;
    rows.push(`<div class="row"><span>${key}</span>${bar(v)}<span>${v.toFixed(2)}</span></div>`);
  }
  rows.push('<div class="head">→ VRM expression</div>');
  const driven = expressions.driven.filter(([, v]) => v > 0).slice(0, 4);
  if (driven.length === 0) rows.push('<div class="row"><span>(neutral)</span></div>');
  for (const [name, v] of driven) {
    rows.push(`<div class="row"><span>${name}</span>${bar(v)}<span>${v.toFixed(2)}</span></div>`);
  }
  rows.push(`<div class="head">hands tracked: ${hands}</div>`);
  readoutEl.innerHTML = rows.join("");
}

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
  const pose = tracker.update();
  if (pose) {
    // Filter before retargeting, not after: this is the measurement being
    // cleaned up, so everything downstream benefits.
    lastWorld = poseFilter.apply(pose.world, delta);
    sinceTracked = 0;
  } else {
    sinceTracked += delta;
  }

  // Face and hands are separate models on the same video frame.
  const face = faceTracker.update();
  const hands = handTracker.update();

  // The webcam delivers ~30 poses a second while the display refreshes faster,
  // so the last pose is re-applied every frame. That keeps the easing running
  // between camera frames instead of stepping once per new landmark set.
  const fresh = lastWorld !== null && sinceTracked < TRACKING_GRACE;
  const posed = fresh && retargeter.apply(lastWorld!, delta);
  if (!posed) retargeter.releaseToRest(delta);

  if (face) expressions.apply(face, delta);
  else expressions.release(delta);
  updateReadout(face, hands.length, now);

  // Fingers are posed after the arms: they are children of the hand, so the
  // arm rotation has to be settled first.
  if (!fingers.apply(hands, delta)) fingers.release(delta);

  // Breathing would fight the tracked spine, so it only runs when idle.
  viewer.idleMotion = !posed;
  viewer.update(delta);

  if (pose) {
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

    // Face and hands are independent of the camera, and nothing waits on them:
    // they download while the permission prompt is up, and start contributing
    // whenever they are ready. The body moves from the first pose either way.
    faceTracker.init().catch(() => {});
    handTracker.init().catch(() => {});

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
