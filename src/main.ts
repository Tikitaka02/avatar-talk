import {
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

const video = document.getElementById("webcam") as HTMLVideoElement;
const canvas = document.getElementById("overlay") as HTMLCanvasElement;
const statusEl = document.getElementById("status") as HTMLDivElement;
const fpsEl = document.getElementById("fps") as HTMLSpanElement;
const mirrorEl = document.getElementById("mirror") as HTMLInputElement;
const stage = document.getElementById("stage") as HTMLDivElement;
const ctx = canvas.getContext("2d")!;

// Upper body only: landmarks 0–24 (face, arms, torso). Legs (25–32) are
// unreliable from a desk webcam, so we never draw or use them.
const UPPER_BODY_CUTOFF = 25;

// Connection pairs limited to the upper body subset of MediaPipe's pose
// topology: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
const UPPER_BODY_CONNECTIONS: [number, number][] = [
  // face outline
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
  // torso
  [11, 12], [11, 23], [12, 24], [23, 24],
  // left arm + hand
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  // right arm + hand
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
];

const MIN_VISIBILITY = 0.5;

let landmarker: PoseLandmarker | undefined;
let lastVideoTime = -1;

// Simple moving-average FPS counter
let frameCount = 0;
let fpsWindowStart = performance.now();

mirrorEl.addEventListener("change", () => {
  stage.classList.toggle("mirrored", mirrorEl.checked);
});
stage.classList.toggle("mirrored", mirrorEl.checked);

async function createLandmarker(): Promise<PoseLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

async function startWebcam(): Promise<void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise<void>((resolve) => {
    video.onloadedmetadata = () => resolve();
  });
}

function drawLandmarks(landmarks: NormalizedLandmark[]): void {
  const w = canvas.width;
  const h = canvas.height;

  ctx.lineWidth = 3;
  ctx.strokeStyle = "#4ade80";
  for (const [a, b] of UPPER_BODY_CONNECTIONS) {
    const la = landmarks[a];
    const lb = landmarks[b];
    if ((la.visibility ?? 1) < MIN_VISIBILITY) continue;
    if ((lb.visibility ?? 1) < MIN_VISIBILITY) continue;
    ctx.beginPath();
    ctx.moveTo(la.x * w, la.y * h);
    ctx.lineTo(lb.x * w, lb.y * h);
    ctx.stroke();
  }

  ctx.fillStyle = "#fb7185";
  for (let i = 0; i < UPPER_BODY_CUTOFF; i++) {
    const lm = landmarks[i];
    if ((lm.visibility ?? 1) < MIN_VISIBILITY) continue;
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function renderLoop(): void {
  if (!landmarker) return;

  // The overlay canvas must match the video's intrinsic resolution so that
  // normalized landmark coordinates line up with the pixels underneath.
  if (
    canvas.width !== video.videoWidth ||
    canvas.height !== video.videoHeight
  ) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  // Only run inference when the video has a new frame.
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (result.landmarks.length > 0) {
      drawLandmarks(result.landmarks[0]);
    }

    frameCount++;
    const elapsed = performance.now() - fpsWindowStart;
    if (elapsed >= 1000) {
      fpsEl.textContent = `${((frameCount * 1000) / elapsed).toFixed(0)} fps`;
      frameCount = 0;
      fpsWindowStart = performance.now();
    }
  }

  requestAnimationFrame(renderLoop);
}

async function main(): Promise<void> {
  try {
    statusEl.textContent = "Loading pose model…";
    landmarker = await createLandmarker();

    statusEl.textContent = "Requesting webcam…";
    await startWebcam();

    statusEl.classList.add("hidden");
    renderLoop();
  } catch (err) {
    statusEl.textContent =
      err instanceof DOMException && err.name === "NotAllowedError"
        ? "Webcam access was denied. Allow camera access and reload."
        : `Failed to start: ${err instanceof Error ? err.message : err}`;
  }
}

main();
