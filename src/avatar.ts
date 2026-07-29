import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { VRM, VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

export class AvatarViewer {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private loader: GLTFLoader;
  private elapsed = 0;

  vrm: VRM | undefined;
  /** Idle breathing; switched off while pose tracking drives the body. */
  idleMotion = true;

  constructor(private canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      // Keeps the rendered frame readable after drawing, so the canvas can be
      // screenshotted or recorded.
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.scene = new THREE.Scene();

    // Framed on the upper body, since that is all we track.
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
    this.camera.position.set(0, 1.3, 2.2);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 1.25, 0);
    this.controls.enablePan = false;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 5;
    this.controls.update();

    // Key light plus ambient fill — VRM materials are unlit-ish, so this is
    // mostly about giving the model some shape.
    const key = new THREE.DirectionalLight(0xffffff, 2);
    key.position.set(1, 2, 2);
    this.scene.add(key);
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.2));

    const grid = new THREE.GridHelper(4, 8, 0x2c2d33, 0x2c2d33);
    this.scene.add(grid);

    this.loader = new GLTFLoader();
    this.loader.register((parser) => new VRMLoaderPlugin(parser));

    this.resize();
  }

  async load(source: string | File): Promise<void> {
    const url = typeof source === "string" ? source : URL.createObjectURL(source);
    try {
      const gltf = await this.loader.loadAsync(url);
      const vrm = gltf.userData.vrm as VRM;

      // VRM 0.x models face +Z; normalize so every avatar faces the camera.
      VRMUtils.rotateVRM0(vrm);
      // Frees GPU memory and merges skeletons — meaningful on a 15 MB model.
      VRMUtils.removeUnnecessaryVertices(vrm.scene);
      VRMUtils.combineSkeletons(vrm.scene);

      if (this.vrm) {
        this.scene.remove(this.vrm.scene);
        VRMUtils.deepDispose(this.vrm.scene);
      }
      this.vrm = vrm;
      this.scene.add(vrm.scene);

      this.applyRestPose();
      this.frameUpperBody();
    } finally {
      if (typeof source !== "string") URL.revokeObjectURL(url);
    }
  }

  /**
   * VRM models are authored in a T-pose. Drop the arms into a relaxed A-pose
   * so the avatar looks natural before any tracking drives it.
   */
  private applyRestPose(): void {
    const humanoid = this.vrm?.humanoid;
    if (!humanoid) return;
    const left = humanoid.getNormalizedBoneNode("leftUpperArm");
    const right = humanoid.getNormalizedBoneNode("rightUpperArm");
    if (left) left.rotation.z = -1.15;
    if (right) right.rotation.z = 1.15;
    const leftLower = humanoid.getNormalizedBoneNode("leftLowerArm");
    const rightLower = humanoid.getNormalizedBoneNode("rightLowerArm");
    if (leftLower) leftLower.rotation.z = -0.2;
    if (rightLower) rightLower.rotation.z = 0.2;
    humanoid.update();
  }

  /**
   * Frames the camera from chest height to the very top of the model, using
   * its bounding box so tall hair or a hat is never clipped. Any dropped VRM
   * ends up composed the same way regardless of its height.
   */
  private frameUpperBody(): void {
    const vrm = this.vrm;
    const chest =
      vrm?.humanoid?.getNormalizedBoneNode("chest") ??
      vrm?.humanoid?.getNormalizedBoneNode("spine");
    if (!vrm || !chest) return;

    vrm.scene.updateWorldMatrix(true, true);
    const chestY = new THREE.Vector3().setFromMatrixPosition(chest.matrixWorld).y;
    const topY = new THREE.Box3().setFromObject(vrm.scene).max.y;

    const viewHeight = (topY - chestY) * 1.3;
    const centerY = (topY + chestY) / 2;
    const distance =
      viewHeight / 2 / Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));

    this.controls.target.set(0, centerY, 0);
    this.camera.position.set(0, centerY, distance);
    this.controls.update();
  }

  setExpression(name: string, value: number): void {
    this.vrm?.expressionManager?.setValue(name, value);
  }

  get expressionNames(): string[] {
    const manager = this.vrm?.expressionManager;
    if (!manager) return [];
    return manager.expressions.map((e) => e.expressionName);
  }

  update(delta: number): void {
    this.elapsed += delta;

    // Idle motion so the avatar reads as alive before retargeting exists:
    // a slow breathing rise in the chest and a small sway in the spine.
    const humanoid = this.idleMotion ? this.vrm?.humanoid : undefined;
    if (humanoid) {
      const breath = Math.sin(this.elapsed * 1.6);
      const chest = humanoid.getNormalizedBoneNode("chest");
      if (chest) chest.rotation.x = breath * 0.02;
      const spine = humanoid.getNormalizedBoneNode("spine");
      if (spine) spine.rotation.y = Math.sin(this.elapsed * 0.5) * 0.04;
    }

    this.controls.update();
    this.vrm?.update(delta);
    this.renderer.render(this.scene, this.camera);
  }

  resize(): void {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }
}
