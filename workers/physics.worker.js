/* Physics runs entirely in this classic worker.
 * The main thread only sends tilt / hover / poke commands and receives a packed
 * Float32Array of [px,py,pz, qx,qy,qz,qw] for each item, transferred zero-copy.
 */

const STRIDE = 7;
const FIXED_STEP = 1 / 120;

let AmmoLib = null;
let world = null;
let items = [];
let pool = [];
let hoverIndex = -1;
let tiltX = 0;
let tiltZ = 0;
let running = false;
let lastTime = 0;
let tickTimer = null;
let initSpecs = [];
let pseudoForce = [0, 0, 0];

const vec = (a) => new AmmoLib.btVector3(a[0], a[1], a[2]);
const quat = (a) => new AmmoLib.btQuaternion(a[0], a[1], a[2], a[3]);

function qMul(a, b) {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
  ];
}

function qConj(q) {
  return [-q[0], -q[1], -q[2], q[3]];
}

function makeBody(spec, isStatic) {
  const mass = isStatic ? 0 : spec.mass;
  let shape;

  if (spec.shapes.length === 1) {
    shape = makeShape(spec.shapes[0]);
  } else {
    shape = new AmmoLib.btCompoundShape();
    for (const child of spec.shapes) {
      const transform = new AmmoLib.btTransform();
      transform.setIdentity();
      transform.setOrigin(vec(child.offset || [0, 0, 0]));
      transform.setRotation(quat(child.rotation || [0, 0, 0, 1]));
      shape.addChildShape(transform, makeShape(child));
    }
  }

  const inertia = new AmmoLib.btVector3(0, 0, 0);
  if (!isStatic) shape.calculateLocalInertia(mass, inertia);

  const transform = new AmmoLib.btTransform();
  transform.setIdentity();
  transform.setOrigin(vec(spec.position || [0, 0, 0]));
  transform.setRotation(quat(spec.rotation || [0, 0, 0, 1]));

  const motionState = new AmmoLib.btDefaultMotionState(transform);
  const info = new AmmoLib.btRigidBodyConstructionInfo(mass, motionState, shape, inertia);
  const body = new AmmoLib.btRigidBody(info);

  body.setFriction(isStatic ? 0.72 : 0.55);
  body.setRestitution(isStatic ? 0.02 : 0.04);

  if (!isStatic) {
    body.setDamping(0.14, 0.3);
    // Continuous collision detection: stops fast pokes from tunnelling through the wire walls.
    body.setCcdMotionThreshold(0.1);
    body.setCcdSweptSphereRadius(0.06);
  }

  world.addRigidBody(body);
  return body;
}

function makeShape(s) {
  if (s.type === 'cylinder') {
    // Ammo cylinders are Y-up; child rotations handle the camera lens.
    return new AmmoLib.btCylinderShape(new AmmoLib.btVector3(s.radius, s.height / 2, s.radius));
  }
  return new AmmoLib.btBoxShape(vec(s.half));
}

function buildBasket(basket) {
  const { halfX, halfZ, height } = basket;
  const t = 0.055;
  const hy = height / 2;
  const specs = [
    // floor
    { position: [0, -t, 0], shapes: [{ type: 'box', half: [halfX, t, halfZ] }] },
    // left / right
    { position: [-halfX - t, hy, 0], shapes: [{ type: 'box', half: [t, hy, halfZ] }] },
    { position: [halfX + t, hy, 0], shapes: [{ type: 'box', half: [t, hy, halfZ] }] },
    // front / back
    { position: [0, hy, -halfZ - t], shapes: [{ type: 'box', half: [halfX, hy, t] }] },
    { position: [0, hy, halfZ + t], shapes: [{ type: 'box', half: [halfX, hy, t] }] }
  ];
  for (const s of specs) makeBody(s, true);
}

function initAmmo(base) {
  if (AmmoLib) return Promise.resolve(AmmoLib);
  self.importScripts(base + 'ammo.wasm.js');
  const factory = self.Ammo;
  return Promise.resolve(factory({ locateFile: (path) => base + path })).then((lib) => {
    AmmoLib = lib;
    return lib;
  });
}

function buildWorld(message) {
  initSpecs = message.objects;
  const config = new AmmoLib.btDefaultCollisionConfiguration();
  world = new AmmoLib.btDiscreteDynamicsWorld(
    new AmmoLib.btCollisionDispatcher(config),
    new AmmoLib.btDbvtBroadphase(),
    new AmmoLib.btSequentialImpulseConstraintSolver(),
    config
  );
  buildBasket(message.basket);

  items = initSpecs.map((spec) => {
    const body = makeBody(spec, false);
    return { body, spec, home: spec.position.slice(), homeRotation: spec.rotation.slice() };
  });

  for (let i = 0; i < 4; i++) pool.push(new ArrayBuffer(items.length * STRIDE * 4));
  applyGravity();
}

function applyGravity() {
  const g = 9.8;
  const cx = Math.cos(tiltX);
  const sx = Math.sin(tiltX);
  const sz = Math.sin(tiltZ);
  const cz = Math.cos(tiltZ);
  // Inverse-rotate world gravity into the basket's local frame, so the basket
  // collider can stay axis-aligned and never needs to be re-built.
  world.setGravity(new AmmoLib.btVector3(-g * cx * sz, -g * cx * cz, g * sx));
}

function step() {
  const now = performance.now();
  let dt = (now - lastTime) / 1000;
  lastTime = now;
  if (dt > 0.05) dt = 0.05;

  applyGravity();

  if (pseudoForce[0] || pseudoForce[1] || pseudoForce[2]) {
    for (const item of items) {
      const m = item.spec.mass;
      item.body.applyCentralForce(new AmmoLib.btVector3(
        pseudoForce[0] * m,
        pseudoForce[1] * m,
        pseudoForce[2] * m
      ));
      item.body.activate();
    }
    pseudoForce = pseudoForce.map((v) => v * 0.86);
  }

  if (hoverIndex >= 0 && hoverIndex < items.length) {
    const item = items[hoverIndex];
    const body = item.body;
    const t = body.getWorldTransform();
    const rot = t.getRotation();
    const current = [rot.x(), rot.y(), rot.z(), rot.w()];
    // Target = identity (the model's front faces +Z / the viewer).
    const delta = qConj(current);
    let w = Math.min(1, Math.max(-1, delta[3]));
    const angle = 2 * Math.acos(w);
    const s = Math.sqrt(Math.max(0, 1 - w * w));
    let tx = 0, ty = 0, tz = 0;
    if (s > 1e-4 && angle > 0.002) {
      const gain = Math.min(angle * 3.2, 5);
      tx = (delta[0] / s) * gain;
      ty = (delta[1] / s) * gain;
      tz = (delta[2] / s) * gain;
    }
    const av = body.getAngularVelocity();
    const k = 0.09;
    body.setAngularVelocity(new AmmoLib.btVector3(
      av.x() + (tx - av.x()) * k,
      av.y() + (ty - av.y()) * k,
      av.z() + (tz - av.z()) * k
    ));
    body.applyCentralForce(new AmmoLib.btVector3(0, item.spec.mass * 2.4, 0));
    body.activate();
  }

  world.stepSimulation(dt, 8, FIXED_STEP);
  clampVelocities();
  postFrame();
  tickTimer = setTimeout(step, 8);
}

function clampVelocities() {
  const maxLinear = 4.5;
  const maxAngular = 7;
  for (const item of items) {
    const v = item.body.getLinearVelocity();
    const speed = Math.hypot(v.x(), v.y(), v.z());
    if (speed > maxLinear) {
      const scale = maxLinear / speed;
      item.body.setLinearVelocity(new AmmoLib.btVector3(v.x() * scale, v.y() * scale, v.z() * scale));
    }
    const av = item.body.getAngularVelocity();
    const spin = Math.hypot(av.x(), av.y(), av.z());
    if (spin > maxAngular) {
      const scale = maxAngular / spin;
      item.body.setAngularVelocity(new AmmoLib.btVector3(av.x() * scale, av.y() * scale, av.z() * scale));
    }
  }
}

function postFrame() {
  const buffer = pool.pop();
  if (!buffer) return; // main thread has not recycled yet — skip rather than allocate
  const out = new Float32Array(buffer);
  for (let i = 0; i < items.length; i++) {
    const t = items[i].body.getWorldTransform();
    const p = t.getOrigin();
    const q = t.getRotation();
    const o = i * STRIDE;
    out[o] = p.x(); out[o + 1] = p.y(); out[o + 2] = p.z();
    out[o + 3] = q.x(); out[o + 4] = q.y(); out[o + 5] = q.z(); out[o + 6] = q.w();
  }
  self.postMessage({ type: 'frame', buffer }, [buffer]);
}

function poke(message) {
  const item = items[message.index];
  if (!item) return;
  const body = item.body;
  const dir = message.dir || [1, 0.35, 0];
  const strength = message.strength || 0.9;
  body.applyImpulse(
    vec([dir[0] * strength, dir[1] * strength, dir[2] * strength]),
    new AmmoLib.btVector3(0, 0.12, 0)
  );
  body.applyTorqueImpulse(new AmmoLib.btVector3(
    (Math.random() - 0.5) * 0.35,
    (Math.random() - 0.5) * 0.35,
    (Math.random() - 0.5) * 0.35
  ));
  body.activate();
}

function shake(message) {
  const strength = message.strength || 1.2;
  for (const item of items) {
    const body = item.body;
    body.applyImpulse(
      new AmmoLib.btVector3(
        (Math.random() - 0.5) * 1.2 * strength,
        (0.5 + Math.random() * 0.6) * strength,
        (Math.random() - 0.5) * 0.9 * strength
      ),
      new AmmoLib.btVector3(0, 0.08, 0)
    );
    body.applyTorqueImpulse(new AmmoLib.btVector3(
      (Math.random() - 0.5) * 0.6 * strength,
      (Math.random() - 0.5) * 0.6 * strength,
      (Math.random() - 0.5) * 0.6 * strength
    ));
    body.activate();
  }
}

function reset() {
  for (const item of items) {
    const transform = new AmmoLib.btTransform();
    transform.setIdentity();
    transform.setOrigin(vec(item.home));
    transform.setRotation(quat(item.homeRotation));
    item.body.setWorldTransform(transform);
    item.body.getMotionState().setWorldTransform(transform);
    item.body.setLinearVelocity(new AmmoLib.btVector3(0, 0, 0));
    item.body.setAngularVelocity(new AmmoLib.btVector3(0, 0, 0));
    item.body.activate();
  }
}

self.onmessage = async (event) => {
  const message = event.data;
  switch (message.type) {
    case 'init':
      try {
        await initAmmo(message.ammoBase);
        buildWorld(message);
        running = true;
        lastTime = performance.now();
        self.postMessage({ type: 'ready' });
        step();
      } catch (error) {
        self.postMessage({ type: 'error', message: String(error && error.message || error) });
      }
      break;
    case 'recycle':
      pool.push(message.buffer);
      break;
    case 'tilt':
      tiltX = message.x;
      tiltZ = message.z;
      break;
    case 'motion': {
      const clamp = (v) => Math.max(-20, Math.min(20, v || 0));
      const clampForce = (v) => Math.max(-20, Math.min(20, v));
      // Basket-frame pseudo force: contents lag behind sudden lifts and swings.
      pseudoForce = [
        clampForce(-clamp(message.ax) * 4.0),
        clampForce(-clamp(message.ay) * 4.5),
        0
      ];
      break;
    }
    case 'hover':
      hoverIndex = message.index;
      break;
    case 'poke':
      if (running) poke(message);
      break;
    case 'shake':
      if (running) shake(message);
      break;
    case 'reset':
      if (running) reset();
      break;
  }
};
