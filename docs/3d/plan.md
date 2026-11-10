# 3D suite plan

Checklist for adding image→3D, 3D→3D, auto rigging, mesh tools, and mesh→SDF to pyre. This document is the plan only. It does not add code, weights, or UI.

Build on the cabin path that already exists: `src/triposr.rs` (native Burn TripoSR, marching cubes, GLB), `POST /api/cabin/to3d`, and the WebGL viewer in `ui/cabin.js` (`parseGlb`, `meshViewer`). Do not start a second 3D stack. Do not change STT, TTS, Qwen-Next, or context code except where a capabilities field has to grow a new flag, and even then leave existing flags and their tests behaving as they do now.

Neural inference stays native Rust/Burn inside pyre. No Python, ONNX, or external service at runtime, including as a temporary wrapper. Weight conversion for anything we do ship is Rust, same idea as the in-tree safetensors readers. Pure geometry may use Rust crates. A forked or vendored dependency goes in its own GitHub repo and is added as a submodule, matching `vendor/burn-pyre`, `vendor/burn-gguf`, and `vendor/cubecl-pipe`. This plan vendors nothing.

## VRAM budget

- Target card: 12 GB RTX 3060.
- Qwen-Next already uses about 8.7 GB, so about 3.5 GB is free, and at least 1 GB must stay free. A job that runs while Qwen stays loaded has about 2.5 GB.
- Cabin already does this for TripoSR: `small_gpu(2_300 << 20, …)` leases VRAM, the expert cache shrinks, and the cabin reaper unloads the model after idle (`PYRE_CABIN_IDLE_S`, default 120 s). New neural 3D models have to use that lease, or unload Qwen first, or run on CPU.
- Numbers below are from model cards, Hugging Face file sizes, and source comments. Nothing in this plan was loaded on a GPU here.

## Research

- [x] Read the four references and the existing TripoSR cabin path before choosing anything.
- [x] Record license, size, VRAM fit, and chosen-or-not for each reference.
- [x] Flag non-commercial and non-OSI weights the way `src/isnet.rs` and `src/cabin_catalog.rs` flag BRIA RMBG-1.4.

| Model | Task | License | Size | VRAM fit on a 12 GB 3060 with Qwen loaded | Chosen |
| --- | --- | --- | --- | --- | --- |
| TripoSR (`stabilityai/TripoSR`), already in pyre | Image → textured mesh (triplane NeRF, marching cubes, vertex colours, GLB) | MIT (code and weights, as already catalogued) | Catalogued checkpoint about 1.68 GB (`size_bytes: 1_680_000_000`). Runtime lease 2300 MiB | Fits the ~2.5 GB headroom if the existing lease still holds | Yes. Keep as the image→3D model |
| IDOL (`speridlabs/idol`) | One image → 3D Gaussian splat (`.ply`, 32,768–1,048,576 Gaussians) | Own weights and SDK: Apache-2.0 (HF `cardData.license`, repo tag `license:apache-2.0`). DINOv3 ViT-H+/16 is Meta’s DINOv3 License, a custom non-OSI license: commercial use is allowed, redistribution must carry the agreement, trade-control and weapons uses are prohibited. FLUX.2 VAE (Klein) is Apache-2.0. Not the same flag as RMBG-1.4, and it still has to be shown in the UI | HF `usedStorage` 3,572,854,245 bytes. The four shipped safetensors sum to 3,078,979,896 bytes: `denoiser.safetensors` 1,482,258,308; `decoder_gs.safetensors` 949,083,036; `birefnet.safetensors` 444,466,856; `decoder.safetensors` 203,171,696. Extra encoders are not in that repo: DINOv3 ViT-H+ ~1.68 GB bf16 (840.6M), FLUX.2 VAE ~0.17 GB (84.1M). Card: ~1.80B params, ~3.7–4.4 GB of weights in VRAM, 7.2 GB peak | Does not fit beside Qwen. After Qwen is unloaded, 7.2 GB peak can fit a 12 GB card. Not measured here | Not in the first implementation PRs. Best later neural upgrade if we accept a splat output and the DINOv3 license |
| Pixal3D (`TencentARC/Pixal3D`) | One image → textured GLB with PBR. Three DiT stages: sparse structure, shape, texture. Also a multi-view pipeline | Code and HF card: MIT. `extra_gated_eu_disallowed: true` on the card. Third-party pieces (DINOv3 ViT-L, MoGe-2, rembg, NAF) keep their own licenses; DINOv3 is the same custom Meta license | HF `usedStorage` 46,046,913,845 bytes (~43 GB), single-view and multi-view copies of several 1.3B bf16 DiTs plus decoders | Source comment: all stages at once are ~18 GB. Low-VRAM mode keeps weights on CPU and moves one flow model plus one DINOv3 to the GPU. A 1.3B bf16 DiT is ~2.6 GB of weights before activations. That does not fit in the 2.5 GB headroom, and a 1024 / 1536 cascade is not a documented 12 GB fit | No. Too large to port and too large to run on the target card |
| SDF-StyleGAN (`SerdarHelli/SDF-StyleGAN-3D`, paper Zheng et al. 2022, code `Zhengxinyang/SDF-StyleGAN`) | Latent → 3D feature volume → SDF → mesh. GAN inversion can reconstruct, complete, and edit shapes. It is not a general mesh→SDF converter | Code repo: MIT. HF card: `license: other`. The published file is `cars.ckpt`, trained on ShapeNet. ShapeNet’s terms are non-commercial research. Treat the checkpoint as non-commercial / unclear until someone reads the actual weight license. Do not ship it casually | HF `usedStorage` 687,975,763 bytes (the `cars.ckpt` upload) | A 656 MB generator can sit in the headroom or on CPU. That does not make the task match | No for the neural generator. Yes for the geometric idea: a mesh becomes a sampled SDF grid, and marching cubes (already in `src/mc_tables.rs`) can turn a grid back into a mesh |
| Modly (`lightningpixel/modly`) | Desktop app: image or prompt → mesh, model manager, workflows, decimate, remesh, texture, export | Application code is open source and Python (PyTorch, trimesh, pymeshlab). Generators are extensions (for example `sf3d`, Trellis.2 GGUF) with their own licenses | Not one model | The app assumes a GPU that can hold a generator. That is the wrong runtime for pyre | Borrow the geometry UX only. Do not embed Modly, its Python server, or its generators |

### IDOL, in more detail

- [x] Architecture: DiT denoiser 370.6M (FP16 blocks, FP32 embeddings and norms), octree decoder 50.8M, Gaussian decoder 237.3M, BiRefNet 220.7M for the cut-out, DINOv3 ViT-H+/16 840.6M, FLUX.2 VAE 84.1M. Euler sampler, 50 steps, CFG 3.0, time shift 3.0. Output is a standard 3DGS `.ply`, not a GLB.
- [x] The cabin viewer only reads one mesh primitive (`POSITION`, `NORMAL`, `COLOR_0`). A splat file would need a new viewer. Mesh extraction from Gaussians is a separate algorithm and is not claimed here.
- [x] Porting it means a Burn DiT, two decoders, BiRefNet, DINOv3, and a VAE, plus Rust conversion of those safetensors. That is a later project, after the geometry tools exist, and only with Qwen unloaded for the run.

### Pixal3D, in more detail

- [x] Stages, from the README: sparse structure 32→64 (`ss_flow_img_dit_*`), shape 256→512→1024 (`slat_flow_img2shape_*`), texture 256→512→1024 (`slat_flow_imgshape2tex_*`). Conditioning is pixel-aligned back-projection. Inference also uses DINOv3 ViT-L (`camenduru/dinov3-vitl16-pretrain-lvd1689m`), a NAF upsampler, MoGe-2 for FOV (`Ruicheng/moge-2-vitl`), and a background removal step.
- [x] The texture stage is the reference’s real 3D→3D feature (image + shape latent → PBR). It uses the same 1.3B class of weights, so it stays out of scope for the same VRAM and porting reasons.
- [x] Low-VRAM mode is still “one DiT on GPU”, not “2.5 GB beside Qwen”.

### Modly, what is worth copying

- [x] Worth copying as cabin workflows, in pure Rust: load on demand (cabin already does this), a target vertex count, triangle decimation, a remesh choice, smooth, format export (GLB, OBJ, STL, PLY), and a note when a texture or UVs must be preserved.
- [x] Not copied: the node graph, the extension installer, the Python model manager, Gaussian splat viewing, the transform gizmo, and quad remesh via MeshLab. Quad remesh can be a later item if a small pure-Rust method shows up. It is not required for the first mesh-tools PR.

### SDF-StyleGAN, what actually applies

- [x] The paper’s generator emits a feature volume (about 32³ × 16), trilinear lookup plus a shallow MLP produces an SDF, and marching cubes at 128³ extracts the mesh. Training uses global and local discriminators on SDF values and gradients. The HF upload is a cars generator, not an encoder from an arbitrary mesh.
- [x] Mesh→SDF in pyre is geometric: closest-point distance, sign from a winding number or pseudonormal, written as a dense grid. The StyleGAN checkpoint is not converted and not downloaded until its license is explicit. If that license is non-commercial, the catalog entry must say so the way RMBG-1.4 does.

### TripoSR, what already exists

- [x] `src/triposr.rs`: DINO ViT-B/16, triplane transformer, NeRF MLP, marching cubes, GLB. Weights: `<store>/cabin/triposr/triposr_f16.safetensors`.
- [x] `POST /api/cabin/to3d` cuts the subject out with IS-Net, then runs TripoSR. RMBG-1.4 is marked non-commercial in the catalog and in the cabin card.
- [x] `cabin_catalog::convert_triposr` still shells out to `python convert.py`. That violates the “conversion in Rust” rule for any new work. Replacing that converter is part of the image→3D PR, not this plan PR. Until a Rust converter exists, document the Python step as legacy rather than adding new Python.

## Decisions

- [x] Image→3D for the next PRs: keep TripoSR. Do not port IDOL or Pixal3D in this series.
- [x] 3D→3D for the next PRs: geometric. Smooth, remesh, decimate, and re-apply vertex colours. A neural retexture that runs SD 1.5 over a UV atlas is optional and only if it reuses the existing image model and its lease. Pixal3D’s texture DiT is out.
- [x] Auto rigging: no reference in the list is a rigger. Implement a geometric fit (template skeleton, bone embedding, heat-diffusion skin weights in the sense of Baran and Popović 2007) for a humanoid and a simpler character template. Export glTF skin (`JOINTS_0`, `WEIGHTS_0`, inverse bind matrices). That is the FBX-compatible interchange. Do not write FBX.
- [x] Mesh→SDF: geometric grid export, plus a round trip through the existing marching-cubes tables. Do not port SDF-StyleGAN.
- [x] Every shipped 3D tool reports capabilities on `/v1/models` and `/api/models` through the existing `capabilities` object.

## Phases and PR breakdown

Later PRs branch from `qwen-int` (or from the previous 3D PR only when they stack). No force-push. Each PR updates the checkboxes in this file.

- [x] PR A — this document (and `docs/stt/plan.md`). Docs only.
- [ ] PR B — mesh tools and geometric mesh→SDF. Pure Rust, unit-tested, cabin API and UI. No new neural net.
- [ ] PR C — image→3D cleanup on TripoSR: capabilities card, Rust weight conversion to replace `convert.py`, viewer reused. Still no IDOL or Pixal3D port.
- [ ] PR D — automatic rigging, glTF skin export, skeleton preview in the existing viewer.
- [ ] PR E — 3D→3D geometric pass (smooth / project colours / optional SD atlas only if the image model is already the one cabin loads).

Effort, as scope rather than calendar time:

- PR B is local geometry and I/O. The risky pieces are hole filling and UV charts, which can ship as a simple projection if LSCM is not solid.
- PR C is a converter plus catalog wiring. Numerical parity needs the existing TripoSR checkpoint and a reference mesh; do not claim parity without that run.
- PR D is a new algorithm with a visual check. Skin weights need a small analytic test (a cylinder, a known bone).
- PR E is mostly PR B operations composed, plus an optional cabin image-model path.
- An IDOL or Pixal3D Burn port is a separate project after these PRs, and only with an explicit license decision for DINOv3.

## PR B — mesh tools and mesh→SDF

### API

- [ ] `POST /api/cabin/mesh` with `{ mesh, op, params }`. `mesh` is base64 GLB, OBJ, PLY, or STL, or `{ glb }` matching `/api/cabin/to3d`.
- [ ] Ops: `stats`, `normals`, `center`, `scale`, `decimate`, `remesh`, `repair`, `uv`, `convert`, `sdf`.
- [ ] Response always includes `stats` (vertices, triangles, bbox, area, watertight flag, edge counts) and, when the op emits a mesh, `glb` plus `bytes`. `convert` also returns the requested format (`glb`, `obj`, `ply`, `stl`).
- [ ] `sdf` returns a float32 grid, the origin, the spacing, the dimensions, and the sign convention (negative inside). Default resolution 64³, cap 128³ unless the request asks and the byte size stays bounded. Also return a marching-cubes GLB of the zero level set so the viewer can show it.
- [ ] Errors are the usual cabin JSON errors. Reject a mesh over a stated triangle cap rather than allocating without limit.
- [ ] Workflow card in `GET /api/cabin/workflows`: id `mesh`, always `ready: true`, model string names the pure-Rust toolkit and says it uses no GPU.

### Geometry

- [ ] In-tree mesh type: positions, triangles, optional normals, optional vertex colours, optional UVs. Load and save GLB (extend the writer in `src/triposr.rs` rather than a second GLB dialect), OBJ, PLY, STL.
- [ ] `normals`: area-weighted vertex normals, consistent winding where the mesh is orientable.
- [ ] `center` / `scale`: bbox center to origin, uniform scale to a requested extent (default 1).
- [ ] `decimate`: quadric edge collapse to a target face or vertex count. Preserve vertex colours by copying the kept vertex. If UVs exist, keep them on the kept vertex and say when seams are not preserved.
- [ ] `remesh`: triangle isotropic or voxel remesh at a requested edge length. Not Modly’s quad remesh.
- [ ] `repair`: weld vertices, drop degenerate triangles, fix inconsistent winding on orientable patches, report non-manifold edges. Fill only small boundary loops. Say in the response what was left unfixed.
- [ ] `uv`: box or planar projection as the first version. LSCM only if the implementation stays in-tree and the tests pass. Do not add an xatlas submodule in this PR.
- [ ] `sdf`: closest-point distance with winding-number or pseudonormal sign. Document the method in the response (`method` field) so it is not confused with SDF-StyleGAN.

### UI

- [ ] New cabin card “Mesh tools” next to the existing “Image to 3D” card.
- [ ] File input accepts GLB, OBJ, PLY, STL. A button can send the last TripoSR GLB if one is still in the page.
- [ ] Op selector and the few numeric fields each op needs (target faces, resolution, target size, output format).
- [ ] Reuse `meshViewer` for any returned GLB. Show stats under the viewer the way `to3d` shows vertex and triangle counts.
- [ ] SDF result: show the marching-cubes preview and a download for the grid (`.npy` if the existing `npy` crate writes it, otherwise raw float32 plus a JSON header). State which one shipped.
- [ ] No new tab unless the cabin card list cannot hold the tools. Prefer the cabin tab.

### Capabilities

- [ ] Extend `Capabilities` with `mesh_input` and `mesh_output` without changing the meaning of the current flags. Existing tests in `src/capabilities.rs` stay green.
- [ ] Add a `three_d` object on the JSON, the way `speech` is added for transcription models: tasks, ops, `neural: false`, license `MIT` (pyre’s own code), and a one-line VRAM note (`cpu`, no lease).
- [ ] Register a builtin card (same idea as the diarizer card in `App::model_cards`) so `/v1/models` and `/api/models` list it even though nothing is loaded. Suggested id: `pyre-mesh`. Kind string needs a new `Kind` variant or a documented `unknown` card. If a new variant is added, update every `match` and do not change detection of GGUF, Whisper, or SD folders.
- [ ] Summary text must mention mesh in/out so the Models table shows it.

### Tests

- [ ] Unit tests, no GPU and no weights: cube and tetrahedron stats; OBJ/STL/GLB round trip within a tolerance; decimate reduces a subdivided plane and keeps the corners; normals of an axis-aligned cube face point outward; center/scale of a translated box; repair welds a duplicated vertex.
- [ ] SDF of a cube: sample points known to be inside are negative, outside are positive, and the zero isosurface sits near the faces. Marching cubes of that grid produces a box with six sides.
- [ ] Non-manifold input returns a report and does not panic.
- [ ] These tests are the parity check for this PR. There is no neural reference to match.

## PR C — image→3D

- [ ] TripoSR, when its f16 weights are present, appears in `/v1/models` and `/api/models` with `image_input`, `mesh_output`, and a `three_d` block: task `image-to-3d`, license `MIT`, lease about 2300 MiB, backend the one cabin uses.
- [ ] The card states that IDOL and Pixal3D are not loaded and why (size and VRAM), so the choice is visible. Do not pretend they are implemented.
- [ ] Replace `convert_triposr`’s `python convert.py` call with a Rust reader of `model.ckpt` into `triposr_f16.safetensors`. Layout must match what `Tsr::load` already requires (`dino_pos512`, `decoder.layers.18.weight`).
- [ ] Parity: on a machine that has the checkpoint, compare a fixed image’s vertex count and a few density samples against the current converter’s output, or against the PyTorch reference if it is still available. Record the command and the deltas in the PR. If the checkpoint is not on the machine, say so and do not invent a match.
- [ ] UI stays the existing `to3d` workflow. Point the mesh-tools card at its GLB.
- [ ] IS-Net cut-out remains, including the non-commercial label on RMBG-1.4.

## PR D — auto rigging

- [ ] `POST /api/cabin/rig` with a mesh plus `template: humanoid | character`.
- [ ] Fit a named skeleton (hips, spine, head, arms, legs for humanoid; a shorter chain for character). Place joints from the mesh bbox and principal axis, then refine by the geometric embedding method. Document the assumptions (upright, facing +Z or the glTF axis the TripoSR writer uses).
- [ ] Skin weights: heat diffusion or a documented distance falloff if the heat solve is unstable. Weights per vertex sum to 1. At most 4 influences, matching glTF.
- [ ] GLB contains a skin, nodes, and inverse bind matrices. The mesh primitive still has the attributes `parseGlb` needs, plus `JOINTS_0` and `WEIGHTS_0`.
- [ ] Viewer: draw bone segments when those attributes or a `skins` array exist. Orbit controls stay as they are.
- [ ] Capabilities: the `pyre-mesh` card grows `rig: [humanoid, character]` and notes that this is geometric, not a learned rigger.
- [ ] Tests: a capsule or box character gets a spine of the expected joint count; weights are finite and sum to 1; the GLB JSON contains a `skins` entry. No claim of animation quality beyond that.

## PR E — 3D→3D

- [ ] Ops on the mesh endpoint: `smooth` (Taubin or Laplacian, a few iterations), `recolour` (project vertex colours from a second mesh or from an image with a stated camera).
- [ ] Optional: UV atlas plus a call into the existing SD 1.5 img2img path, only when that model is already how cabin paints images, with the same VRAM lease. Skip it if wiring it would fork the image pipeline.
- [ ] UI card “Refine mesh” with the viewer before and after, or one viewer swapped to the result plus a download.
- [ ] Capabilities list the ops. `neural` stays false unless the optional SD path is actually exposed, in which case the card points at the SD model’s own capabilities rather than claiming a new 3D network.
- [ ] State in the PR that Pixal3D texture generation and IDOL were not run.

## Weight conversion

- [ ] PR B: none.
- [ ] PR C: Rust conversion of TripoSR `model.ckpt` → `triposr_f16.safetensors`. Delete the requirement on `convert.py` once the Rust output loads in `Tsr::load`.
- [ ] IDOL files are already safetensors. A future port would map them in Rust. Not in this series.
- [ ] Pixal3D files are already safetensors. Not in this series.
- [ ] `cars.ckpt` is a PyTorch checkpoint. Do not convert it while the license is `other` plus ShapeNet.

## What was not run

- [x] No IDOL, Pixal3D, SDF-StyleGAN, or TripoSR weights were loaded in this planning pass.
- [x] No GPU timing and no 3060 measurement. VRAM figures are from cards and source comments.
- [x] `C:\Temp` reference trees used by cabin converters were not available here.

## Open questions

- [ ] Does Lyam want unloaded tools (`pyre-mesh`) on `/v1/models`, or only on `/api/models`? The plan puts a builtin card on both so the capabilities object is where clients already look.
- [ ] Is a DINOv3-licensed IDOL port acceptable later, with the agreement shown in the UI? Default is no port until that is a yes.
- [ ] Who confirms the SDF-StyleGAN weight license before any download?
- [ ] UV: is planar projection enough for the first mesh PR, with LSCM deferred?
- [ ] Rigging axis: TripoSR writes glTF as `(y, z, x)` from its `(x toward camera, z up)` frame. The rigger must use that frame for TripoSR meshes and the file’s own up-axis for imports.
- [ ] A new GitHub repo for a geometry crate is unnecessary if the code stays in pyre. A later neural port needs a `Lyamc/*` repo before it can be a submodule. This agent cannot create that repo.
