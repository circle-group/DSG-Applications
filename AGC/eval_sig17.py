from pathlib import Path
import torch
from digeo import load_mesh_from_file
import yaml
import polyscope as ps

from models import ResNet
from utils import heat_kernel_signature

device = "cuda"
output_path = Path(__file__).parent / "output" / "sig17_xyz"
mesh_path = (
    Path(__file__).parent
    / "data"
    / "sig17_seg_benchmark"
    / "meshes"
    / "test"
    / "shrec"
    / "14.off"
)

with open(output_path / "config.yaml", "r") as file:
    config = yaml.safe_load(file)
model = ResNet(config).to(device)
model.load_state_dict(torch.load(output_path / "model.pth", map_location=device))
model.eval()

ps.set_allow_headless_backends(True)
ps.init()

subsampled_mesh_path = mesh_path.replace(".off", "_subsampled.obj")
subsampled_mesh = load_mesh_from_file(subsampled_mesh_path, device=device)
mesh = load_mesh_from_file(mesh_path, device=device)
if config["model"]["input"] == "hks":
    model_input = torch.tensor(heat_kernel_signature(mesh)).to(device, torch.float32)
else:
    model_input = None
outputs = model(mesh, subsampled_mesh, model_input)

ps.register_surface_mesh(
    "mesh",
    mesh.vertices.detach().cpu().numpy(),
    mesh.faces.detach().cpu().numpy(),
    smooth_shade=True,
)

ps.get_surface_mesh("mesh").add_scalar_quantity(
    "scalar",
    torch.argmax(outputs, dim=-1).float().detach().cpu().numpy(),
    defined_on="faces",
    enabled=True,
    cmap="jet",
)

# ps.set_up_dir("z_up")
ps.look_at((3.5, -0.2, 0), (0.0, -0.5, 0.0))

ps.set_ground_plane_mode("shadow_only")
ps.set_shadow_darkness(0.2)  # Adjust shadow darkness
ps.set_shadow_blur_iters(10)  # Adjust shadow softness
ps.set_window_size(3840, 2160)  # 4K resolution
ps.screenshot()
