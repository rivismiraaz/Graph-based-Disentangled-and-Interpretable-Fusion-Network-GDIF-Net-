import dearpygui.dearpygui as dpg
import numpy as np
import torch
import json
import time

from core.graph_constructor import create_3d_grid_graph_pyg
from core.model import GDIFNet
from simulators.modality_sources import ModalityGenerator
from visualization.visualizer import RealtimeVisualizer

# --- 1. Initialization ---
dpg.create_context()

with open('configs/default_config.json', 'r') as f:
    config = json.load(f)

DIMS = config['grid']['dimensions']
SPACING = config['grid']['spacing']
MODEL_CONFIG = config['model']
NUM_MODALITIES = MODEL_CONFIG['num_modalities']
MODALITY_NAMES = ["Crowd", "Traffic", "Terrain", "Communication", "Wind"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graph = create_3d_grid_graph_pyg(DIMS).to(device)
model = GDIFNet(**MODEL_CONFIG).to(device)

try:
    model.load_state_dict(torch.load("gdif_net_final.pth", map_location=device))
    print("Successfully loaded pre-trained weights.")
except FileNotFoundError:
    print("WARNING: No pre-trained weights found. Using a randomly initialized model.")
model.eval()

modality_gen = ModalityGenerator(DIMS)

# --- Global State ---
params = {
    "crowd": {"hotspots": [{"center": [10, 10, 2], "scale_x": 5, "scale_y": 8, "scale_z": 2, "amplitude": 1.0}]},
    "traffic": {"hotspots": [{"center": [40, 40, 1], "scale_x": 10, "scale_y": 3, "scale_z": 1, "amplitude": 0.8}]},
    "terrain": {"frequency": 0.1, "octaves": 3, "persistence": 0.5, "amplitude": 1.0, "height_ratio": 0.3, "seed": 42},
    "communication": {"stations": [{"center": [25, 25, 10], "radius": 30, "amplitude": 1.0}]},
    "wind": {"gradient_strength": 0.01, "optimal_speed": 5.0}
}
# Store full results for inspector
latest_results = {}

# --- 2. Callbacks & Logic ---
def update_risk_field():
    # a. Generate multi-modal data from simulator
    p = params # shorthand
    modalities_data = np.stack([
        modality_gen.generate_crowd_traffic(p['crowd']).flatten(order='F'),
        modality_gen.generate_crowd_traffic(p['traffic']).flatten(order='F'),
        modality_gen.generate_terrain_complexity({'frequency_x': p['terrain']['frequency'], 'frequency_y': p['terrain']['frequency'], **p['terrain']}).flatten(order='F'),
        modality_gen.generate_communication_signal(p['communication']).flatten(order='F'),
        modality_gen.generate_wind_field({'gradient': {'direction': [0.5,0.5,0], 'strength': p['wind']['gradient_strength']}, 'vortices': [], 'optimal_speed': p['wind']['optimal_speed']}).flatten(order='F')
    ], axis=1)
    
    modalities_tensor = torch.from_numpy(modalities_data).float().to(device)

    # b. Compute risk with the model
    with torch.no_grad():
        risk, uncertainty, weights, latents = model(modalities_tensor, graph.edge_index)

    # c. Store results and update visualizer
    global latest_results
    latest_results = {
        'risk': risk.cpu().numpy(),
        'uncertainty': uncertainty.cpu().numpy(),
        'weights': weights.cpu().numpy(),
        'latents': latents.cpu().numpy(),
        'raw_data': modalities_data
    }
    
    display_mode = dpg.get_value("display_mode_selector")
    display_name_map = {m: m for m in MODALITY_NAMES}
    display_name_map.update({"Total Risk": "Total Risk", "Uncertainty": "Uncertainty"})
    
    if display_mode == "Total Risk":
        grid_data = latest_results['risk'].reshape(DIMS, order='F')
    elif display_mode == "Uncertainty":
        grid_data = latest_results['uncertainty'].reshape(DIMS, order='F')
    else:
        idx = MODALITY_NAMES.index(display_mode)
        contribution = latest_results['weights'][:, idx] * latest_results['risk'].flatten()
        grid_data = contribution.reshape(DIMS, order='F')

    visualizer.update_volume(grid_data, display_name_map[display_mode])

def on_point_pick(cell_id):
    if not latest_results: return
    dpg.set_value("inspector_cell_id", f"Cell ID: {cell_id}")

    risk_val = latest_results['risk'][cell_id][0]
    unc_val = latest_results['uncertainty'][cell_id][0]
    weights_val = latest_results['weights'][cell_id]
    
    dpg.set_value("inspector_risk", f"{risk_val:.4f}")
    dpg.set_value("inspector_uncertainty", f"{unc_val:.4f}")
    
    for i, name in enumerate(MODALITY_NAMES):
        dpg.set_value(f"inspector_weight_{name}", f"{weights_val[i]:.3f}")
        dpg.set_value(f"inspector_raw_{name}", f"{latest_results['raw_data'][cell_id, i]:.3f}")
        dpg.set_value(f"inspector_contrib_{name}", f"{weights_val[i] * risk_val:.4f}")

visualizer = RealtimeVisualizer(
    dimensions=DIMS, spacing=SPACING, 
    window_size=config['visualization']['window_size'],
    title="GDIF-Net Risk Field",
    on_pick_callback=on_point_pick
)

# --- 3. UI Definition ---
def build_hotspot_ui(modality_name):
    with dpg.tree_node(label=f"{modality_name} Hotspots", default_open=True):
        for i, hotspot in enumerate(params[modality_name]['hotspots']):
            with dpg.group(horizontal=True):
                dpg.add_text(f"H{i+1}")
                dpg.add_slider_floatx(label="Center", default_value=hotspot['center'], min_value=0, max_value=50, tag=f"{modality_name}_{i}_center", callback=update_risk_field, width=200)
            dpg.add_slider_float(label="Amplitude", default_value=hotspot['amplitude'], min_value=0, max_value=2.0, tag=f"{modality_name}_{i}_amplitude", callback=update_risk_field)
            dpg.add_separator()

        def add_hotspot():
            params[modality_name]['hotspots'].append({"center": [25, 25, 5], "scale_x": 5, "scale_y": 5, "scale_z": 2, "amplitude": 0.8})
            # This is a hack to rebuild the UI. In a real app, use a more robust state management.
            dpg.delete_item("control_panel_window")
            build_gui()
            update_risk_field()

        def remove_hotspot():
            if len(params[modality_name]['hotspots']) > 0:
                params[modality_name]['hotspots'].pop()
                dpg.delete_item("control_panel_window")
                build_gui()
                update_risk_field()
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Add Hotspot", callback=add_hotspot)
            dpg.add_button(label="Remove Last", callback=remove_hotspot)


def build_gui():
    with dpg.window(label="Control Panel", tag="control_panel_window", width=config['visualization']['control_panel_width'], height=config['visualization']['window_size'][1]):
        dpg.add_text("Real-time Risk Field Simulator")
        dpg.add_separator()

        with dpg.collapsing_header(label="Display Options", default_open=True):
            dpg.add_combo(
                ["Total Risk", "Uncertainty"] + MODALITY_NAMES,
                default_value="Total Risk",
                label="Display Mode",
                tag="display_mode_selector",
                callback=update_risk_field
            )
        
        with dpg.collapsing_header(label="Risk Inspector (Shift+Click in 3D View)", default_open=True):
            dpg.add_text("Cell ID: N/A", tag="inspector_cell_id")
            dpg.add_text("Total Risk: ", tag="inspector_risk")
            dpg.add_text("Uncertainty: ", tag="inspector_uncertainty")
            dpg.add_separator()
            for name in MODALITY_NAMES:
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{name}:", color=[200,200,200])
                    dpg.add_text("Raw:", tag=f"inspector_raw_{name}")
                    dpg.add_text("Weight:", tag=f"inspector_weight_{name}")
                    dpg.add_text("Contrib:", tag=f"inspector_contrib_{name}")
            
        with dpg.collapsing_header(label="Modalities", default_open=True):
            build_hotspot_ui("crowd")
            build_hotspot_ui("traffic")
            # ... Add similar UI builders for other modalities ...

# --- 4. Main Loop ---
build_gui()
dpg.create_viewport(title='GDIF-Net Simulator', width=config['visualization']['window_size'][0], height=config['visualization']['window_size'][1])
dpg.setup_dearpygui()
dpg.show_viewport()

update_risk_field()
visualizer.start(interactive_update=True) 

while dpg.is_dearpygui_running():
    dpg.render_dearpygui_frame()

visualizer.close()
dpg.destroy_context()
