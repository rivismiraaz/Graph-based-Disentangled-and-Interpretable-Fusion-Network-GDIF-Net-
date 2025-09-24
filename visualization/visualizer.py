import pyvista as pv
import numpy as np

class RealtimeVisualizer:
    def __init__(self, dimensions, spacing, window_size, title, on_pick_callback=None):
        self.grid = pv.UniformGrid(dimensions=dimensions, spacing=spacing)
        self.plotter = pv.Plotter(window_size=window_size, title=title)
        self.plotter.background_color = 'black'
        
        self.volume = self.plotter.add_volume(
            self.grid,
            scalars=np.zeros(self.grid.n_cells),
            cmap="coolwarm",
            scalar_bar_args={'title': 'Risk Level', 'color': 'white'}
        )
        self.on_pick_callback = on_pick_callback
        if self.on_pick_callback:
            self.plotter.enable_cell_picking(
                callback=self._handle_pick, 
                show=False, 
                style='wireframe',
                font_size=1, # Hack to make the selection box invisible
                color='white',
                line_width=0
            )

    def _handle_pick(self, *args):
        # We only care about the picked cell index
        if self.plotter.picked_cell and self.on_pick_callback:
            cell_id = self.plotter.picked_cell.id
            self.on_pick_callback(cell_id)

    def update_volume(self, data_grid, name):
        """Updates the volume data and scalar bar title."""
        self.grid.cell_data['data'] = data_grid.flatten(order='F')
        self.volume.scalar_bar.SetTitle(f"{name}")
        self.plotter.update()

    def start(self, interactive_update=False):
        self.plotter.show(interactive_update=interactive_update)
        
    def close(self):
        self.plotter.close()
