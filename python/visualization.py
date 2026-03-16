"""
Flow Visualization Module for RANS-SST Solver

3D visualization of turbulent flow fields using PyVista.
Supports scalar/vector field rendering, animation, and playback
Usage:
    from visualization import FlowData, FlowVisualizer

    # From C++ solver (after evaluate):
    data = flow_data_from_solver(mesh, forward_model)
    vis = FlowVisualizer(data)
    vis.show("U_mag")

    # From raw numpy arrays:
    data = flow_data_from_numpy(cell_centers, U=U_array, p=p_array)
    vis = FlowVisualizer(data)
    vis.show("p")

    # Animation across parameter sweep:
    frames = [flow_data_from_solver(mesh, fm) for theta in thetas
              if fm.evaluate(theta)]
    vis = FlowVisualizer(frames[0])
    vis.animate(frames, field="U_mag")

Run standalone demo:
    python3.12 visualization.py                   # with C++ solver
    python3.12 visualization.py --synthetic        # synthetic data (no solver needed)
"""

import numpy as np
from pathlib import Path

class FlowData:
    """
    Container for cell-centered flow field data on a computational mesh.

    Stores cell center coordinates and an arbitrary dict of named field arrays.
    Scalar fields are (n_cells,) arrays; vector fields are (n_cells, 3).
    Derived quantities (velocity magnitude, turbulence intensity, etc.)
    are computed lazily on first access.

    Parameters
    cell_centers : ndarray, shape (n_cells, 3)
        Cell centroid coordinates.
    fields : dict[str, ndarray]
        Named field arrays. Convention:
            "U"     -> (n_cells, 3) velocity
            "p"     -> (n_cells,)   pressure
            "k"     -> (n_cells,)   turbulent kinetic energy
            "omega" -> (n_cells,)   specific dissipation rate
            "nuT"   -> (n_cells,)   eddy viscosity
            "F1"    -> (n_cells,)   SST blending function 1
            "F2"    -> (n_cells,)   SST blending function 2
            "Pk"    -> (n_cells,)   production of k
    node_coords : ndarray, shape (n_nodes, 3), optional
        Mesh node coordinates (for structured grid reconstruction).
    cell_volumes : ndarray, shape (n_cells,), optional
        Cell volumes.
    metadata : dict, optional
        Arbitrary metadata (Re, nu, theta, iteration count, etc.).
    """

    def __init__(self,
                cell_centers,
                fields=None,
                node_coords=None,
                cell_volumes=None,
                metadata=None):
        
        self.cell_centers = np.asarray(cell_centers, dtype=np.float64)
        self.fields = {k: np.asarray(v, dtype=np.float64)
                       for k, v in (fields or {}).items()}
        self.node_coords = (np.asarray(node_coords, dtype=np.float64)
                            if node_coords is not None else None)
        self.cell_volumes = (np.asarray(cell_volumes, dtype=np.float64)
                             if cell_volumes is not None else None)
        self.metadata = metadata or {}
        self.n_cells = self.cell_centers.shape[0]
        self._derived_cache = {}

    def get_field(self, name):
        """
        Get a field by name.  Supports raw fields and derived quantities:
            "U_mag"    — velocity magnitude
            "TI"       — turbulence intensity  sqrt(2k/3) / |U|
            "Ux","Uy","Uz" — velocity components
            "mu_t_ratio"   — nuT / nu
            "log_k", "log_omega" — log10 of k, omega
        """
        if name in self.fields:
            return self.fields[name]
        if name in self._derived_cache:
            return self._derived_cache[name]
        arr = self._compute_derived(name)
        self._derived_cache[name] = arr
        return arr

    def _compute_derived(self, name):
        if name == "U_mag":
            U = self.fields["U"]
            return np.linalg.norm(U, axis=1)
        if name in ("Ux", "Uy", "Uz"):
            idx = {"Ux": 0, "Uy": 1, "Uz": 2}[name]
            return self.fields["U"][:, idx]
        if name == "TI":
            k = self.fields["k"]
            U_mag = self.get_field("U_mag")
            safe_U = np.maximum(U_mag, 1e-30)
            return np.sqrt(2.0 * np.maximum(k, 0.0) / 3.0) / safe_U
        if name == "mu_t_ratio" and "nuT" in self.fields:
            nu = self.metadata.get("nu", 1e-5)
            return self.fields["nuT"] / nu
        if name == "log_omega" and "omega" in self.fields:
            return np.log10(np.maximum(self.fields["omega"], 1e-30))
        if name == "log_k" and "k" in self.fields:
            return np.log10(np.maximum(self.fields["k"], 1e-30))
        raise KeyError(f"Unknown field '{name}'. Available: "
                       f"{list(self.fields.keys())}")

    def get_available_fields(self):
        """List all available field names (raw + computable derived)."""
        names = list(self.fields.keys())
        if "U" in self.fields:
            names.extend(["U_mag", "Ux", "Uy", "Uz"])
        if "k" in self.fields and "U" in self.fields:
            names.append("TI")
        if "nuT" in self.fields:
            names.append("mu_t_ratio")
        if "omega" in self.fields:
            names.append("log_omega")
        if "k" in self.fields:
            names.append("log_k")
        return names

    def save(self, path):
        """Save to compressed .npz file."""
        save_dict = {"cell_centers": self.cell_centers}
        for k, v in self.fields.items():
            save_dict[f"field_{k}"] = v
        if self.node_coords is not None:
            save_dict["node_coords"] = self.node_coords
        if self.cell_volumes is not None:
            save_dict["cell_volumes"] = self.cell_volumes
        np.savez_compressed(path, **save_dict)


# Factory functions
def flow_data_from_solver(mesh, forward_model):
    """
    Build FlowData from a C++ Mesh and ForwardModel after evaluate().

    Parameters
    mesh : rans_sst_py.Mesh
        The mesh object.
    forward_model : rans_sst_py.ForwardModel
        Forward model with a completed evaluation.
    """
    if not forward_model.has_last_fields():
        raise RuntimeError("No fields available, call evaluate() first")
    centers = np.array(mesh.cell_centers())
    volumes = np.array(mesh.cell_volumes())
    nodes = np.array(mesh.node_coords())
    raw = forward_model.last_fields()  # dict of numpy arrays
    return FlowData(
        cell_centers=centers,
        fields=raw,
        node_coords=nodes,
        cell_volumes=volumes,
    )


def flow_data_from_numpy(cell_centers, U=None, p=None, k=None, omega=None, nuT=None, **extra_fields):
    """
    Build FlowData from raw numpy arrays (for standalone / external use).
    """
    fields = {}
    if U is not None:     fields["U"] = U
    if p is not None:     fields["p"] = p
    if k is not None:     fields["k"] = k
    if omega is not None: fields["omega"] = omega
    if nuT is not None:   fields["nuT"] = nuT
    fields.update(extra_fields)
    return FlowData(cell_centers=cell_centers, fields=fields)


def flow_data_from_vtk(path):
    """Load FlowData from a VTK/VTU file."""
    import pyvista as pv
    mesh = pv.read(path)
    centers = np.array(mesh.cell_centers().points)
    fields = {}
    for name in mesh.cell_data:
        fields[name] = np.array(mesh.cell_data[name])
    nodes = np.array(mesh.points) if mesh.n_points > 0 else None
    return FlowData(cell_centers=centers, fields=fields, node_coords=nodes)


def load_flow_data(path):
    """Load FlowData from a .npz file."""
    data = np.load(path)
    fields = {k.replace("field_", ""): data[k]
              for k in data if k.startswith("field_")}
    return FlowData(
        cell_centers=data["cell_centers"],
        fields=fields,
        node_coords=data.get("node_coords"),
        cell_volumes=data.get("cell_volumes"),
    )


# FlowVisualizer — interactive 3D rendering via PyVista

# Colormaps well-suited for CFD fields
_DEFAULT_CMAPS = {
    "U_mag": "jet",   "Ux": "coolwarm", "Uy": "coolwarm", "Uz": "coolwarm",
    "p": "coolwarm",  "k": "inferno",   "omega": "magma",
    "nuT": "viridis", "F1": "RdYlBu",   "F2": "RdYlBu",
    "TI": "plasma",   "Pk": "hot",
    "log_k": "inferno", "log_omega": "magma", "mu_t_ratio": "viridis",
}

_FIELD_LABELS = {
    "U_mag": "Velocity Magnitude |U| [m/s]",
    "Ux": "Velocity Ux [m/s]", "Uy": "Uy [m/s]", "Uz": "Uz [m/s]",
    "p": "Pressure [Pa]",
    "k": "Turbulent KE k [m^2/s^2]",
    "omega": "Specific Dissipation w [1/s]",
    "nuT": "Eddy Viscosity vt [m^2/s]",
    "F1": "SST Blending F1", "F2": "SST Blending F2",
    "TI": "Turbulence Intensity",
    "Pk": "Production of k [m^2/s^3]",
    "log_k": "log10(k)", "log_omega": "log10(w)",
    "mu_t_ratio": "vt / v",
}


def _cell_centers_to_nodes(c):
    """Convert sorted cell center coordinates to node edge coordinates."""
    nodes = np.empty(len(c) + 1)
    nodes[0] = c[0] - 0.5 * (c[1] - c[0]) if len(c) > 1 else c[0] - 0.5
    nodes[-1] = c[-1] + 0.5 * (c[-1] - c[-2]) if len(c) > 1 else c[-1] + 0.5
    for i in range(1, len(c)):
        nodes[i] = 0.5 * (c[i - 1] + c[i])
    return nodes


class FlowVisualizer:
    """
    Interactive 3D visualizer for turbulent flow fields.

    Features:
    - Click/drag rotation, scroll zoom, right-click pan (PyVista defaults)
    - Scalar field rendering with configurable colormap
    - Vector glyphs and streamlines for velocity
    - Clip planes, iso-surfaces, threshold filters
    - Animation playback across multiple FlowData frames

    Parameters
    data : FlowData
        Initial flow data to display.
    theme : str
        PyVista theme: "document" (white bg), "dark", "paraview".
    """

    def __init__(self, data, theme="document"):
        import pyvista as pv
        self.data = data
        self.theme = theme
        self._pv_mesh = None
        pv.global_theme.load_theme(pv.themes.DocumentTheme())
        if theme == "dark":
            pv.global_theme.load_theme(pv.themes.DarkTheme())
        elif theme == "paraview":
            pv.global_theme.load_theme(pv.themes.ParaViewTheme())

    def _build_point_cloud(self, data=None):
        """Convert FlowData cell centers to a PyVista PolyData object."""
        import pyvista as pv
        d = data or self.data
        cloud = pv.PolyData(d.cell_centers)
        for name in d.fields:
            arr = d.fields[name]
            if arr.ndim == 1:
                cloud.point_data[name] = arr
            elif arr.ndim == 2 and arr.shape[1] == 3:
                cloud.point_data[name] = arr
        # add derived fields that are cheap to compute
        if "U" in d.fields:
            available = d.get_available_fields()
            for dname in ("U_mag", "Ux", "Uy", "Uz"):
                if dname in available and dname not in cloud.point_data:
                    cloud.point_data[dname] = d.get_field(dname)
        return cloud

    def _build_structured_grid(self, data=None):
        """
        Reconstruct a structured grid from cell centers.

        Works for channel2D meshes where cells are arranged on a regular
        (possibly stretched) nx-by-ny layout.  Falls back to None
        if reconstruction fails.
        """
        import pyvista as pv
        d = data or self.data
        cc = d.cell_centers

        # detect structured layout by unique sorted coordinates
        ux = np.unique(np.round(cc[:, 0], decimals=10))
        uy = np.unique(np.round(cc[:, 1], decimals=10))
        uz = np.unique(np.round(cc[:, 2], decimals=10))
        nx, ny, nz = len(ux), len(uy), len(uz)

        if nx * ny * nz != d.n_cells:
            return None  # not a structured grid

        # build node coordinates from cell-center midpoints
        xn = _cell_centers_to_nodes(ux)
        yn = _cell_centers_to_nodes(uy)
        zn = _cell_centers_to_nodes(uz)

        # build 3D node grid
        X, Y, Z = np.meshgrid(xn, yn, zn, indexing="ij")
        grid = pv.StructuredGrid(X, Y, Z)

        # map cell data: build index mapping from structured (ix,iy,iz) -> original cell_id
        idx_map = np.empty(d.n_cells, dtype=int)
        x_idx = np.searchsorted(ux, np.round(cc[:, 0], decimals=10))
        y_idx = np.searchsorted(uy, np.round(cc[:, 1], decimals=10))
        z_idx = np.searchsorted(uz, np.round(cc[:, 2], decimals=10))

        # VTK StructuredGrid cell ordering: x varies fastest (Fortran-like)
        # cell(i,j,k) = i + j*nx + k*nx*ny
        for ci in range(d.n_cells):
            structured_idx = x_idx[ci] + y_idx[ci] * nx + z_idx[ci] * nx * ny
            idx_map[structured_idx] = ci

        for name, arr in d.fields.items():
            if arr.ndim == 1:
                grid.cell_data[name] = arr[idx_map]
            elif arr.ndim == 2 and arr.shape[1] == 3:
                grid.cell_data[name] = arr[idx_map]

        if "U" in d.fields:
            available = d.get_available_fields()
            for dname in ("U_mag", "Ux", "Uy", "Uz"):
                if dname in available:
                    grid.cell_data[dname] = d.get_field(dname)[idx_map]

        return grid

    def _get_mesh(self, data=None):
        """Get the best available PyVista mesh representation."""
        grid = self._build_structured_grid(data)
        if grid is not None:
            return grid
        return self._build_point_cloud(data)

    # Main display methods

    def show(self, field="U_mag", cmap=None, clim=None,
             show_edges=False, opacity=1.0, point_size=5.0,
             window_size=(1400, 800)):
        """
        Display a scalar field interactively.

        Parameters
        field : str
            Field name to color by (e.g. "U_mag", "p", "k", "omega").
        cmap : str, optional
            Matplotlib colormap name. Auto-selected if None.
        clim : tuple, optional
            (min, max) for color scaling. Auto-computed if None.
        show_edges : bool
            Show cell/element edges (structured grids only).
        opacity : float
            Mesh opacity (0-1).
        point_size : float
            Point size (point cloud mode only).
        window_size : tuple
            Window dimensions in pixels.
        """
        import pyvista as pv
        mesh = self._get_mesh()
        is_structured = isinstance(mesh, pv.StructuredGrid)

        data_dict = mesh.cell_data if is_structured else mesh.point_data
        if field not in data_dict:
            arr = self.data.get_field(field)
            data_dict[field] = arr

        if cmap is None:
            cmap = _DEFAULT_CMAPS.get(field, "viridis")
        label = _FIELD_LABELS.get(field, field)

        pl = pv.Plotter(window_size=window_size)
        pl.add_text(label, font_size=12, position="upper_left")

        if is_structured:
            pl.add_mesh(mesh, scalars=field, cmap=cmap, clim=clim,
                        show_edges=show_edges, opacity=opacity,
                        scalar_bar_args={"title": label})
        else:
            pl.add_mesh(mesh, scalars=field, cmap=cmap, clim=clim,
                        point_size=point_size, render_points_as_spheres=True,
                        opacity=opacity,
                        scalar_bar_args={"title": label})

        pl.add_axes()
        pl.show()

    def show_vector(self, field="U", scale=None, factor=0.02,
                    n_arrows=2000, color_by="U_mag", cmap="jet",
                    window_size=(1400, 800)):
        """
        Display velocity arrows on the mesh.

        Parameters
        field : str
            Vector field name (default "U").
        scale : float, optional
            Arrow length scaling. Auto-computed if None.
        factor : float
            Glyph scale factor relative to domain size.
        n_arrows : int
            Max number of arrows to display (subsampled if needed).
        color_by : str
            Scalar field to color arrows by.
        cmap : str
            Colormap for arrow coloring.
        window_size : tuple
            Window size in pixels.
        """
        import pyvista as pv
        cloud = self._build_point_cloud()

        if self.data.n_cells > n_arrows:
            ids = np.random.choice(self.data.n_cells, n_arrows, replace=False)
            cloud = cloud.extract_points(ids)

        if scale is None:
            bbox = cloud.bounds
            domain_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
            U_mag = np.linalg.norm(cloud.point_data[field], axis=1)
            max_vel = np.percentile(U_mag, 95) if len(U_mag) > 0 else 1.0
            scale = factor * domain_size / max(max_vel, 1e-30)

        glyphs = cloud.glyph(orient=field, scale=field, factor=scale)

        pl = pv.Plotter(window_size=window_size)
        pl.add_text("Velocity Vectors", font_size=12, position="upper_left")

        if color_by in glyphs.point_data:
            pl.add_mesh(glyphs, scalars=color_by, cmap=cmap,
                        scalar_bar_args={"title": _FIELD_LABELS.get(color_by, color_by)})
        else:
            pl.add_mesh(glyphs, color="steelblue")

        pl.add_axes()
        pl.show()

    def show_streamlines(self, n_seeds=50, max_time=100.0, tube_radius=None, color_by="U_mag", cmap="jet", window_size=(1400, 800)):
        """
        Display streamlines computed from the velocity field.

        Parameters
        n_seeds : int
            Number of seed points for streamline integration.
        max_time : float
            Maximum integration time.
        tube_radius : float, optional
            Tube radius for streamline rendering. Auto if None.
        color_by : str
            Scalar field to color streamlines by.
        cmap : str
            Colormap.
        window_size : tuple
            Window size.
        """
        import pyvista as pv
        mesh = self._get_mesh()
        is_structured = isinstance(mesh, pv.StructuredGrid)

        if not is_structured:
            print("Streamlines require a structured grid. "
                  "Falling back to vector glyphs.")
            self.show_vector(color_by=color_by, cmap=cmap, window_size=window_size)
            return

        if "U" not in mesh.cell_data:
            mesh.cell_data["U"] = self.data.fields["U"]

        mesh.set_active_vectors("U")

        bounds = mesh.bounds
        y_vals = np.linspace(bounds[2], bounds[3], n_seeds)
        z_mid = 0.5 * (bounds[4] + bounds[5])
        x_start = bounds[0] + 0.01 * (bounds[1] - bounds[0])
        seed_pts = np.column_stack([
            np.full(n_seeds, x_start), y_vals, np.full(n_seeds, z_mid)
        ])
        seed = pv.PolyData(seed_pts)

        streamlines = mesh.streamlines_from_source(
            seed, vectors="U", max_time=max_time,
            integration_direction="forward",
        )

        if tube_radius is None:
            domain_size = max(bounds[1]-bounds[0], bounds[3]-bounds[2])
            tube_radius = 0.003 * domain_size

        tubes = streamlines.tube(radius=tube_radius)

        pl = pv.Plotter(window_size=window_size)
        pl.add_text("Streamlines", font_size=12, position="upper_left")

        if color_by == "U_mag" and "U" in tubes.point_data:
            U = tubes.point_data["U"]
            tubes.point_data["U_mag"] = np.linalg.norm(U, axis=1)

        if color_by in tubes.point_data:
            pl.add_mesh(tubes, scalars=color_by, cmap=cmap,
                        scalar_bar_args={"title": _FIELD_LABELS.get(color_by, color_by)})
        else:
            pl.add_mesh(tubes, color="steelblue")

        pl.add_axes()
        pl.show()

    def show_multi(self, fields=None, shape=None, window_size=(1600, 900), link_views=True):
        """
        Display multiple fields side by side in linked subplots.

        Parameters
        fields : list of str
            Field names to show. Defaults to ["U_mag", "p", "k", "nuT"].
        shape : tuple, optional
            Subplot grid shape. Auto-computed if None.
        window_size : tuple
            Window size in pixels.
        link_views : bool
            Link camera across subplots for synchronized navigation.
        """
        import pyvista as pv

        if fields is None:
            fields = [f for f in ["U_mag", "p", "k", "nuT"] if f in self.data.get_available_fields()]
        n = len(fields)
        if shape is None:
            ncols = min(n, 3)
            nrows = (n + ncols - 1) // ncols
            shape = (nrows, ncols)

        pl = pv.Plotter(shape=shape, window_size=window_size)

        for i, fname in enumerate(fields):
            row, col = divmod(i, shape[1])
            pl.subplot(row, col)
            mesh = self._get_mesh()
            is_structured = isinstance(mesh, pv.StructuredGrid)
            data_dict = mesh.cell_data if is_structured else mesh.point_data

            if fname not in data_dict:
                data_dict[fname] = self.data.get_field(fname)

            cmap = _DEFAULT_CMAPS.get(fname, "viridis")
            label = _FIELD_LABELS.get(fname, fname)
            pl.add_text(label, font_size=9, position="upper_left")

            if is_structured:
                pl.add_mesh(mesh, scalars=fname, cmap=cmap,
                            scalar_bar_args={"title": fname, "n_labels": 3})
            else:
                pl.add_mesh(mesh, scalars=fname, cmap=cmap,
                            point_size=4, render_points_as_spheres=True,
                            scalar_bar_args={"title": fname, "n_labels": 3})
            pl.add_axes()

        if link_views and n > 1:
            pl.link_views()
        pl.show()

    # Clip / threshold

    def show_clip(self, field="U_mag", normal="y", origin=None, cmap=None, window_size=(1400, 800)):
        """
        Display field with an interactive clip plane.

        Parameters
        field : str
            Scalar field to display.
        normal : str or tuple
            Clip plane normal direction ("x", "y", "z") or (nx, ny, nz).
        origin : tuple, optional
            Clip plane origin. Defaults to mesh center.
        cmap : str, optional
            Colormap.
        window_size : tuple
            Window size.
        """
        import pyvista as pv
        mesh = self._get_mesh()
        is_structured = isinstance(mesh, pv.StructuredGrid)
        data_dict = mesh.cell_data if is_structured else mesh.point_data

        if field not in data_dict:
            data_dict[field] = self.data.get_field(field)
        if cmap is None:
            cmap = _DEFAULT_CMAPS.get(field, "viridis")
        label = _FIELD_LABELS.get(field, field)

        pl = pv.Plotter(window_size=window_size)
        pl.add_text(f"{label} (clip)", font_size=12, position="upper_left")
        clip_kwargs = {"normal": normal, "scalars": field, "cmap": cmap,
                       "scalar_bar_args": {"title": label}}
        if origin is not None:
            clip_kwargs["origin"] = origin
        pl.add_mesh_clip_plane(mesh, **clip_kwargs)
        pl.add_axes()
        pl.show()

    def show_threshold(self, field="U_mag", value=None, cmap=None,
                       window_size=(1400, 800)):
        """
        Display field filtered by a threshold range.

        Parameters
        field : str
            Scalar field to threshold.
        value : tuple (min, max), optional
            Threshold range. Defaults to (median, max).
        cmap : str, optional
            Colormap.
        """
        import pyvista as pv
        mesh = self._get_mesh()
        is_structured = isinstance(mesh, pv.StructuredGrid)
        data_dict = mesh.cell_data if is_structured else mesh.point_data

        if field not in data_dict:
            data_dict[field] = self.data.get_field(field)
        if cmap is None:
            cmap = _DEFAULT_CMAPS.get(field, "viridis")

        arr = self.data.get_field(field)
        if value is None:
            value = (np.median(arr), np.max(arr))

        label = _FIELD_LABELS.get(field, field)
        threshed = mesh.threshold(value=value, scalars=field)

        pl = pv.Plotter(window_size=window_size)
        pl.add_text(f"{label} (threshold)", font_size=12, position="upper_left")
        pl.add_mesh(threshed, scalars=field, cmap=cmap,
                    scalar_bar_args={"title": label})
        pl.add_axes()
        pl.show()

    # Animation

    def animate(self, frames, field="U_mag", cmap=None, clim=None,
                fps=10, window_size=(1400, 800), save_path=None):
        """
        Animate across multiple FlowData snapshots with playback controls.

        Keyboard controls during playback:
            Space  -- pause / resume
            Right  -- step forward
            Left   -- step backward
            q      -- quit

        Parameters
        frames : list of FlowData
            Ordered sequence of flow snapshots.
        field : str
            Scalar field to animate.
        cmap : str, optional
            Colormap.
        clim : tuple, optional
            Fixed (min, max) color limits across all frames.
            If None, computed from global min/max.
        fps : int
            Frames per second.
        save_path : str, optional
            If set, save animation to file (GIF or MP4).
        window_size : tuple
            Window size.
        """
        import pyvista as pv
        if not frames:
            raise ValueError("No frames to animate")

        if cmap is None:
            cmap = _DEFAULT_CMAPS.get(field, "viridis")
        label = _FIELD_LABELS.get(field, field)

        # compute global color limits
        if clim is None:
            all_vals = np.concatenate([f.get_field(field) for f in frames])
            clim = (float(np.min(all_vals)), float(np.max(all_vals)))

        # build first frame mesh
        first_mesh = self._get_mesh(frames[0])
        is_structured = isinstance(first_mesh, pv.StructuredGrid)

        pl = pv.Plotter(window_size=window_size, off_screen=save_path is not None)

        # state for playback controls
        state = {"paused": False, "frame_idx": 0, "step_dir": 0}

        def toggle_pause():
            state["paused"] = not state["paused"]
        def step_forward():
            state["step_dir"] = 1
        def step_backward():
            state["step_dir"] = -1

        if save_path is None:
            pl.add_key_event("space", toggle_pause)
            pl.add_key_event("Right", step_forward)
            pl.add_key_event("Left", step_backward)

        # open movie file if saving
        if save_path:
            if save_path.endswith(".gif"):
                pl.open_gif(save_path, fps=fps)
            else:
                pl.open_movie(save_path, framerate=fps)

        # initial mesh
        data_dict = first_mesh.cell_data if is_structured else first_mesh.point_data
        field_data = frames[0].get_field(field)
        data_dict[field] = field_data

        title_actor = pl.add_text(
            f"{label} -- Frame 1/{len(frames)}",
            font_size=12, position="upper_left"
        )

        if is_structured:
            pl.add_mesh(first_mesh, scalars=field, cmap=cmap, clim=clim,
                        scalar_bar_args={"title": label}, name="flow")
        else:
            pl.add_mesh(first_mesh, scalars=field, cmap=cmap, clim=clim,
                        point_size=5, render_points_as_spheres=True,
                        scalar_bar_args={"title": label}, name="flow")
        pl.add_axes()

        n_frames = len(frames)

        def update_frame(idx):
            mesh = self._get_mesh(frames[idx])
            dd = mesh.cell_data if isinstance(mesh, pv.StructuredGrid) else mesh.point_data
            dd[field] = frames[idx].get_field(field)
            pl.add_mesh(mesh, scalars=field, cmap=cmap, clim=clim,
                        point_size=5,
                        render_points_as_spheres=not isinstance(mesh, pv.StructuredGrid),
                        scalar_bar_args={"title": label}, name="flow")
            pl.remove_actor(title_actor)
            pl.add_text(
                f"{label} -- Frame {idx + 1}/{n_frames}",
                font_size=12, position="upper_left"
            )

        if save_path:
            for i in range(n_frames):
                update_frame(i)
                pl.write_frame()
            pl.close()
            print(f"Animation saved to {save_path}")
        else:
            # interactive playback loop
            pl.show(interactive_update=True, auto_close=False)
            import time
            frame_dt = 1.0 / fps
            idx = 0
            running = True
            while running and pl.render_window:
                t0 = time.time()
                if state["step_dir"] != 0:
                    idx = (idx + state["step_dir"]) % n_frames
                    state["step_dir"] = 0
                    update_frame(idx)
                    pl.render()
                elif not state["paused"]:
                    idx = (idx + 1) % n_frames
                    update_frame(idx)
                    pl.render()
                elapsed = time.time() - t0
                if elapsed < frame_dt:
                    time.sleep(frame_dt - elapsed)
                # process_events returns False when the window is closed
                running = pl.process_events()
            pl.close()

    # Export helpers

    def screenshot(self, field="U_mag", path="flow.png", cmap=None,
                   clim=None, window_size=(1920, 1080), camera_position=None):
        """
        Save a static screenshot without opening a window.

        Parameters
        field : str
            Scalar field to render.
        path : str
            Output file path (.png).
        cmap : str, optional
            Colormap.
        clim : tuple, optional
            Color limits.
        window_size : tuple
            Image resolution.
        camera_position : str, optional
            "xy", "xz", "yz", "iso" or a PyVista camera position tuple.
        """
        import pyvista as pv
        mesh = self._get_mesh()
        is_structured = isinstance(mesh, pv.StructuredGrid)
        data_dict = mesh.cell_data if is_structured else mesh.point_data

        if field not in data_dict:
            data_dict[field] = self.data.get_field(field)
        if cmap is None:
            cmap = _DEFAULT_CMAPS.get(field, "viridis")
        label = _FIELD_LABELS.get(field, field)

        pl = pv.Plotter(off_screen=True, window_size=window_size)
        pl.add_text(label, font_size=12, position="upper_left")

        if is_structured:
            pl.add_mesh(mesh, scalars=field, cmap=cmap, clim=clim,
                        scalar_bar_args={"title": label})
        else:
            pl.add_mesh(mesh, scalars=field, cmap=cmap, clim=clim,
                        point_size=5, render_points_as_spheres=True,
                        scalar_bar_args={"title": label})
        pl.add_axes()

        if camera_position == "xy":
            pl.view_xy()
        elif camera_position == "xz":
            pl.view_xz()
        elif camera_position == "yz":
            pl.view_yz()
        elif camera_position == "iso":
            pl.view_isometric()

        pl.screenshot(path)
        pl.close()
        print(f"Screenshot saved to {path}")

    def show_turbulence(self, window_size=(1800, 1000), show_edges=False, link_views=True):
        """
        Display turbulence characterization: k, nuT, nuT/nu ratio, and
        turbulence intensity in a 2x2 linked subplot layout.

        Parameters
        window_size : tuple
            Window size in pixels.
        show_edges : bool
            Show mesh edges.
        link_views : bool
            Synchronize camera across all subplots.
        """
        import pyvista as pv

        panels = []
        available = self.data.get_available_fields()
        # build list of (field_name, label, cmap) for available fields
        candidates = [
            ("k",          "Turbulent KE  k [m^2/s^2]",    "inferno"),
            ("nuT",        "Eddy Viscosity  vt [m^2/s]",   "viridis"),
            ("mu_t_ratio", "Viscosity Ratio  vt/v",        "viridis"),
            ("TI",         "Turbulence Intensity  sqrt(2k/3)/|U|", "plasma"),
            ("log_k",      "log10(k)",                     "inferno"),
            ("log_omega",  "log10(omega)",                 "magma"),
        ]
        for fname, label, cmap in candidates:
            if fname in available:
                panels.append((fname, label, cmap))
        # pick at most 4 for 2x2
        panels = panels[:4]
        n = len(panels)
        if n == 0:
            print("No turbulence fields available (need k, nuT, or U).")
            return

        ncols = min(n, 2)
        nrows = (n + ncols - 1) // ncols
        shape = (nrows, ncols)

        pl = pv.Plotter(shape=shape, window_size=window_size)

        for i, (fname, label, cmap) in enumerate(panels):
            row, col = divmod(i, ncols)
            pl.subplot(row, col)
            mesh = self._get_mesh()
            is_structured = isinstance(mesh, pv.StructuredGrid)
            dd = mesh.cell_data if is_structured else mesh.point_data

            if fname not in dd:
                dd[fname] = self.data.get_field(fname)

            pl.add_text(label, font_size=10, position="upper_left")
            if is_structured:
                pl.add_mesh(mesh, scalars=fname, cmap=cmap,
                            show_edges=show_edges,
                            scalar_bar_args={"title": fname, "n_labels": 5})
            else:
                pl.add_mesh(mesh, scalars=fname, cmap=cmap,
                            point_size=4, render_points_as_spheres=True,
                            scalar_bar_args={"title": fname, "n_labels": 5})
            pl.add_axes()

        if link_views and n > 1:
            pl.link_views()
        pl.show()

    def screenshot_turbulence(self, path="turbulence.png", window_size=(1800, 1000), show_edges=False, camera_position="xy"):
        """
        Save a 2x2 turbulence characterization screenshot (off-screen).

        Parameters
        path : str
            Output file path.
        window_size : tuple
            Image resolution.
        show_edges : bool
            Show mesh edges.
        camera_position : str
            "xy", "xz", "yz", or "iso".
        """
        import pyvista as pv

        panels = []
        available = self.data.get_available_fields()
        candidates = [
            ("k",          "Turbulent KE  k",              "inferno"),
            ("nuT",        "Eddy Viscosity  vt",           "viridis"),
            ("mu_t_ratio", "Viscosity Ratio  vt/v",        "viridis"),
            ("TI",         "Turbulence Intensity",         "plasma"),
        ]
        for fname, label, cmap in candidates:
            if fname in available:
                panels.append((fname, label, cmap))
        panels = panels[:4]
        n = len(panels)
        if n == 0:
            print("No turbulence fields available.")
            return

        ncols = min(n, 2)
        nrows = (n + ncols - 1) // ncols
        shape = (nrows, ncols)

        pl = pv.Plotter(shape=shape, window_size=window_size, off_screen=True)

        for i, (fname, label, cmap) in enumerate(panels):
            row, col = divmod(i, ncols)
            pl.subplot(row, col)
            mesh = self._get_mesh()
            is_structured = isinstance(mesh, pv.StructuredGrid)
            dd = mesh.cell_data if is_structured else mesh.point_data

            if fname not in dd:
                dd[fname] = self.data.get_field(fname)

            pl.add_text(label, font_size=10, position="upper_left")
            if is_structured:
                pl.add_mesh(mesh, scalars=fname, cmap=cmap,
                            show_edges=show_edges,
                            scalar_bar_args={"title": fname, "n_labels": 5})
            else:
                pl.add_mesh(mesh, scalars=fname, cmap=cmap,
                            point_size=4, render_points_as_spheres=True,
                            scalar_bar_args={"title": fname, "n_labels": 5})
            pl.add_axes()

            if camera_position == "xy":   pl.view_xy()
            elif camera_position == "xz": pl.view_xz()
            elif camera_position == "yz": pl.view_yz()
            elif camera_position == "iso": pl.view_isometric()

        pl.screenshot(path)
        pl.close()
        print(f"Turbulence screenshot saved to {path}")

    def to_vtk(self, path):
        """Export the current FlowData to a VTK file."""
        mesh = self._get_mesh()
        mesh.save(path)
        print(f"VTK file saved to {path}")

# Main

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="RANS-SST Flow Visualizer")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no C++ solver needed)")
    parser.add_argument("--field", default="U_mag",
                        help="Field to display (default: U_mag)")
    parser.add_argument("--multi", action="store_true",
                        help="Show multi-field side-by-side view")
    parser.add_argument("--vectors", action="store_true",
                        help="Show velocity vector glyphs")
    parser.add_argument("--streamlines", action="store_true",
                        help="Show streamlines")
    parser.add_argument("--screenshot", default=None,
                        help="Save screenshot to file instead of interactive view")
    parser.add_argument("--theme", default="document",
                        choices=["document", "dark", "paraview"],
                        help="Visual theme")
    parser.add_argument("--nx", type=int, default=40, help="Mesh cells in x")
    parser.add_argument("--ny", type=int, default=30, help="Mesh cells in y")
    args = parser.parse_args()

    if args.synthetic:
        # Generate synthetic turbulent channel flow data
        print(f"Generating synthetic {args.nx}x{args.ny} channel flow...")
        nx, ny = args.nx, args.ny
        Lx, Ly = 6.0, 2.0
        n = nx * ny

        # cell centers (uniform x, stretched y)
        xc = np.linspace(Lx / (2*nx), Lx - Lx / (2*nx), nx)
        stretch = 2.0
        eta = np.linspace(1.0/(2*ny), 1.0 - 1.0/(2*ny), ny)
        yc = 0.5 * Ly * (1.0 + np.tanh(stretch * (2.0 * eta - 1.0))
                         / np.tanh(stretch))
        xx, yy = np.meshgrid(xc, yc, indexing="ij")
        zz = np.full_like(xx, 0.5)
        centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        # synthetic velocity: parabolic profile + small perturbation
        y_norm = yy.ravel() / Ly
        Ux = 4.0 * y_norm * (1.0 - y_norm) + 0.02 * np.random.randn(n)
        Uy = 0.01 * np.sin(2 * np.pi * xx.ravel() / Lx) * np.random.randn(n)
        Uz = np.zeros(n)
        U = np.column_stack([Ux, Uy, Uz])

        # synthetic pressure: slight streamwise gradient
        p = -0.1 * xx.ravel() / Lx + 0.01 * np.random.randn(n)

        # synthetic turbulence: peaks in center, near zero at walls
        k = 0.01 * np.sin(np.pi * y_norm) ** 2 + 1e-6
        omega = 100.0 * (1.0 + 5.0 * np.exp(-50.0 * np.minimum(y_norm, 1.0 - y_norm)))
        nuT = k / omega

        data = flow_data_from_numpy(centers, U=U, p=p, k=k, omega=omega, nuT=nuT)
        print(f"  {n} cells, fields: {list(data.fields.keys())}")
        print(f"  available: {data.get_available_fields()}")

    else:
        # Use C++ solver
        import importlib.util
        build_dir = str(Path(__file__).resolve().parent.parent / "build")
        sys.path.insert(0, build_dir)
        if importlib.util.find_spec("rans_sst_py") is None:
            print("ERROR: cannot import rans_sst_py.")
            print(f"  Looked in: {build_dir}")
            print("  Build the project first: cd build && cmake --build .")
            print("  Or run with --synthetic for a demo without the solver.")
            sys.exit(1)
        import rans_sst_py as rs

        # Channel flow validation case
        # Re_tau ~ 395, half-height = 1.0, U_bulk ~ 1.0
        Lx, Ly = 6.0, 2.0
        nu = 1e-4
        Uin = 1.0
        kIn = 1e-4
        omIn = 10.0

        print(f"Building {args.nx}x{args.ny} channel mesh and solving...")
        mesh = rs.Mesh.make_channel_2d(nx=args.nx, ny=args.ny,
                                        Lx=Lx, Ly=Ly,
                                        Re=Uin * (Ly / 2.0) / nu,
                                        yPlusTarget=1.0)
        mesh.compute_wall_distance()

        param_set = rs.InferenceParameterSet.a1_betaStar()
        bcs = rs.FlowBoundaryConditions.channel_defaults(mesh, Uin, kIn, omIn)
        obs = rs.ObservationOperator()

        settings = rs.SolverSettings()
        settings.max_iterations = 500
        settings.verbose = True

        fm = rs.ForwardModel(mesh, param_set, obs, bcs, nu, settings,
                             rs.Vec3(Uin, 0, 0), 0.0, kIn, omIn)
        defaults = param_set.pack(rs.SSTCoefficients())
        result = fm.evaluate(list(defaults))
        print(f"  Status: {result.status}, iters: {result.simple_iters}")

        data = flow_data_from_solver(mesh, fm)
        print(f"  {data.n_cells} cells, fields: {list(data.fields.keys())}")

    # Visualize
    vis = FlowVisualizer(data, theme=args.theme)

    if args.screenshot:
        vis.screenshot(field=args.field, path=args.screenshot,
                       camera_position="xy")
    elif args.multi:
        vis.show_multi()
    elif args.vectors:
        vis.show_vector()
    elif args.streamlines:
        vis.show_streamlines()
    else:
        vis.show(field=args.field)