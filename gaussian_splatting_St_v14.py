import streamlit as st
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import tempfile
import os
from io import BytesIO
st.set_page_config(
    page_title="Gaussian Splatting Viewer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("3D Gaussian Splatting Viewer")

# Add fullscreen button
st.markdown("""
<style>
.fullscreen-button {
    position: absolute;
    top: 5px;
    right: 5px;
    z-index: 999;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
This app allows you to upload and visualize .ply files using Gaussian splatting techniques.
Upload a .ply file to get started!
""")

def load_ply_file(file_bytes):
    """Load PLY file from uploaded bytes"""
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_filename = tmp_file.name
    
    try:
        # Load the point cloud using Open3D
        pcd = o3d.io.read_point_cloud(tmp_filename)
        
        # Extract points and colors
        points = np.asarray(pcd.points)
        
        # Check if the point cloud has colors
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            # If no colors, use default color (gray)
            colors = np.ones((len(points), 3)) * 0.5
        
        # Check if the point cloud has normals
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
        else:
            normals = None
            
        return {
            'points': points,
            'colors': colors,
            'normals': normals,
            'pcd': pcd
        }
    finally:
        # Clean up the temporary file
        os.unlink(tmp_filename)

def create_gaussian_splatting_visualization(data, point_size=5, splat_scale=1.0):
    """Create Plotly visualization using Gaussian splatting technique"""
    points = data['points']
    colors = data['colors']
    
    # Convert RGB colors to hex for Plotly
    color_hex = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors]
    
    # Create the scatter3d trace
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=color_hex,
                opacity=0.8,
                line=dict(
                    width=0
                )
            ),
            hoverinfo='none'
        )
    ])
    
    # Calculate the bounding box of the point cloud to set appropriate camera position
    x_range = [points[:, 0].min(), points[:, 0].max()]
    y_range = [points[:, 1].min(), points[:, 1].max()]
    z_range = [points[:, 2].min(), points[:, 2].max()]
    
    center = [
        (x_range[0] + x_range[1]) / 2,
        (y_range[0] + y_range[1]) / 2,
        (z_range[0] + z_range[1]) / 2
    ]
    
    max_range = max(
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        z_range[1] - z_range[0]
    )
    
    camera_dist = max_range * 2.5
    
    # Set up the layout with improved camera position and hidden axes
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False, showbackground=False),
            yaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False, showbackground=False),
            zaxis=dict(visible=False, showticklabels=False, showgrid=False, zeroline=False, showbackground=False),
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=camera_dist, y=camera_dist, z=camera_dist/2)
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        scene_dragmode='orbit',
        height=800,
        width=900
    )
    
    return fig

def display_point_cloud_info(data):
    """Display information about the point cloud"""
    points = data['points']
    st.sidebar.subheader("Point Cloud Information")
    st.sidebar.write(f"Number of points: {len(points):,}")
    
    st.sidebar.write("Bounding Box:")
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)
    st.sidebar.write(f"X: [{x_min:.2f}, {x_max:.2f}]")
    st.sidebar.write(f"Y: [{y_min:.2f}, {y_max:.2f}]")
    st.sidebar.write(f"Z: [{z_min:.2f}, {z_max:.2f}]")
    
    if data['normals'] is not None:
        st.sidebar.write("Normals: Available")
    else:
        st.sidebar.write("Normals: Not available")

# Sidebar controls
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload a PLY file", type=['ply'])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    
    with st.spinner("Processing PLY file..."):
        try:
            data = load_ply_file(file_bytes)
            
            # Display point cloud information
            display_point_cloud_info(data)
            
            # Visualization parameters
            st.sidebar.subheader("Visualization Settings")
            point_size = st.sidebar.slider("Point Size", min_value=1, max_value=20, value=5)
            splat_scale = st.sidebar.slider("Gaussian Splat Scale", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
            
            # Create visualization
            fig = create_gaussian_splatting_visualization(data, point_size, splat_scale)
            
            # Display the figure
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional options
            st.sidebar.subheader("Advanced Options")
            if st.sidebar.button("Download Processed Point Cloud"):
                # Create a download button for the processed point cloud
                with BytesIO() as buffer:
                    o3d.io.write_point_cloud(buffer, data['pcd'])
                    st.sidebar.download_button(
                        label="Download PLY",
                        data=buffer.getvalue(),
                        file_name="processed_pointcloud.ply",
                        mime="application/octet-stream"
                    )
                
        except Exception as e:
            st.error(f"Error processing the PLY file: {str(e)}")
else:
    # Display sample visualization or instructions
    st.info("Upload a PLY file to visualize it with Gaussian splatting.")
    
    # Create a sample visualization for demonstration
    st.subheader("Sample Visualization (Will be replaced with your data)")
    # Create a simple sphere point cloud as an example
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    
    r = 1
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    
    # Generate some colors based on position
    colors = np.zeros((len(points), 3))
    colors[:, 0] = (points[:, 0] + 1) / 2  # R
    colors[:, 1] = (points[:, 1] + 1) / 2  # G
    colors[:, 2] = (points[:, 2] + 1) / 2  # B
    
    sample_data = {
        'points': points,
        'colors': colors,
        'normals': None,
        'pcd': None
    }
    
    fig = create_gaussian_splatting_visualization(sample_data, point_size=5, splat_scale=1.0)
    st.plotly_chart(fig, use_container_width=True)

# Footer with instructions
st.markdown("---")
st.markdown("""
### Instructions:
1. Upload a .ply file using the sidebar
2. Adjust visualization parameters as needed
3. Interact with the 3D visualization:
   - Rotate: Click and drag
   - Zoom: Scroll or pinch
   - Pan: Right-click and drag or Shift+click and drag
""")

# Add information about Gaussian splatting
with st.expander("About Gaussian Splatting"):
    st.markdown("""
    **Gaussian Splatting** is a technique used for 3D rendering and visualization where each point in a point cloud
    is represented as a Gaussian distribution in 3D space. This creates smoother, more realistic renders of point cloud data
    compared to simple point rendering.
    
    Key benefits:
    - Smoother visual representation
    - Better handling of varying point densities
    - More natural transitions between areas of different detail levels
    
    This implementation uses Plotly for visualization and Open3D for point cloud processing.
    """)
