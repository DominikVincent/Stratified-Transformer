import numpy as np
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Get list of all .npy files in the folder
folder_path = '/data/vision/polina/projects/wmh/dhollidt/documents/Pointnet_Pointnet2_pytorch/data/nesf_s3dis_format_65536'  # Update this path
all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
current_index = [0]  # Use a list to hold the current index to make it mutable inside callback

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='pointcloud-plot'),
    html.Button('Previous', id='prev-button', n_clicks=0),
    html.Button('Next', id='next-button', n_clicks=0)
])

@app.callback(
    Output('pointcloud-plot', 'figure'),
    Input('prev-button', 'n_clicks'),
    Input('next-button', 'n_clicks')
)
def update_pointcloud(prev_clicks, next_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    
    # Determine which button was pressed and update index accordingly
    if "prev-button" in changed_id and current_index[0] > 0:
        current_index[0] -= 1
    elif "next-button" in changed_id and current_index[0] < len(all_files) - 1:
        current_index[0] += 1

    # Load point cloud from current file
    pointcloud = np.load(all_files[current_index[0]])
    x, y, z = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]
    colors = pointcloud[:, 3:6]  # Assuming RGB values are in [0, 255] range
    labels = pointcloud[:, -1]

    # Create 3D scatter plot
    fig = px.scatter_3d(
        x=x, y=y, z=z, color=labels, 
        color_continuous_scale='Viridis', 
        range_color=[labels.min(), labels.max()],
        render_mode='webgl'
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=7030)
