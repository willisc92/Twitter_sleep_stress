import pandas as pd
import json
import folium
from folium import plugins
import os
import sys
import webbrowser

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth', 100)


# Generic method for plotting heatmap plotting taken from https://alysivji.github.io/getting-started-with-folium.html
def map_points(df, lat_col='latitude', lon_col='longitude', zoom_start=4, \
               plot_points=False, pt_radius=10, \
               draw_heatmap=False, heat_map_weights_col=None, \
               heat_map_weights_normalize=True, heat_map_radius=15):
    """Creates a map given a dataframe of points. Can also produce a heatmap overlay

    Arg:
        df: dataframe containing points to maps
        lat_col: Column containing latitude (string)
        lon_col: Column containing longitude (string)
        zoom_start: Integer representing the initial zoom of the map
        plot_points: Add points to map (boolean)
        pt_radius: Size of each point
        draw_heatmap: Add heatmap to map (boolean)
        heat_map_weights_col: Column containing heatmap weights
        heat_map_weights_normalize: Normalize heatmap weights (boolean)
        heat_map_radius: Size of heatmap point

    Returns:
        folium map object
    """

    ## center map in the middle of points center in
    middle_lat = 56.1304
    middle_lon = -106.3468

    curr_map = folium.Map(location=[middle_lat, middle_lon],
                          zoom_start=zoom_start, control_scale=True)

    # add points to map
    if plot_points:
        for _, row in df.iterrows():
            folium.CircleMarker([row[lat_col], row[lon_col]],
                                radius=pt_radius,
                                popup=row['name'],
                                fill_color="#3db7e4",  # divvy color
                                ).add_to(curr_map)

    # add heatmap
    if draw_heatmap:
        # convert to (n, 2) or (n, 3) matrix format
        if heat_map_weights_col is None:
            cols_to_pull = [lat_col, lon_col]
        else:
            # if we have to normalize
            if heat_map_weights_normalize:
                df[heat_map_weights_col] = \
                    df[heat_map_weights_col] / df[heat_map_weights_col].sum()

            cols_to_pull = [lat_col, lon_col, heat_map_weights_col]

        stations = df[cols_to_pull].as_matrix()
        curr_map.add_children(plugins.HeatMap(stations, radius=heat_map_radius))

    plugins.Fullscreen(
        position='topright',
        title='Expand me',
        title_cancel='Exit me',
        force_separate_button=True
    ).add_to(curr_map)

    return curr_map


input_file = 'D:\\Documents\\MEng Software\\ENSF 619-4 Machine Learning\\ensf-619-3-4-project\\Data To Label\\ITER_2_labeled_combined.csv'
df = pd.read_csv(input_file, index_col=0)
df_sleep = df[(df["label"] == "Z") | (df["label"] == "B")]
df_stress = df[(df["label"] == "P") | (df["label"] == "B")]
df_both = df[(df["label"] == "B") | (df["label"] == "P") | (df["label"] == "Z")]

map_pts_slp = map_points(df_sleep, draw_heatmap=True)
map_pts_stress = map_points(df_stress, draw_heatmap=True)
map_pts_both = map_points(df_both, draw_heatmap=True)

sleep_filepath = os.path.dirname(sys.modules['__main__'].__file__) + "/sleep_heatmap.html"
stress_filepath = os.path.dirname(sys.modules['__main__'].__file__) + "/stress_heatmap.html"
both_filepath = os.path.dirname(sys.modules['__main__'].__file__) + "/both_heatmap.html"

map_pts_slp.save(sleep_filepath)
map_pts_stress.save(stress_filepath)
map_pts_both.save(both_filepath)

webbrowser.open('file://' + sleep_filepath)
webbrowser.open('file://' + stress_filepath)
webbrowser.open('file://' + both_filepath)


