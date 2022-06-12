"""Visualisation methods."""
from typing import Any, Dict, Tuple

import ee
import folium


def add_ee_layer(
    self: Any, ee_object: Any, vis_params: Dict[str, Any], name: str, show: bool = True
) -> None:
    """Display Earth Engine image tiles on folium map."""
    try:
        # display ee.Image()
        if isinstance(ee_object, ee.Image):
            map_id_dict = ee_object.getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict["tile_fetcher"].url_format,
                attr="Google Earth Engine",
                name=name,
                overlay=True,
                control=True,
                show=show,
            ).add_to(self)

        elif isinstance(ee_object, ee.image.Image):
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict["tile_fetcher"].url_format,
                attr="Google Earth Engine",
                name=name,
                overlay=True,
                control=True,
                show=show,
            ).add_to(self)

        # display ee.ImageCollection()
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict["tile_fetcher"].url_format,
                attr="Google Earth Engine",
                name=name,
                overlay=True,
                control=True,
                show=show,
            ).add_to(self)

        # display ee.Geometry()
        elif isinstance(ee_object, ee.geometry.Geometry):
            folium.GeoJson(
                data=ee_object.getInfo(),
                style_function=vis_params,
                name=name,
                overlay=True,
                control=True,
                show=show,
            ).add_to(self)

        # display ee.FeatureCollection()
        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict["tile_fetcher"].url_format,
                attr="Google Earth Engine",
                name=name,
                overlay=True,
                control=True,
                show=show,
            ).add_to(self)

    # Catch any exception
    except:  # noqa F722
        print(f"Could not display {name}")


# Add EE drawing method to folium.
folium.Map.add_ee_layer = add_ee_layer


def show_polygon(
    mp: Any,
    polygon: ee.Geometry.Polygon,
    color: str = "#ff0000",
    tag: str = "Bounding Box",
) -> Any:
    """Show a polygon on the map."""
    mp.add_ee_layer(
        polygon,
        lambda x: {"color": color, "fillOpacity": 0},
        tag,
    )
    return mp


def show_point(
    mp: Any,
    point: ee.Geometry.Point,
    color: str = "#ff0000",
    tag: str = "Point",
) -> Any:
    """Show a polygon on the map."""
    mp.add_ee_layer(
        point,
        lambda x: {"color": color},
        tag,
    )
    return mp


def create_map(
    coordinate: Tuple[float, float],
    zoom: int = 15,
) -> Any:
    """Create a map-instance hovering over the specified coordinate."""
    return folium.Map(location=coordinate, zoom_start=zoom)
