import pytest

import pandas as pd

@pytest.fixture()
def external_targets():
    # from 102021539 tile, mer catalog, OTF, random 4 targets, made up FoV
    return pd.DataFrame([
        {
            'right_ascension': 92.87298692502482,
            'declination': -48.19444984968764,
            'field_of_view': 100
        },
        {
            'right_ascension': 92.85155819758324,
            'declination': -47.90361272388686,
            'field_of_view': 100
        },
        {
            'right_ascension': 92.75345265411455,
            'declination': -48.06510822714088,
            'field_of_view': 100
        },
        {
            'right_ascension': 92.66503743195992,
            'declination': -48.22056418652982,
            'field_of_view': 100
        }
    ])

@pytest.fixture()
def vis_loc():
    return 'path/to/vis.fits'