print(help('modules'))

import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
# TODO: Install correct sklearn version 0.24.1
import sklearn
import streamlit as st
import yaml

st.set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)  # Not sure why RF gets mutated
def load_interpolator(sim, output_feature):
    interpolate_dir = f'interpolators/{sim}/'
    interpolator_name = f'{output_feature}_interpolator.joblib'
    interpolator = joblib.load(interpolate_dir+interpolator_name)
    with open(interpolate_dir+'output_features.yaml', 'r') as yaml_file:
        input_properties = yaml.safe_load(yaml_file)[output_feature]
    snapshots = np.load(interpolate_dir+'snapshots.npy')
    return interpolator, snapshots, input_properties

short_names = {
    'BH mass': 'bh_mass',
    'DM mass': 'dm_sub_mass',
    'Gas mass': 'gas_mass',
    'SFR': 'sfr',
    'Stellar mass': 'stellar_mass',
    'Stellar metallicity': 'stellar_metallicity',
}
proper_names = {v: k for (k, v) in short_names.items()}
colors = {
    'bh_mass': '#1F77B4',             # Blue
    'dm_sub_mass': '#7F7F7F',         # Grey
    'gas_mass': '#2BA02B',            # Green
    'stellar_mass': '#FF7F0E',        # Orange
}

st.header('Feature importance when predicting the z=0 properties of [CAMELS](https://camels.readthedocs.io) galaxies - See [this paper](https://github.com/robmcgibboned/camels_feature_importance/blob/main/paper_draft.pdf) for details')

# Creating layout
row1 = st.columns([1, 3, 2, 3, 1])
row2 = st.columns([1, 1, 1, 0.7, 1, 1, 1])

# Left plot sliders
with row2[0]:
    omega_m = st.slider('Omega matter',
        min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    sigma_8 = st.slider('Sigma 8', 
        min_value=0.6, max_value=1.0, value=0.8, step=0.05)
    sim = st.selectbox(
        'CAMELS simulation suite',
        ['IllustrisTNG', 'SIMBA']
    )
with row2[1]:
    a_sn_1 = st.slider('Supernova 1',
        min_value=-2.0, max_value=2.0, value=0.0, step=0.5)
    a_sn_2 = st.slider('Supernova 2',
        min_value=-1.0, max_value=1.0, value=0.0, step=0.25)
    output_feature_proper = st.selectbox(
        'Property to predict',
        ['BH mass', 'Gas mass', 'SFR', 'Stellar mass', 'Stellar metallicity'],
        index=3,
    )
with row2[2]:
    a_agn_1 = st.slider('AGN 1',
        min_value=-2.0, max_value=2.0, value=0.0, step=0.5)
    a_agn_2 = st.slider('AGN 2',
        min_value=-1.0, max_value=1.0, value=0.0, step=0.25)

# Right plot sliders
with row2[4]:
    r_omega_m = st.slider('Omega matter', key='r_omega_m',
        min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    r_sigma_8 = st.slider('Sigma 8', key='r_sigma_8',
        min_value=0.6, max_value=1.0, value=0.8, step=0.05)
    r_sim = st.selectbox(
        'CAMELS simulation suite',
        ['IllustrisTNG', 'SIMBA'],
        key='r_sim'
    )
with row2[5]:
    r_a_sn_1 = st.slider('Supernova 1', key='r_a_sn_1',
        min_value=-2.0, max_value=2.0, value=0.0, step=0.5)
    r_a_sn_2 = st.slider('Supernova 2', key='r_a_sn_2',
        min_value=-1.0, max_value=1.0, value=0.0, step=0.25)
    r_output_feature_proper = st.selectbox(
        'Property to predict',
        ['BH mass', 'Gas mass', 'SFR', 'Stellar mass', 'Stellar metallicity'],
        index=3,
        key='r_output_feature_proper',
    )
with row2[6]:
    r_a_agn_1 = st.slider('AGN 1', key='r_a_agn_1',
        min_value=-2.0, max_value=2.0, value=0.0, step=0.5)
    r_a_agn_2 = st.slider('AGN 2', key='r_a_agn_2',
        min_value=-1.0, max_value=1.0, value=0.0, step=0.25)

# Left plot figure
with row1[1]:
    output_feature = short_names[output_feature_proper]
    interpolator, snapshots, input_properties = load_interpolator(sim, output_feature)
    input_features = [str(snap) + prop for snap in snapshots for prop in input_properties]
    params = [omega_m, sigma_8, a_sn_1, a_agn_1, a_sn_2, a_agn_2]
    mean_importance = interpolator.predict([params])[0]

    fig, ax = plt.subplots(1, dpi=200)

    data = {}
    max_fi = 0
    for input_property in input_properties:
        fi = []
        for snap in snapshots:
            idx = input_features.index(str(snap)+input_property)
            fi.append(mean_importance[idx])
        data[input_property] = fi
        max_fi = max(max_fi, np.max(fi))

    for input_property in input_properties:
        fi = data[input_property] / max_fi
        ax.plot(snapshots, fi, 
                label=proper_names[input_property], 
                color=colors[input_property])

    with open('redshifts.yaml', 'r') as yaml_file:
        redshifts = yaml.safe_load(yaml_file)

    padding = 0.015 * (np.max(snapshots) - np.min(snapshots))
    ax.set_xlim(np.min(snapshots)-padding, np.max(snapshots)+padding)
    xticks = np.linspace(np.min(snapshots), np.max(snapshots), 5)
    xticklabels = [round(redshifts[round(s)], 1) for s in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    anchor = (1.01, 1.15) if len(input_properties) == 4 else (0.9, 1.15)
    ax.legend(bbox_to_anchor=anchor, ncol=4)
    ax.set_xlabel('z', fontsize=14)
    ax.set_ylabel('Feature importance', fontsize=14)
    st.write(fig)

# Right plot figure
with row1[3]:
    output_feature = short_names[r_output_feature_proper]
    interpolator, snapshots, input_properties = load_interpolator(r_sim, output_feature)
    input_features = [str(snap) + prop for snap in snapshots for prop in input_properties]
    params = [r_omega_m, r_sigma_8, r_a_sn_1, r_a_agn_1, r_a_sn_2, r_a_agn_2]
    mean_importance = interpolator.predict([params])[0]

    r_fig, r_ax = plt.subplots(1, dpi=200)

    data = {}
    max_fi = 0
    for input_property in input_properties:
        fi = []
        for snap in snapshots:
            idx = input_features.index(str(snap)+input_property)
            fi.append(mean_importance[idx])
        data[input_property] = fi
        max_fi = max(max_fi, np.max(fi))

    for input_property in input_properties:
        fi = data[input_property] / max_fi
        r_ax.plot(snapshots, fi, 
                label=proper_names[input_property], 
                color=colors[input_property])

    with open('redshifts.yaml', 'r') as yaml_file:
        redshifts = yaml.safe_load(yaml_file)

    padding = 0.015 * (np.max(snapshots) - np.min(snapshots))
    r_ax.set_xlim(np.min(snapshots)-padding, np.max(snapshots)+padding)
    xticks = np.linspace(np.min(snapshots), np.max(snapshots), 5)
    xticklabels = [round(redshifts[round(s)], 1) for s in xticks]
    r_ax.set_xticks(xticks)
    r_ax.set_xticklabels(xticklabels)

    anchor = (1.01, 1.15) if len(input_properties) == 4 else (0.9, 1.15)
    r_ax.legend(bbox_to_anchor=anchor, ncol=4)
    r_ax.set_xlabel('z', fontsize=14)
    r_ax.set_ylabel('Feature importance', fontsize=14)
    st.write(r_fig)

# Left plot download button
with row2[2]:
    st.write('')
    st.write('')
    fn = 'camels_feature_importance.png'
    img = io.BytesIO()
    fig.savefig(img, format='png')
 
    btn = st.download_button(
       label="Download figure",
       data=img,
       file_name=fn,
       mime="image/png"
    )

# Right plot download button
with row2[6]:
    st.write('')
    st.write('')
    fn = 'camels_feature_importance.png'
    img = io.BytesIO()
    r_fig.savefig(img, format='png')
 
    btn = st.download_button(
       label="Download figure",
       data=img,
       file_name=fn,
       mime="image/png",
       key='r_download'
    )


