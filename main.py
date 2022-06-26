import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yaml

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

col1, col2, col3, = st.columns([1, 1, 2])

with col1:
    st.write('CAMELS feature importance')
    sim = st.selectbox(
        'Which CAMELS simulation suite?',
        ['IllustrisTNG', 'SIMBA']
    )
    output_feature_proper = st.selectbox(
        'Which output feature do you want to predict?',
        ['BH mass', 'Gas mass', 'SFR', 'Stellar mass', 'Stellar metallicity'],
        index=3,
    )

with col2:
    omega_m = st.slider('Omega matter',
        min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    sigma_8 = st.slider('Sigma 8', 
        min_value=0.6, max_value=1.0, value=0.8, step=0.05)
    a_sn_1 = st.slider('Supernova 1',
        min_value=-2.0, max_value=2.0, value=0.0, step=0.5)
    a_agn_1 = st.slider('AGN 1',
        min_value=-2.0, max_value=2.0, value=0.0, step=0.5)
    a_sn_2 = st.slider('Supernova 2',
        min_value=-1.0, max_value=1.0, value=0.0, step=0.25)
    a_agn_2 = st.slider('AGN 2',
        min_value=-1.0, max_value=1.0, value=0.0, step=0.25)

with col3:
    output_feature = short_names[output_feature_proper]
    interpolator, snapshots, input_properties = load_interpolator(sim, output_feature)
    input_features = [str(snap) + prop for snap in snapshots for prop in input_properties]
    params = [omega_m, sigma_8, a_sn_1, a_agn_1, a_sn_2, a_agn_2]
    mean_importance = interpolator.predict([params])[0]

    fig, ax = plt.subplots(1, dpi=400)

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

    # ax.set_title(f'Predicting z=0 {output_feature_proper}')
    # TODO: Plot redshift on x axis
    padding = 0.015 * (np.max(snapshots) - np.min(snapshots))
    ax.set_xlim(np.min(snapshots)-padding, np.max(snapshots)+padding)
    # TODO: Place loc below figure
    ax.legend()
    ax.set_xlabel('Snapshot', fontsize=14)
    ax.set_ylabel('Feature importance', fontsize=14)
    st.write(fig)
    # TODO: Change figure size
# TODO: Install correct sklearn version 0.24.1

