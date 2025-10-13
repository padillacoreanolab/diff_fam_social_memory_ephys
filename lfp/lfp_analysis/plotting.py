import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
from itertools import permutations
import seaborn as sns
from bidict import bidict
from collections import defaultdict
from matplotlib import cm
from itertools import combinations
from lfp.lfp_analysis.LFP_collection import LFPCollection
from lfp.lfp_analysis.LFP_recording import LFPRecording
import lfp.lfp_analysis.event_extraction as ee
from matplotlib.colors import LinearSegmentedColormap


#SPECTRUM LINE GRAPHS
def plot_event_spectrum(lfp_collection, event_averages, mode, regions=None, freq_range = None):
    if mode == "power":
        plot_power_spectrum(lfp_collection, event_averages, regions, freq_range)
    if mode == "coherence":
        plot_coherence_spectrum(lfp_collection, event_averages, regions, freq_range)
    if mode == "granger":
        plot_granger_spectrum(lfp_collection, event_averages, regions, freq_range)


def plot_power_spectrum(lfp_collection, event_averages, regions=None, freq_range=None):
    if regions is None:
        regions = lfp_collection.brain_region_dict.keys()
    if freq_range is None:
        freq_range = [1,101]
    for region in regions:
        plt.figure(figsize=(10, 5))
        for event, averages in event_averages.items():
            # averages = [trials, f, b]
            averages = event_averages[event]
            event_average = np.nanmean(averages, axis=0)
            # event_average = [f,b]; average across all trials
            # calculate sem for the trial average
            event_sem = stats.sem(averages, axis=0, nan_policy="omit")
            region_index = lfp_collection.brain_region_dict[region]
            # pick only the region of interest
            y = event_average[freq_range[0]:freq_range[1], region_index]
            y_sem = event_sem[freq_range[0]:freq_range[1], region_index]
            x = range(freq_range[0],freq_range[1])
            (line,) = plt.plot(x, y, label=event)
            plt.fill_between(x, y - y_sem, y + y_sem, alpha=0.2, color=line.get_color())
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, ymax)
        plt.title(f"{region} power")
        plt.legend()
        plt.show()


def plot_coherence_spectrum(lfp_collection, event_averages, regions=None, freq_range=None):
    if regions is None:
        brain_regions = list(lfp_collection.brain_region_dict.values())
        pair_indices = list(combinations(brain_regions, 2))
    if regions is not None:
        pairs_indices = []
        for region_pair in regions:
            try:
                # Try ordered pair
                pairs_index = [lfp_collection.brain_region_dict[region_pair[0]], lfp_collection.brain_region_dict[region_pair[1]]]
            except KeyError:
                print(f"Warning: Pair {region_pair} not found")
                return None
            pairs_indices.append(pairs_index)
    if freq_range is None:
        freq_range = [1,101]
    for i in range(len(pair_indices)):
        for event, averages in event_averages.items():
            # averages = [trials, f, b, b]
            first_region, second_region = list(pair_indices[i])
            first_region_name = lfp_collection.brain_region_dict.inverse[first_region]
            second_region_name = lfp_collection.brain_region_dict.inverse[second_region]
            averages = event_averages[event]
            event_average = np.nanmean(np.array(averages), axis=0)
            # event_average = [f, b, b]; average across all trials
            # calculate sem for the trial average
            event_sem = stats.sem(averages, axis=0, nan_policy="omit")
            # pick only the region of interest
            y_sem = event_sem[freq_range[0]:freq_range[1], first_region, second_region]
            y = event_average[freq_range[0]:freq_range[1], first_region, second_region]
            x = range(freq_range[0],freq_range[1])
            (line,) = plt.plot(x, y, label=event)
            plt.fill_between(x, y - y_sem, y + y_sem, color=line.get_color(), alpha=0.2)
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, ymax)
        plt.title(f"{first_region_name} & {second_region_name} coherence")
        #plt.legend()
        plt.show()


def plot_granger_spectrum(lfp_collection, event_averages, regions=None, freq_range=None):
    if regions is not None:
        pair_indices = []
        for region in regions:
            first_index = lfp_collection.brain_region_dict[region[0]]
            second_index = lfp_collection.brain_region_dict[region[1]]
            pair_indices.append([first_index, second_index])
    if regions is None:
        pair_indices = list(permutations(range(len(lfp_collection.brain_region_dict.keys())), 2))
        regions = []
        for pair in pair_indices:
            regions.append([lfp_collection.brain_region_dict.inverse[pair[0]], lfp_collection.brain_region_dict.inverse[pair[1]]])
    if freq_range is None:
        freq_range = [1,101]
    for i in range(len(pair_indices)):
        for event, averages in event_averages.items():
            # averages = [trials, b, b, f]
            region = regions[i]
            averages = event_averages[event]
            event_average = np.nanmean(averages, axis=0)
            # event_average = [b,b,f]; average across all trials
            # calculate sem for the trial average
            event_sem = stats.sem(averages, axis=0, nan_policy="omit")
            # pick only the region of interest
            y_sem = event_sem[freq_range[0]:freq_range[1], pair_indices[i][0], pair_indices[i][1]]
            y = event_average[freq_range[0]:freq_range[1], pair_indices[i][0], pair_indices[i][1]]
            x = np.arange(freq_range[0],freq_range[1])
            (line,) = plt.plot(x, y, label=event)
            plt.fill_between(x, y - y_sem, y + y_sem, color=line.get_color(), alpha=0.2)
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, ymax)
        plt.title(f"Granger causality: {region[1]} to {region[0]}")
        plt.legend()
        plt.show()

#HEATMAPS 
def plot_heatmap(lfp_collection, events, freq, color, vmax=None, vmin = None, baselines=None, event_len=None,mode = 'granger'):
    """
    Plot Granger causality heatmaps for multiple events as subplots with a shared colorbar per row.
    
    Parameters:
    -----------
    lfp_collection : LFP collection object
    events : list of strings, event names to plot
    freq : tuple, frequency range (min_freq, max_freq)
    color : list of two colors for colormap gradient
    vmax : float, maximum value for colorbar (applied to all plots)
    baselines : optional, list of baseline events corresponding to each event or single baseline for all
    event_len : optional, event length to analyze
    """
    from matplotlib.gridspec import GridSpec
    import numpy as np
    
    # Handle baselines parameter
    if baselines is None:
        # No baselines specified, use None for all events
        event_baselines = [None] * len(events)
    elif isinstance(baselines, list) and len(baselines) == len(events):
        # List of baselines matching events
        event_baselines = baselines
    else:
        # Single baseline for all events
        event_baselines = [baselines] * len(events)
    
    # Calculate Granger causality for each event with its corresponding baseline
    event_grangers = {}
    for event, baseline in zip(events, event_baselines):
        event_data = ee.average_events(
            lfp_collection, [event], mode=mode, 
            baseline=baseline, event_len=event_len, plot=False
        )
        event_grangers[event] = event_data[event]
    
    n_events = len(events)
    n_cols = min(3, n_events)  # Max 3 columns
    n_rows = (n_events + n_cols - 1) // n_cols  # Ceiling division
    
    # Create custom grid to accommodate colorbars
    fig = plt.figure(figsize=(5 * n_cols + 0.5, 4 * n_rows))
    gs = GridSpec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.05])
    
    # Get brain regions once since they're the same for all plots
    brain_regions = np.empty(len(lfp_collection.brain_region_dict.keys()), dtype="<U10")
    for i in range(len(lfp_collection.brain_region_dict.keys())):
        brain_regions[i] = lfp_collection.brain_region_dict.inverse[i]
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('custom', color, N=100)
    
    # Calculate global vmax if not provided
    if vmax is None:
        all_values = []
        for event in events:
            event_granger = event_grangers[event]
            avg_granger = np.nanmean(event_granger, axis=0)
            freq_granger = avg_granger[freq[0]:freq[1], :, :]
            avg_freq = np.nanmean(freq_granger, axis=0)
            all_values.append(np.nanmax(avg_freq))
        vmax = np.max(all_values)
    if vmin is None:
        all_values = []
        for event in events:
            event_granger = event_grangers[event]
            avg_granger = np.nanmean(event_granger, axis=0)
            freq_granger = avg_granger[freq[0]:freq[1], :, :]
            avg_freq = np.nanmean(freq_granger, axis=0)
            all_values.append(np.nanmin(avg_freq))
        vmin = np.min(all_values)
    # Get all data matrices for consistent color scaling
    data_matrices = []
    for event in events:
        event_granger = event_grangers[event]
        avg_granger = np.nanmean(event_granger, axis=0)
        freq_granger = avg_granger[freq[0]:freq[1], :, :]
        avg_freq = np.nanmean(freq_granger, axis=0)
        data_matrices.append(avg_freq)
    
    # Plot each event
    axes = []
    for idx, event in enumerate(events):
        row = idx // n_cols
        col = idx % n_cols
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)
        
        # Get data
        avg_freq = data_matrices[idx]
        
        # Create heatmap without colorbar
        sns.heatmap(avg_freq, xticklabels=brain_regions, yticklabels=brain_regions,
                   annot=True, cmap=cmap, ax=ax, vmax=vmax, vmin=vmin, cbar=False)
        
        # Add baseline information to title if available
        baseline_info = ""
        if event_baselines[idx] is not None:
            baseline_info = f"\nBaseline: {event_baselines[idx]}"
            
        ax.set_title(f"{event} {mode}\n{freq[0]}Hz to {freq[1]}Hz{baseline_info}")
        
        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_ylabel("From")
        ax.set_xlabel("To")
        
        # Add colorbar at the end of each row
        if col == n_cols - 1 or idx == len(events) - 1:
            cbar_ax = fig.add_subplot(gs[row, -1])
            plt.colorbar(ax.collections[0], cax=cbar_ax)
    
    # Add overall title
    plt.suptitle(f"Granger Causality Analysis ({freq[0]}-{freq[1]}Hz)", 
                fontsize=16, y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    
    return None
   
def plot_granger_heatmap(lfp_collection, events, freq, baseline=None, event_len=None):
    event_granger = ee.average_events(lfp_collection,events, mode="granger", baseline=baseline, event_len=event_len, plot=False)
    n_events = len(events)
    n_cols = min(3, n_events)  # Max 3 columns
    n_rows = (n_events + n_cols - 1) // n_cols  # Ceiling division
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_events == 1:
        axes = np.array([axes])  # Make single axis iterable
    axes = axes.flatten()  # Flatten for easy iteration
    # Get brain regions once since they're the same for all plots
    brain_regions = np.empty(len(lfp_collection.brain_region_dict.keys()), dtype="<U10")
    for i in range(len(lfp_collection.brain_region_dict.keys())):
        brain_regions[i] = lfp_collection.brain_region_dict.inverse[i]
    for idx, (event, ax) in enumerate(zip(events, axes)):
        event_granger = event_granger[event]
        avg_granger = np.nanmean(event_granger, axis=0)
        freq_granger = avg_granger[freq[0] : freq[1], :, :]
        avg_freq = np.nanmean(freq_granger, axis=0)
        sns.heatmap(avg_freq, xticklabels=brain_regions, yticklabels=brain_regions, annot=True, cmap="viridis", ax=ax)
        ax.set_title(f"{event} Granger Causality\n{freq[0]}Hz to {freq[1]}Hz")
        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_ylabel("To")
        ax.set_xlabel("From")
    # Remove any empty subplots
    for idx in range(len(events), len(axes)):
        fig.delaxes(axes[idx])
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    return None
    
#SPECTROGRAMS
def plot_spectrogram(lfp_collection, events, mode, event_len, baseline=None, pre_window = 0, post_window = 0, freq_range=(0,100)):
      # Process collection vs single recording
    if isinstance(lfp_collection, LFPCollection):
        recordings = lfp_collection.recordings
    elif isinstance(lfp_collection, LFPRecording):
        recordings = [lfp_collection]
    elif isinstance(lfp_collection, list): 
        recordings = lfp_collection
    else:
        raise TypeError("lfp_collection must be either LFPCollection or LFPRecording")
    
    # Dictionary to store the average event data for each event type
    event_averages_dict = {}
    # Process each event type
    for i in range(len(events)):
        all_events = []
        for recording in recordings:
            # Events shape: [trials, time, freq, regions]
            recording_events = ee.get_events(recording, events[i], mode, event_len, 
                                                  pre_window, post_window, average=False)
            if baseline is not None:
                adj_averages = ee.__baseline_diff__(
                    recording, recording_events, baseline[i], mode, event_len, 
                    pre_window=0, post_window=0, average = False
                )
                all_events.extend(adj_averages)
            else:
                all_events.extend(recording_events)
        
        # Calculate average across trials for this event type
        average_event = np.nanmean(np.array(all_events), axis=0)
        event_averages_dict[events[i]] = average_event
    
    # Get dimensions and axes from the first event (assuming all have same shape)
    first_event = list(event_averages_dict.values())[0]
    n_timepoints, n_freqs, n_regions = first_event.shape
    time_axis = np.linspace(-pre_window, event_len+post_window, n_timepoints)
    freq_axis = range(freq_range[0], freq_range[1])

    # Get region names
    region_names = [f'{lfp_collection.brain_region_dict.inverse[i]}' for i in range(n_regions)]
    if mode == 'power':
        # Calculate vmin and vmax per brain region across all events
        region_bounds = []
        for region_idx in range(n_regions):
            # Collect data for this region across all events
            region_data = np.array([])
            for event_data in event_averages_dict.values():
                # Extract only the frequency range we want to display 
                sliced_data = event_data[:, freq_range[0]:freq_range[1], region_idx]
                if region_data.size == 0:
                    region_data = sliced_data.flatten()
                else:
                    region_data = np.concatenate([region_data, sliced_data.flatten()])
            # Calculate percentiles for this region's data
            region_vmin = np.percentile(region_data, 5)  # 5th percentile to avoid outliers
            region_vmax = np.percentile(region_data, 95)  # 95th percentile to avoid outliers

            # Store bounds for this region
            region_bounds.append((region_vmin, region_vmax))

        # Setup figure layout - rows are brain regions, columns are event types
        n_rows = n_regions
    else:
        if mode == 'coherence': 
            for i in range(n_regions):
                for j in range(i+1, n_regions):  # Start from i+1 to avoid self-pairs and duplicates
                    region_pairs.append(f"{region_names[i]}_{region_names[j]}")
                    region_pair_indices.append((i, j))
        if mode == 'granger':
            for i in range(n_regions):
                for j in range(n_regions):
                    if i != j:  # Skip self-pairs
                        region_pairs.append(f"{region_names[j]} → {region_names[i]}")
                        region_pair_indices.append((i, j))
        n_pairs = len(region_pairs)

        # Calculate vmin and vmax per region pair across all events
        pair_bounds = []
        for pair_idx, (region_i, region_j) in enumerate(region_pair_indices):
            # Collect data for this region pair across all events
            pair_data = np.array([])
            for event_data in event_averages_dict.values():
                # Extract only the frequency range we want to display
                sliced_data = event_data[:, freq_range[0]:freq_range[1], region_i, region_j]
                if pair_data.size == 0:
                    pair_data = sliced_data.flatten()
                else:
                    pair_data = np.concatenate([pair_data, sliced_data.flatten()])

            # Calculate percentiles for this pair's data
            pair_vmin = np.percentile(pair_data, 5)
            pair_vmax = np.percentile(pair_data, 95)

            # Store bounds for this pair
            pair_bounds.append((pair_vmin, pair_vmax))

        # Setup figure layout - rows are region pairs, columns are event types
        n_rows = n_pairs
        
    n_cols = len(events)
    
    # Adjust figure size based on number of subplots
    fig_size = (15, 10)
    adjusted_width = max(fig_size[0], 4 * n_cols)
    adjusted_height = max(fig_size[1], 3 * n_rows)
    
    # Create figure with extra space for everything
    fig = plt.figure(figsize=(adjusted_width + 4, adjusted_height))
    
    # Create layout with extra space for labels and colorbars
    # Main gridspec for all content
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[0.15, 0.85], wspace=0.0)
    # Left side for region labels
    labels_gs = outer_gs[0].subgridspec(n_rows, 1)
    # Right side for plots and colorbars
    right_gs = outer_gs[1].subgridspec(1, 2, width_ratios=[0.95, 0.015], wspace=0.01)
    # Further divide the plots area into a grid for each region and event
    plots_gs = right_gs[0].subgridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)
    # And the colorbar area into a column for each region
    cbar_gs = right_gs[1].subgridspec(n_rows, 1)
    
    # Create axes
    label_axes = []
    plot_axes = []
    cbar_axes = []
    
    # Create label axes
    for row_idx in range(n_rows):
        label_ax = fig.add_subplot(labels_gs[row_idx, 0])
        label_ax.axis('off')  # Hide axis elements
        label_axes.append(label_ax)
    
    # Create plot axes
    for row_idx in range(n_rows):
        row_axes = []
        for col_idx in range(n_cols):
            ax = fig.add_subplot(plots_gs[row_idx, col_idx])
            row_axes.append(ax)
        plot_axes.append(row_axes)
    
    # Create colorbar axes
    for row_idx in range(n_rows):
        cbar_ax = fig.add_subplot(cbar_gs[row_idx, 0])
        cbar_axes.append(cbar_ax)
    
    # Add the region labels
    for row_idx, region_idx in enumerate(range(n_regions)):
        if row_idx < len(region_names):
            label_axes[row_idx].text(0.5, 0.5, region_names[region_idx],
                                     fontsize=18, fontweight='bold',
                                     rotation=90, ha='center', va='center',
                                     transform=label_axes[row_idx].transAxes)
    
    # Plot spectrograms and add colorbars
    if mode == 'power':
        for row_idx, region_idx in enumerate(range(n_regions)):
            region_vmin, region_vmax = region_bounds[region_idx]

            # Reference to store the last image for colorbar
            region_im = None

            # Plot each event for this region
            for col_idx, event in enumerate(events):
                ax = plot_axes[row_idx][col_idx]
                # Get the event data for this specific region
                event_data = event_averages_dict[event][:, freq_range[0]:freq_range[1], region_idx].T
                # Plot the spectrogram
                im = ax.pcolormesh(time_axis, freq_axis, event_data,
                                  cmap='viridis', vmin=region_vmin, vmax=region_vmax, shading='gouraud')

                region_im = im  # Save for colorbar

                # Add vertical line at t=0 (event onset)
                ax.axvline(x=0, color='white', linestyle='--', alpha=0.7)

                # Set labels only on left and bottom edges
                if col_idx == 0:
                    ax.set_ylabel('Frequency (Hz)')

                if row_idx == n_rows - 1:
                    ax.set_xlabel('Time (s)')

                # Add event type title
                if row_idx == 0:
                    ax.set_title(f"{event}", fontsize=14, fontweight='bold')

            # Add colorbar for this region
            if region_im is not None:
                cbar = plt.colorbar(region_im, cax=cbar_axes[row_idx])
                cbar.set_label('Power', rotation=270, labelpad=15)
            
    else:
        for row_idx, pair_name in enumerate(region_pairs):
            label_axes[row_idx].text(0.5, 0.5, pair_name,
                                     fontsize=14, fontweight='bold',
                                     rotation=0, ha='center', va='center',
                                     transform=label_axes[row_idx].transAxes)
    
    # Plot spectrograms and add colorbars
        for row_idx, (region_i, region_j) in enumerate(region_pair_indices):
            pair_vmin, pair_vmax = pair_bounds[row_idx]

            # Reference to store the last image for colorbar
            pair_im = None

            # Plot each event for this region pair
            for col_idx, event in enumerate(events):
                ax = plot_axes[row_idx][col_idx]

                # Get the event data for this specific region pair
                event_data = event_averages_dict[event][:, freq_range[0]:freq_range[1], region_i, region_j].T

                # Plot the spectrogram
                im = ax.pcolormesh(time_axis, freq_axis, event_data,
                                  cmap='viridis', vmin=pair_vmin, vmax=pair_vmax, shading='gouraud')

                pair_im = im  # Save for colorbar

                # Add vertical line at t=0 (event onset)
                ax.axvline(x=0, color='white', linestyle='--', alpha=0.7)

                # Set labels only on left and bottom edges
                if col_idx == 0:
                    ax.set_ylabel('Frequency (Hz)')

                if row_idx == n_rows - 1:
                    ax.set_xlabel('Time (s)')

                # Add event type title
                if row_idx == 0:
                    ax.set_title(f"{event}", fontsize=14, fontweight='bold')

            # Add colorbar for this region pair
            if pair_im is not None:
                cbar = plt.colorbar(pair_im, cax=cbar_axes[row_idx])
                measure_label = "Coherence" if mode == "coherence" else "Granger Causality"
                cbar.set_label(measure_label, rotation=270, labelpad=15)
    fig.suptitle(f'{mode} Spectrograms', fontsize=16, y=0.98)




def event_power_bar(lfp_collection, events, baseline=None):
    powers = ee.average_events(lfp_collection, events=events, mode="power", baseline=baseline, plot=False)
    [unflipped, flipped] = band_calcs(powers)
    brain_regions = np.empty(len(lfp_collection.brain_region_dict.keys()), dtype="<U10")
    for i in range(len(lfp_collection.brain_region_dict.keys())):
        brain_regions[i] = lfp_collection.brain_region_dict.inverse[i]
    avg_values = {key: {subset: {event: [] for event in events} for subset in brain_regions} for key in flipped.keys()}
    sem_values = {key: {subset: {event: [] for event in events} for subset in brain_regions} for key in flipped.keys()}
    for key in flipped.keys():
        for i, subset in enumerate(brain_regions):
            for event in events:
                avg_values[key][subset][event] = np.nanmean(flipped[key][event][:, i])
                sem_values[key][subset][event] = stats.sem(flipped[key][event][:, i], nan_policy="omit")

    # Adjust bar width and spacing based on number of events
    total_width = 0.8  # Total width available for each group of bars
    bar_width = total_width / len(events)  # Width of each bar
    col = cm.rainbow(np.linspace(0, 1, len(events)))

    # Spacing between groups of bars
    group_spacing = 1  # Increased for better separation between brain regions

    sorted_avg_values = {key: {subset: avg_values[key][subset] for subset in brain_regions} for key in flipped.keys()}
    sorted_sem_values = {key: {subset: sem_values[key][subset] for subset in brain_regions} for key in flipped.keys()}

    # Create a separate plot for each key
    for key in flipped.keys():
        plt.figure(figsize=(25, 10))
        x = np.arange(len(brain_regions)) * group_spacing  # x-axis positions for subsets

        for i, subset in enumerate(brain_regions):
            for k, event in enumerate(events):
                # Center the group of bars and space them evenly
                center = x[i]
                offset = (k - (len(events) - 1) / 2) * bar_width
                position = center + offset

                plt.bar(
                    position,
                    sorted_avg_values[key][subset][event],
                    width=bar_width,
                    yerr=sorted_sem_values[key][subset][event],
                    capsize=5,
                    linewidth=2,
                    error_kw={"elinewidth": 2, "capthick": 2},
                    color=col[k],
                    label=event if i == 0 else "",
                )

        plt.yticks(fontsize=20)
        plt.xticks(x, brain_regions, fontsize=24, rotation=45)
        plt.axhline(y=0, color="black", linestyle="--", alpha=0.8)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_linewidth(2)
        plt.gca().spines["left"].set_linewidth(2)
        plt.title(f"Average Power for {key}", fontsize=40)
        plt.legend(fontsize=26, frameon=False)
        plt.subplots_adjust(hspace=0.5)
        plt.show()


def event_coherence_bar(lfp_collection, events, baseline=None):
    region_dict = lfp_collection.brain_region_dict
    brain_regions = list(combinations(list((region_dict.keys())), 2))  # Example subset names
    coherences = ee.average_events(lfp_collection, events=events, mode="coherence", baseline=baseline, plot=False)
    plot_bars(lfp_collection, brain_regions, coherences, events, title = 'Coherence')
    return None
  
def event_granger_bar(lfp_collection, events, baseline=None):
    region_dict = lfp_collection.brain_region_dict
    brain_regions = list(permutations(region_dict.keys(), 2))  # Ordered pairs

    grangers = ee.average_events(lfp_collection, events=events, mode="granger", baseline=baseline, plot=False)
    
    plot_bars(lfp_collection, brain_regions, grangers, events, title= 'Granger')
    return None

def plot_bars(lfp_collection, brain_regions, values, events, title):
    region_dict = lfp_collection.brain_region_dict
    [unflipped, flipped] = band_calcs(values)
    avg_values = {key: {subset: {event: [] for event in events} for subset in brain_regions} for key in flipped.keys()}
    sem_values = {key: {subset: {event: [] for event in events} for subset in brain_regions} for key in flipped.keys()}

    for key in flipped.keys():
        for i, subset in enumerate(brain_regions):
            pair_index_1 = region_dict[brain_regions[i][0]]
            pair_index_2 = region_dict[brain_regions[i][1]]
            for event in events:
                avg_values[key][subset][event] = np.nanmean(flipped[key][event][:, pair_index_1, pair_index_2])
                sem_values[key][subset][event] = stats.sem(
                    flipped[key][event][:, pair_index_1, pair_index_2], nan_policy="omit"
                )

    # Width of each bar
    col = cm.rainbow(np.linspace(0, 1, len(events)))
    spacing = 0
    # Create a separate plot for each key
    # Spacing between different subsets
    total_width = 0.8  # Total width available for each group of bars
    bar_width = total_width / len(events)
    group_spacing = 1
    # Create a separate plot for each key
    for key in flipped.keys():
        plt.figure(figsize=(25, 10))
        x = np.arange(len(brain_regions)) * group_spacing  # x-axis positions for subsets

        for i, subset in enumerate(brain_regions):
            for k, event in enumerate(events):
                positions = x[i] + (k - 1) * (bar_width + spacing)  # Adjust positions for each event
                plt.bar(
                    positions,
                    avg_values[key][subset][event],
                    width=bar_width,
                    yerr=sem_values[key][subset][event],
                    capsize=5,
                    linewidth=2,
                    error_kw={"elinewidth": 2, "capthick": 2},
                    color=col[k],
                    label=event if i == 0 else "",
                )

        plt.yticks(fontsize=16)
        if title == 'Granger':
            region_labels = [f"{pair[1]} to {pair[0]}" for pair in brain_regions]
        if title == 'Coherence':
            region_labels = [f"{pair[0]} &{pair[1]}" for pair in brain_regions]
        plt.xticks(x, region_labels, rotation=45, ha='right', fontsize = 18)
        plt.axhline(y=0, color="black", linestyle="--", alpha=0.8)
        plt.ylabel("Average Coherence", fontsize=20)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_linewidth(2)  # X-axis
        plt.gca().spines["left"].set_linewidth(2)
        plt.title(f"Average {title} for {key}", fontsize=26)
        plt.legend(fontsize=16, frameon=False)
        plt.subplots_adjust(hspace=0.5)
        plt.show()

def band_calcs(values):
    agent_band_dict = {}
    for agent, calculations in values.items():
        calculations = np.array(calculations)
        # calculations = [trials, frequencies, brain regions]
        delta = np.nanmean(calculations[:, 0:4, ...], axis=1)
        theta = np.nanmean(calculations[:, 4:13, ...], axis=1)

        beta = np.nanmean(calculations[:, 13:31, ...], axis=1)

        low_gamma = np.nanmean(calculations[:, 31:71, ...], axis=1)

        high_gamma = np.nanmean(calculations[:, 71:100, ...], axis=1)

        agent_band_dict[agent] = {
            "Delta": delta,
            "Theta": theta,
            "Beta": beta,
            "Low gamma": low_gamma,
            "High gamma": high_gamma,
        }

    band_agent_dict = defaultdict(dict)
    for agent, bands in agent_band_dict.items():
        for band, values in bands.items():
            band_agent_dict[band][agent] = values

    return [agent_band_dict, band_agent_dict]

def diff_band_calcs(values, freq_range_dict):
    agent_band_dict = {}
    for agent, calculations in values.items():
        agent_band_dict[agent] = {}
        for name, freq_range in freq_range_dict.items():
            calculations = np.array(calculations)
            # calculations = [trials, frequencies, brain regions]
            temp = np.nanmean(calculations[:, freq_range[0]:freq_range[1], ...], axis=1)
            agent_band_dict[agent][name] = temp

    band_agent_dict = defaultdict(dict)
    for agent, bands in agent_band_dict.items():
        for band, values in bands.items():
            band_agent_dict[band][agent] = values

    return [agent_band_dict, band_agent_dict]