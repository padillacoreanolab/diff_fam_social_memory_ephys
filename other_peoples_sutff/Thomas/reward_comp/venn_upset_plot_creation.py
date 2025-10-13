import spike.spike_analysis.spike_collection as sc
import spike.spike_analysis.spike_recording as sr
import spike.spike_analysis.firing_rate_calculations as fr
import spike.spike_analysis.normalization as norm
import spike.spike_analysis.single_cell as single_cell
import spike.spike_analysis.spike_collection as collection
import spike.spike_analysis.zscoring as zscoring
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import os
import behavior.boris_extraction as boris
import matplotlib.pyplot as plt
import pickle
import re
import ast


import matplotlib_venn
from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
from itertools import combinations

# Getting Counts of Significant Neurons for an Event, helper function for venn/upset plots
def get_significant_units_for_event(df, event_name, significance_type='both'):
    """
    Get set of significant units for a specific event. You can optionally filter for a significance type.
    By default, it includes both 'increase' and 'decrease'. Should work for dataframes that aren't already 
    filtered for significance.
    
    Returns:
    set: Set of unique unit identifiers that are significant for this event
    """
    event_df = df[df['Event name'] == event_name]
    
    if significance_type == 'both':
        significant_units = event_df[event_df['sig'].isin(['increase', 'decrease'])]['Unit number'].tolist()

        inc = event_df[event_df['sig'] == 'increase']['Unit number'].tolist()
        dec = event_df[event_df['sig'] == 'decrease']['Unit number'].tolist()

        print("Significant units for increase:", inc)
        print("Significant units for decrease:", dec)
        print("All significant units:", significant_units)
    elif significance_type == 'increase':
        significant_units = event_df[event_df['sig'] == 'increase']['Unit number'].tolist()
        print("Significant units for increase:", significant_units)
    elif significance_type == 'decrease':
        significant_units = event_df[event_df['sig'] == 'decrease']['Unit number'].tolist()
        print("Significant units for decrease:", significant_units)
    else:
        raise ValueError("significance_type must be 'both', 'increase', or 'decrease'")
        
    return significant_units

# Plot Creation Function, finds number of significant units for every event to compare then uses plotting functions
def create_overlap_visualization(df, compare_events, significance_type='both', title=""):
    """
    Create visualization for overlapping significant neurons between events.
    Uses Venn diagrams for 2-4 events, UpSet plots for 5+ events.

    compare_events: list of events to plot/compare
    """
    
    # Get sets of significant units for each event
    event_dicts = {}
    for event in compare_events:
        event_dicts[event] = get_significant_units_for_event(df, event, significance_type)
        print(f"{event}: {len(event_dicts[event])} significant units")

    # map identifiers of each unit 1-n to so when we convert to a set we don't lose units from different subjects/recordings with the same unit number
    # Extract subject ID from recording name (e.g., "20230621_111240_standard_comp_to_omission_D5_subj_1-4_t3b3L_box1_merged.rec" -> "subj_1-4")
    event_sets = {}
    for event in compare_events:
        # Filter dataframe to only include significant units for this event and significance type
        event_df = df[df['Event name'] == event]
        if significance_type == 'both':
            filtered_event_df = event_df[event_df['sig'].isin(['increase', 'decrease'])]
        elif significance_type == 'increase':
            filtered_event_df = event_df[event_df['sig'] == 'increase']
        elif significance_type == 'decrease':
            filtered_event_df = event_df[event_df['sig'] == 'decrease']
        else:
            raise ValueError("significance_type must be 'both', 'increase', or 'decrease'")
        
        unique_identifiers = []
        
        for _, row in filtered_event_df.iterrows():
            rec = row['Recording']
            unit = row['Unit number']
            
            # Extract multiple components for a more unique identifier
            # Expected format: "YYYYMMDD_HHMMSS_...subj_X-Y_...merged.rec"
            
            # Extract date (first 8 digits)
            date_match = re.search(r'^(\d{8})', rec)
            date_part = date_match.group(1) if date_match else "unknown_date"
            
            # Extract subject ID using regex to find "subj_X-Y" pattern
            subject_match = re.search(r'subj_[\d-]+', rec)
            subject_id = subject_match.group() if subject_match else "unknown_subj"
            
            # Extract additional session info (like box, condition markers)
            # Look for patterns like "box1", "t3b3L", etc.
            box_match = re.search(r'box\d+', rec)
            box_part = box_match.group() if box_match else ""
            
            session_match = re.search(r't\d+b\d+[A-Z]*', rec)
            session_part = session_match.group() if session_match else ""
            
            # Create comprehensive unique identifier
            # Format: date_subject_session_box_Unit{unit} or fallback to recording_Unit{unit}
            if subject_match and date_match:
                identifier_parts = [date_part, subject_id]
                if session_part:
                    identifier_parts.append(session_part)
                if box_part:
                    identifier_parts.append(box_part)
                identifier_parts.append(f"Unit{unit}")
                unique_identifier = "_".join(identifier_parts)
            else:
                # Fallback to using sanitized recording name if pattern extraction fails
                sanitized_rec = re.sub(r'[^\w\-]', '_', rec.replace('.rec', ''))
                unique_identifier = f"{sanitized_rec}_Unit{unit}"
            
            unique_identifiers.append(unique_identifier)
        
        event_sets[event] = set(unique_identifiers)

    if len(compare_events) <= 4:
        # Use Venn diagrams for 2-4 events
        return _create_venn_diagram(event_sets, compare_events, significance_type, title)
    else:
        # Use UpSet plot for 5+ events
        return _create_upset_plot(event_sets, compare_events, significance_type, title)

# Actual plotting function for venn diagrams of different event sizes
def _create_venn_diagram(event_sets, compare_events, significance_type, title):
    """Helper function for traditional Venn diagrams (2-4 events)"""
    
    if len(compare_events) == 2:
        # Two-way Venn diagram
        plt.figure(figsize=(10, 8))
        venn = venn2([event_sets[compare_events[0]], event_sets[compare_events[1]]], 
                     set_labels=compare_events)
        venn2_circles([event_sets[compare_events[0]], event_sets[compare_events[1]]], linewidth=1)
        
        # Increase font size of the numbers in the circles
        for text in venn.subset_labels:
            if text:
                text.set_fontsize(35)
        
        # Calculate overlap: Events A intersection B
        intersection = event_sets[compare_events[0]] & event_sets[compare_events[1]]
        print("Significant units for overlap:", intersection)
        print(f"\nOverlap between {compare_events[0]} and {compare_events[1]}: {len(intersection)} units")

        # Print units in each event
        print(f"units in {compare_events[0]}: {event_sets[compare_events[0]]}")
        print(f"units in {compare_events[1]}: {event_sets[compare_events[1]]}")

    # Three-way Venn diagram
    elif len(compare_events) == 3:
        plt.figure(figsize=(10, 8))
        venn = venn3([event_sets[compare_events[0]], event_sets[compare_events[1]], event_sets[compare_events[2]]], 
                     set_labels=compare_events)
        venn3_circles([event_sets[compare_events[0]], event_sets[compare_events[1]], event_sets[compare_events[2]]], linewidth=1)
        
        # Increase font size of the numbers in the circles
        for text in venn.subset_labels:
            if text:
                text.set_fontsize(35)
        
        # Calculate overlap statistics, making sure plot is accurate
        all_intersection = event_sets[compare_events[0]] & event_sets[compare_events[1]] & event_sets[compare_events[2]]
        pairwise_overlaps = []
        for i, j in combinations(range(3), 2):
            overlap = event_sets[compare_events[i]] & event_sets[compare_events[j]]
            pairwise_overlaps.append((compare_events[i], compare_events[j], len(overlap)))
            
        print(f"\nThree-way overlap: {len(all_intersection)} units")
        print(f"Three-way overlap units: {sorted(list(all_intersection))}")
        
        print("\nPairwise overlaps:")
        for event1, event2, overlap_size in pairwise_overlaps:
            print(f"  {event1} & {event2}: {overlap_size} units")
            
    elif len(compare_events) == 4:
        # Four-way "Venn-style" diagram using subplots
        return _create_four_way_venn(event_sets, compare_events, significance_type, title)
    
    if len(compare_events) <= 3:
        # Create simple title for 2-3 way diagrams
        simple_title = ' vs '.join(compare_events)
        
        plt.title(simple_title, fontsize=10, pad=20)
        plt.show()
    
    return event_sets

def _create_four_way_venn(event_sets, compare_events, significance_type, title):
    """Create a 4-way Venn diagram using pairwise comparisons"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Create pairwise Venn diagrams
    pairs = [
        (0, 1, ax1), (0, 2, ax2), (0, 3, ax3), (1, 2, ax4)
    ]
    
    pairwise_stats = []
    
    for i, j, ax in pairs:
        # Create Venn diagram for this pair
        plt.sca(ax)
        venn = venn2([event_sets[compare_events[i]], event_sets[compare_events[j]]], 
                     set_labels=[compare_events[i], compare_events[j]])
        venn2_circles([event_sets[compare_events[i]], event_sets[compare_events[j]]], linewidth=1)
        
        # Increase font size of the numbers in the circles
        for text in venn.subset_labels:
            if text:
                text.set_fontsize(35)
        
        # Calculate overlap
        intersection = event_sets[compare_events[i]] & event_sets[compare_events[j]]
        pairwise_stats.append((compare_events[i], compare_events[j], len(intersection)))
        
        ax.set_title(f"{compare_events[i]} vs {compare_events[j]}\nOverlap: {len(intersection)} units")
    
    # Calculate higher-order intersections
    all_four = event_sets[compare_events[0]] & event_sets[compare_events[1]] & event_sets[compare_events[2]] & event_sets[compare_events[3]]
    three_way_intersections = []
    
    for combo in combinations(range(4), 3):
        three_way = event_sets[compare_events[combo[0]]] & event_sets[compare_events[combo[1]]] & event_sets[compare_events[combo[2]]]
        three_way_intersections.append((combo, len(three_way)))
    
    # Simple overall title
    simple_title = ' vs '.join(compare_events)
    
    fig.suptitle(simple_title, fontsize=10, y=0.95)
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    '''
    print(f"\n=== FOUR-WAY OVERLAP STATISTICS ===")
    print(f"Four-way intersection (all events): {len(all_four)} units")
    if len(all_four) > 0:
        print(f"Units in all four events: {sorted(list(all_four))}")
    
    print(f"\nThree-way intersections:")
    for combo, size in three_way_intersections:
        combo_names = [compare_events[i] for i in combo]
        if size > 0:
            print(f"  {' ∩ '.join(combo_names)}: {size} units")
    
    print(f"\nPairwise overlaps:")
    for event1, event2, size in pairwise_stats:
        print(f"  {event1} & {event2}: {size} units")
    
    # Also show remaining pairwise combinations not in the subplot
    remaining_pairs = [(1, 3), (2, 3)]
    for i, j in remaining_pairs:
        intersection = event_sets[compare_events[i]] & event_sets[compare_events[j]]
        print(f"  {compare_events[i]} & {compare_events[j]}: {len(intersection)} units")
    '''
    
    return event_sets

def _create_upset_plot(event_sets, compare_events, significance_type, title):
    """Helper function for UpSet plots (5+ events)"""
    # Create a binary matrix for UpSet plot
    all_units = set()
    for event_set in event_sets.values():
        all_units.update(event_set)
    
    all_units = sorted(list(all_units))
    
    # Create binary membership matrix
    membership_data = []
    for unit in all_units:
        row = {}
        for event in compare_events:
            row[event] = unit in event_sets[event]
        membership_data.append(row)
    
    membership_df = pd.DataFrame(membership_data, index=all_units)
    
    # Create UpSet plot using matplotlib
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                   gridspec_kw={'height_ratios': [1, 2]})
    
    # Calculate intersection sizes
    intersections = {}
    n_events = len(compare_events)
    
    # Generate all possible combinations
    for r in range(1, n_events + 1):
        for combo in combinations(compare_events, r):
            # Find units that are in ALL events in this combo and NONE of the others
            units_in_combo = set(all_units)
            for event in combo:
                units_in_combo &= event_sets[event]
            
            for event in compare_events:
                if event not in combo:
                    units_in_combo -= event_sets[event]
            
            intersections[combo] = len(units_in_combo)
    
    # Sort by intersection size
    sorted_intersections = sorted(intersections.items(), key=lambda x: x[1], reverse=True)
    
    # Plot intersection sizes (top 15 to keep readable)
    top_intersections = sorted_intersections[:15]
    combo_names = [' ∩ '.join(combo) for combo, size in top_intersections]
    sizes = [size for combo, size in top_intersections]
    
    ax1.bar(range(len(sizes)), sizes)
    ax1.set_xticks(range(len(sizes)))
    ax1.set_xticklabels(combo_names, rotation=45, ha='right')
    ax1.set_ylabel('Intersection Size')
    ax1.set_title(f'Top 15 Intersections - {len(compare_events)} Events')
    
    # Plot set sizes
    event_sizes = [len(event_sets[event]) for event in compare_events]
    ax2.bar(compare_events, event_sizes)
    ax2.set_ylabel('Set Size')
    ax2.set_xlabel('Events')
    ax2.tick_params(axis='x', rotation=45)
    
    # Simple overall title
    simple_title = ' vs '.join(compare_events)
    
    fig.suptitle(simple_title, fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\n=== DETAILED OVERLAP STATISTICS ===")
    print(f"Total unique units across all events: {len(all_units)}")
    print(f"Individual event sizes: {dict(zip(compare_events, event_sizes))}")
    
    print(f"\nTop 10 intersections:")
    for i, (combo, size) in enumerate(top_intersections[:10]):
        if size > 0:
            print(f"  {i+1}. {' ∩ '.join(combo)}: {size} units")
    
    # Calculate pairwise overlaps for comparison
    print(f"\nPairwise overlaps:")
    for i, j in combinations(range(len(compare_events)), 2):
        overlap = event_sets[compare_events[i]] & event_sets[compare_events[j]]
        print(f"  {compare_events[i]} & {compare_events[j]}: {len(overlap)} units")
    
    return event_sets

# Example usage function
def analyze_event_overlap(df, event_groups, significance_type='both'):
    """
    Analyze overlaps for multiple groups of events.
    
    Parameters:
    df: DataFrame with neural data
    event_groups: List of lists, each containing 2+ events to compare
    significance_type: 'both', 'increase', 'decrease'
    """
    
    for i, events in enumerate(event_groups):
        print(f"\n{'='*50}")
        print(f"ANALYSIS {i+1}: {' vs '.join(events)} ({len(events)} events)")
        print('='*50)
        
        try:
            event_sets = create_overlap_visualization(df, events, significance_type)
        except Exception as e:
            print(f"Error creating overlap visualization: {e}")
            continue

# Keep the old function name for backward compatibility
def create_venn_diagram(df, compare_events, significance_type='both', title=""):
    """Legacy function name - redirects to create_overlap_visualization"""
    return create_overlap_visualization(df, compare_events, significance_type, title)