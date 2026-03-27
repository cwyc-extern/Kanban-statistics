import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from upsetplot import UpSet, from_indicators
import os
from datetime import date

# =============================
# Section 1 - Run Configuration
# =============================

# Create outputs directory if it doesn't exist.
os.makedirs('outputs', exist_ok=True)

# Output folder for this run (Q1 2026)
output_subfolder = 'outputs/q1_2026'
os.makedirs(output_subfolder, exist_ok=True)

# User-configurable data scope for report and graphics.
# Set to None to disable a filter.
selected_sprint = None
selected_quarter = 'Q1'
selected_year = 2026
#
# Handover note:
# These three variables are the only values a collaborator should change
# to rerun the full pipeline for a different reporting scope.

# Version tag for generated files (ddmmyyyy).
run_date_tag = date.today().strftime('%d%m%Y')
def with_tag(filename: str) -> str:
    base, ext = os.path.splitext(filename)
    return f'{base}-{run_date_tag}{ext}'

# ==========================
# Section 2 - Data Utilities
# ==========================

def load_filtered_data(input_file: str):
    """
    Load source data and apply user-selected sprint/quarter/year filters.

    Handover note:
    Keep filtering centralized here so all outputs use identical scope.
    """
    print(f"Reading Excel file: {input_file}")
    df_full_local = pd.read_excel(input_file)
    print("\nFiltering data using selected sprint/quarter/year")
    print(f"Original shape: {df_full_local.shape}")

    df_filtered = df_full_local.copy()

    if selected_sprint is not None:
        if 'sprint' in df_filtered.columns:
            df_filtered = df_filtered[
                df_filtered['sprint'].astype(str).str.strip() == str(selected_sprint).strip()
            ].copy()
        else:
            print("\nWarning: 'sprint' column not found; sprint filter skipped.")

    if selected_quarter is not None:
        if 'quarter' in df_filtered.columns:
            df_filtered = df_filtered[
                df_filtered['quarter'].astype(str).str.strip().str.upper()
                == str(selected_quarter).strip().upper()
            ].copy()
        else:
            print("\nWarning: 'quarter' column not found; quarter filter skipped.")

    df_filtered['start_year'] = pd.to_datetime(df_filtered.get('start_date'), errors='coerce').dt.year
    df_filtered['end_year'] = pd.to_datetime(df_filtered.get('end_date'), errors='coerce').dt.year
    if selected_year is not None:
        if df_filtered['start_year'].notna().any() or df_filtered['end_year'].notna().any():
            df_filtered = df_filtered[
                (df_filtered['start_year'] == selected_year) | (df_filtered['end_year'] == selected_year)
            ].copy()
        else:
            print("\nWarning: Could not determine year from 'start_date'/'end_date'; year filter skipped.")

    print(f"Filtered shape (selected scope): {df_filtered.shape}")
    return df_full_local, df_filtered

def _scope_filter_text() -> str:
    parts = []
    if selected_sprint is not None:
        parts.append(f"sprint == {selected_sprint}")
    if selected_quarter is not None:
        parts.append(f"quarter == {selected_quarter}")
    if selected_year is not None:
        parts.append(f"year == {selected_year}")
    return ' and '.join(parts) if parts else 'no filters (full dataset)'

def print_dataframe_info(dataframe):
    """Print standardized dataframe diagnostics for handover/debugging."""
    print("\n" + "="*60)
    print("DataFrame Information")
    print("="*60)
    print(f"Shape: {dataframe.shape}")
    print("\nColumn names:")
    for i, col in enumerate(dataframe.columns.tolist(), 1):
        print(f"  {i}. {col}")

# ==========================
# Section 3 - Data Loading
# ==========================

file_path = 'kanban-statistics-v0.xlsx'
df_full, df = load_filtered_data(file_path)
print_dataframe_info(df)

# Find columns starting with "type_"
type_columns = [col for col in df.columns if col.startswith('type_')]
print(f"\nFound {len(type_columns)} columns starting with 'type_':")
for col in type_columns:
    print(f"  - {col}")

# 1. Create upset chart for columns starting with "type_"
if type_columns:
    print("\n" + "="*60)
    print("Creating Upset Chart for Type Categories")
    print("="*60)
    
    # Explicit category order: reversed so "data" appears on top in UpSet rendering
    type_order = [
        'type_consult',
        'type_application',
        'type_code',
        'type_data',
    ]
    type_order = [c for c in type_order if c in type_columns]
    type_label_map = {
        'type_data': 'data',
        'type_code': 'code',
        'type_application': 'data application',
        'type_consult': 'consultation or review',
    }
    
    # Create binary indicators for each type_ column and keep ticket id for traceability
    indicator_cols = ['id'] + type_order
    type_data = df[indicator_cols].copy()
    
    # Ensure each ticket (row) is represented exactly once via its unique id
    type_data['id'] = type_data['id'].astype(str)
    type_data.set_index('id', inplace=True)
    type_data.index.name = 'ticket_id'
    
    # Convert categorical values to boolean indicators: 'Y' -> True, 'NA' -> False
    for col in type_order:
        normalized = (
            type_data[col]
            .fillna('NA')
            .astype(str)
            .str.strip()
            .str.upper()
        )
        type_data[col] = normalized.map({'Y': True, 'NA': False})
        # Any other value defaults to False
        type_data[col] = type_data[col].fillna(False)
    
    # Rename only the requested categories and keep explicit order
    type_data_renamed = type_data.rename(columns=type_label_map)
    renamed_columns = [type_label_map.get(col, col) for col in type_order]

    # Build issue/enhancement flags for coloring subsets (without showing those rows).
    def _flag_from_column(column_name):
        if column_name not in df.columns:
            return pd.Series(False, index=type_data.index)
        tmp = df[['id', column_name]].copy()
        tmp['id'] = tmp['id'].astype(str)
        tmp = tmp.drop_duplicates(subset=['id']).set_index('id')
        normalized = (
            tmp[column_name]
            .fillna('NA')
            .astype(str)
            .str.strip()
            .str.upper()
            .eq('Y')
        )
        return normalized.reindex(type_data.index).fillna(False)

    issue_flag = _flag_from_column('type_issue')
    enhancement_flag = _flag_from_column('type_enhancement')

    subset_color_flags = pd.concat(
        [
            type_data_renamed[renamed_columns],
            pd.DataFrame(
                {
                    '_issue_flag': issue_flag,
                    '_enhancement_flag': enhancement_flag,
                },
                index=type_data.index,
            ),
        ],
        axis=1,
    )
    subset_color_flags = subset_color_flags.groupby(renamed_columns)[['_issue_flag', '_enhancement_flag']].any()
    
    # Build the multi-index Series that includes every ticket
    type_series = from_indicators(renamed_columns, type_data_renamed[renamed_columns])
    
    # Create the UpSet plot using all tickets (each ticket_id contributes exactly once)
    # Larger fonts for inclusion in PPT slides
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    
    upset = UpSet(
        type_series,
        subset_size='count',
        show_counts=False,
        sort_by='degree',
        sort_categories_by=None,
        element_size=42,
    )
    fig = plt.figure(figsize=(20, 10))

    # Category styling requested by user
    medium_red = '#C44E52'
    teal = '#1F9E9E'
    outline_categories = [
        type_label_map.get('type_data', 'type_data'),
        type_label_map.get('type_code', 'type_code'),
        type_label_map.get('type_application', 'type_application'),
        type_label_map.get('type_consult', 'type_consult'),
    ]

    for cat in outline_categories:
        if cat in renamed_columns:
            upset.style_categories(
                cat,
                bar_facecolor='white',
                bar_edgecolor='black',
                bar_linewidth=1.8,
            )
            upset.style_subsets(
                present=cat,
                facecolor='white',
                edgecolor='black',
                linewidth=1.2,
            )

    # Color user-story subsets by hidden issue/enhancement tags:
    # issue -> medium red, enhancement -> teal.
    # If both are present, issue (red) takes precedence.
    for subset_index in upset.intersections.index:
        if subset_index not in subset_color_flags.index:
            continue
        flags = subset_color_flags.loc[subset_index]
        color = None
        if bool(flags['_issue_flag']):
            color = medium_red
        elif bool(flags['_enhancement_flag']):
            color = teal
        if color is None:
            continue

        if not isinstance(subset_index, tuple):
            subset_index = (subset_index,)
        present = [name for name, is_present in zip(renamed_columns, subset_index) if bool(is_present)]
        absent = [name for name, is_present in zip(renamed_columns, subset_index) if not bool(is_present)]
        upset.style_subsets(
            present=present,
            absent=absent,
            facecolor=color,
            edgecolor=color,
            linewidth=1.2,
        )
    
    upset.plot(fig=fig)
    
    # Adjust font sizes and positioning for all text elements in the plot
    axes = fig.get_axes()
    
    # Find and update the intersection size axis label
    # The intersection size axis is typically the rightmost axis (horizontal bar chart)
    # Identify it by position (rightmost) or by checking for "Intersection size" label
    rightmost_x = -1
    rightmost_axis = None
    intersection_axis = None
    
    for ax in axes:
        pos = ax.get_position()
        xlabel = ax.get_xlabel().lower() if ax.get_xlabel() else ''
        
        # Check if this is the intersection size axis by label
        if ('intersection' in xlabel or 'size' in xlabel) and intersection_axis is None:
            intersection_axis = ax
        
        # Track rightmost axis as fallback
        if pos.x0 > rightmost_x:
            rightmost_x = pos.x0
            rightmost_axis = ax
    
    # If not found by label, use the rightmost axis
    if intersection_axis is None:
        intersection_axis = rightmost_axis
    
    # show_counts=False removes intersection-size count labels.
    
    # Update all axes
    for ax in axes:
        ax.tick_params(labelsize=11)
        # Set the intersection size axis label
        if ax == intersection_axis:
            ax.set_xlabel('Nr. of requests', fontsize=13)
        elif ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=13)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=13)
        
        # Adjust y-axis tick labels (category names) - right align next to bars
        yticklabels = ax.get_yticklabels()
        if len(yticklabels) > 0:
            for label in yticklabels:
                label.set_fontsize(11)
                label.set_ha('right')  # Right align so labels align with bars
                label.set_va('center')  # Center align vertically
        
        # Adjust x-axis tick labels
        for label in ax.get_xticklabels():
            label.set_fontsize(11)
        
        # Adjust text elements (category labels) - right align
        for text in ax.texts:
            text.set_fontsize(11)
            text.set_ha('right')  # Right align text elements
    
    # Adjust spacing to prevent label overlap with plot
    # Find the category axis (usually the first one with y-axis labels)
    category_axis = None
    for ax in axes:
        yticklabels = ax.get_yticklabels()
        if len(yticklabels) > 0 and category_axis is None:
            category_axis = ax
            # Reduce padding to bring labels closer to bars, but ensure they don't overlap
            ax.yaxis.set_tick_params(pad=4, labelsize=11)
            break
    
    # Adjust layout to give enough space for right-aligned labels on the left
    plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.1)
    
    # Ensure the intersection size axis label is set (do this last to ensure it sticks)
    if intersection_axis is not None:
        intersection_axis.set_xlabel('Nr. of requests', fontsize=13)
        intersection_axis.xaxis.label.set_fontsize(13)
        # Remove horizontal grid lines on intersection-size graph.
        intersection_axis.grid(False)
        intersection_axis.yaxis.grid(False)

    # Add category totals to the right of bars with ~2 px padding.
    # The category totals axis has y tick labels matching category names and horizontal bars.
    for ax in axes:
        yticklabels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
        if not yticklabels:
            continue
        bars = getattr(ax, "patches", [])
        if len(bars) != len(yticklabels) or len(bars) == 0:
            continue

        for bar in bars:
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2.0
            value = int(round(abs(width)))

            # Place label on the visual right side of the bar with 2 px padding.
            x_start_data = bar.get_x()
            x_end_data = bar.get_x() + width
            start_disp = ax.transData.transform((x_start_data, y))
            end_disp = ax.transData.transform((x_end_data, y))
            right_edge_disp_x = max(start_disp[0], end_disp[0])
            label_disp = (right_edge_disp_x + 2.0, start_disp[1])  # 2 px padding
            label_data = ax.transData.inverted().transform(label_disp)

            ax.text(
                label_data[0],
                label_data[1],
                f"{value}",
                va='center',
                ha='left',
                fontsize=11,
                color='black'
            )
        break

    # Legend in the top-right corner for subset colors.
    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', label='issue',
                   markerfacecolor=medium_red, markeredgecolor=medium_red, markersize=9),
        plt.Line2D([0], [0], marker='s', color='w', label='enhancement or\ncode customization',
                   markerfacecolor=teal, markeredgecolor=teal, markersize=9),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.995, 0.995),
        frameon=True,
        edgecolor='black',
        framealpha=1.0,
        fontsize=10,
    )
    legend.get_frame().set_linewidth(1.0)

    
    plt.savefig(with_tag(f'{output_subfolder}/user-story-distr-deliverables-Q1_2026.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reset font size to default
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    print(f"Upset chart saved to {with_tag(f'{output_subfolder}/user-story-distr-deliverables-Q1_2026.png')}")
else:
    print("\nWarning: No columns starting with 'type_' found")

# 2. Create combined plot with dual y-axes for department distribution
print("\n" + "="*60)
print("Creating Department Distribution Plots")
print("="*60)

# Find department column (case-insensitive)
dept_col = None
story_col = None
for col in df_full.columns:
    if 'department' in col.lower():
        dept_col = col
        break

# Find story points column (case-insensitive)
for col in df_full.columns:
    if 'story' in col.lower() and 'point' in col.lower():
        story_col = col
        break

if not story_col:
    # Try alternative names
    for col in df_full.columns:
        if 'point' in col.lower():
            story_col = col
            break

def create_distribution_requester_chart(dataframe, output_filename, description):
    """Create a dual y-axis bar chart showing number of requests and story points by requester."""
    if not dept_col or not story_col:
        print(f"Warning: Cannot create {description} - missing required columns")
        return False
    
    # Total number of tickets per department
    dept_counts = dataframe[dept_col].value_counts().sort_values(ascending=False)
    
    # Calculate total story points per department
    df_story = dataframe[[dept_col, story_col]].copy()
    df_story[story_col] = pd.to_numeric(df_story[story_col], errors='coerce')
    df_story = df_story.dropna(subset=[story_col])
    dept_story_points = df_story.groupby(dept_col)[story_col].sum().sort_values(ascending=False)
    zero_sp_counts = df_story[df_story[story_col] == 0].groupby(dept_col).size()
    
    # Create combined plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Align both datasets by requester (use dept_counts index as base)
    # Get story points for each requester, defaulting to 0 if not present
    aligned_story_points = []
    for req in dept_counts.index:
        if req in dept_story_points.index:
            aligned_story_points.append(dept_story_points[req])
        else:
            aligned_story_points.append(0)
    aligned_story_points = pd.Series(aligned_story_points, index=dept_counts.index)
    
    # Reorder requester categories:
    # - start with ODT
    # - then all requesters starting with 'L' alphabetically
    # - then all remaining requesters alphabetically
    requester_names = [str(x) for x in dept_counts.index.tolist()]
    odt_name = 'ODT'
    l_names = sorted([r for r in requester_names if r.startswith('L')])
    other_names = sorted([r for r in requester_names if r != odt_name and not r.startswith('L')])
    ordered_requesters = ([odt_name] if odt_name in requester_names else []) + l_names + other_names
    dept_counts = dept_counts.reindex(ordered_requesters)
    aligned_story_points = aligned_story_points.reindex(dept_counts.index).fillna(0)

    x_pos = range(len(dept_counts))

    # Display label mapping on x-axis
    def _display_requester(req: str) -> str:
        return 'other' if str(req).strip().lower() == 'external partner' else req
    display_requesters = [_display_requester(r) for r in dept_counts.index.tolist()]
    
    # Left y-axis: Count-based (black outline without fill)
    gray_color = '#808080'  # kept only for legacy text/color variables
    bars1 = ax1.bar(
        [x - 0.2 for x in x_pos],
        dept_counts.values,
        facecolor='none',
        edgecolor='black',
        linewidth=1.0,
        label='Nr. of requests',
        width=0.4,
    )
    ax1.set_xlabel('Requester', fontsize=12, labelpad=10)
    ax1.set_ylabel('Number of requests', fontsize=12, color='black', labelpad=10)
    ax1.tick_params(axis='y', labelcolor='black', pad=8)
    ax1.tick_params(axis='x', rotation=45, labelsize=9, pad=8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(display_requesters)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Add value labels on outlined bars
    for i, v in enumerate(dept_counts.values):
        ax1.text(i - 0.2, v, str(v), ha='center', va='bottom', fontsize=9, color='black')
    
    # Right y-axis: Story points
    ax2 = ax1.twinx()
    teal_color = '#008080'
    medium_blue = '#005A8C'

    def _is_external_requester(req) -> bool:
        s = str(req).strip().lower()
        return s == 'giz' or s == 'external partner'

    # Zero-SP handling for visualization:
    # each 0-SP request contributes 0.5 SP-equivalent, so 2 zero-SP requests = 1 SP.
    story_points_for_plot = aligned_story_points.copy()
    zero_sp_counts_aligned = zero_sp_counts.reindex(dept_counts.index).fillna(0).astype(int)
    story_points_for_plot = story_points_for_plot + (0.5 * zero_sp_counts_aligned)

    bar_colors = [teal_color if _is_external_requester(req) else medium_blue for req in dept_counts.index.tolist()]

    bars2 = ax2.bar(
        [x + 0.2 for x in x_pos],
        story_points_for_plot.values,
        color=bar_colors,
        edgecolor='none',
        linewidth=0.0,
        label='Resources',
        alpha=1.0,
        width=0.4,
    )
    ax2.set_ylabel('Number of Story Points', fontsize=12, color='black', labelpad=10)
    ax2.tick_params(axis='y', labelcolor='black', pad=8)
    
    # Add value labels on teal bars
    for i, req in enumerate(dept_counts.index.tolist()):
        label_color = teal_color if _is_external_requester(req) else medium_blue
        v_original = float(aligned_story_points.loc[req])
        v_plot = float(story_points_for_plot.loc[req])
        zero_count = int(zero_sp_counts_aligned.loc[req])
        if v_original == 0.0 and zero_count == 1:
            ax2.text(i + 0.2, v_plot, '< 1', ha='center', va='bottom', fontsize=9, color=label_color)
        else:
            label_text = f"{int(v_plot)}" if float(v_plot).is_integer() else f"{v_plot:.1f}"
            ax2.text(i + 0.2, v_plot, label_text, ha='center', va='bottom', fontsize=9, color=label_color)

    # Legend as requested (top-right)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='none', edgecolor='black', linewidth=1.0, label='Nr. of requests'),
        Patch(facecolor=medium_blue, edgecolor='black', label='Resources dedicated to internal teams'),
        Patch(facecolor=teal_color, edgecolor='black', label='Resources dedicated to external partners'),
    ]
    ax1.legend(handles=legend_handles, loc='upper right', fontsize=10, frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined department distribution plot saved to {output_filename}")
    return True

if dept_col and story_col:
    print(f"Using department column: '{dept_col}'")
    print(f"Using story points column: '{story_col}'")
    
    # Create chart for Q1 2026
    print("\nCreating distribution-requester chart for Q1 2026...")
    create_distribution_requester_chart(df, with_tag(f'{output_subfolder}/user-story-distr-requester-Q1_2026.png'), 'Q1 2026')
else:
    print("Warning: 'department' column not found in the data")
    print("Available columns:", df_full.columns.tolist())

# 3. Create plot for int_/ext_ classes weighted by story points (if such columns exist)
print("\n" + "="*60)
print("Creating int_/ext_ Distribution Plot (if columns available)")
print("="*60)

# Find story points column if not already found
if story_col is None:
    for col in df.columns:
        if 'story' in col.lower() and 'point' in col.lower():
            story_col = col
            break
    if not story_col:
        for col in df.columns:
            if 'point' in col.lower():
                story_col = col
                break

# Columns R–X (inclusive) in the sheet: position-based selection
# R..X correspond to indices 18..24 (1-indexed), i.e. python slice [17:24].
rx_columns = list(df.columns[17:24])
all_class_columns = [c for c in rx_columns if str(c).startswith(('int_', 'ext_'))]
int_columns = [c for c in all_class_columns if str(c).startswith('int_')]
ext_columns = [c for c in all_class_columns if str(c).startswith('ext_')]

# Display name mapping for the R–X columns in kanban-statistics-v0.xlsx
class_display_map = {
    'int_req': 'awareness raising and other',
    'int_capacity': 'capacity building',
    'int_tech_PL': 'technical: Project Locations',
    'int_tech': 'technical: other',
    'ext_req': 'geospatial data and applications',
    'ext_social_env': 'Social & Environmental Screenings',
    'ext_PA': 'Protected Areas',
}


def create_int_ext_pie_chart_no_labels(dataframe, output_filename, description):
    """Create a pie chart showing int_/ext_ class distribution weighted by story points, without labels or leader lines."""
    plot_data = []
    
    if not all_class_columns or not story_col:
        print(f"Warning: Cannot create {description} - missing required columns")
        return False
    
    # Process each ticket
    for idx, row in dataframe.iterrows():
        story_points = pd.to_numeric(row[story_col], errors='coerce')
        if pd.isna(story_points):
            continue
        # For visualization aggregation, treat 0 SP as 0.5 SP-equivalent.
        if story_points == 0:
            story_points = 0.5
        
        # Check which classes this ticket belongs to
        ticket_classes = []
        
        for col in all_class_columns:
            value = str(row[col]).strip().upper() if pd.notna(row[col]) else 'NA'
            if value == 'Y':
                ticket_classes.append(col)
        
        # Add story points to each class this ticket belongs to
        for class_name in ticket_classes:
            plot_data.append({
                'class': class_name,
                'story_points': story_points,
            })
    
    if not plot_data:
        print(f"Warning: No data found for {description}")
        return False
    
    plot_df = pd.DataFrame(plot_data)
    
    # Group by class and sum story points
    class_totals_raw = plot_df.groupby('class')['story_points'].sum()
    
    # Sort to group int_ and ext_ together: first all int_ (sorted by value), then all ext_ (sorted by value)
    int_classes = [c for c in class_totals_raw.index if c.startswith('int_')]
    ext_classes = [c for c in class_totals_raw.index if c.startswith('ext_')]
    
    # Sort within each group by value (descending)
    int_sorted = sorted(int_classes, key=lambda x: class_totals_raw[x], reverse=True)
    ext_sorted = sorted(ext_classes, key=lambda x: class_totals_raw[x], reverse=True)
    
    # Combine: int_ first, then ext_
    ordered_classes = int_sorted + ext_sorted
    class_totals = class_totals_raw[ordered_classes]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(13, 12))
    
    # Define color palettes
    # Gray shades for int_ categories (from light to dark)
    gray_palette = ['#D3D3D3', '#A9A9A9', '#808080', '#696969', '#555555', '#2F2F2F']
    # Blue shades for ext_ categories (from light to dark)
    blue_palette = ['#E6F3FF', '#B3D9FF', '#80C0FF', '#4DA6FF', '#1A8CFF', '#0066CC', '#004C99']
    
    # Separate int_ and ext_ categories and sort by value (descending)
    int_classes_chart = [c for c in class_totals.index if c.startswith('int_')]
    ext_classes_chart = [c for c in class_totals.index if c.startswith('ext_')]
    
    # Sort by value (descending) - higher values get darker shades
    int_classes_sorted = sorted(int_classes_chart, key=lambda x: class_totals[x], reverse=True)
    ext_classes_sorted = sorted(ext_classes_chart, key=lambda x: class_totals[x], reverse=True)
    
    # Create color mapping - assign darker shades to higher values
    color_map = {}
    for i, class_name in enumerate(int_classes_sorted):
        # Use darker shades (from end of palette) for higher values
        # Highest value (i=0) gets darkest shade (last index)
        palette_idx = len(gray_palette) - 1 - min(i, len(gray_palette) - 1)
        color_map[class_name] = gray_palette[palette_idx]
    for i, class_name in enumerate(ext_classes_sorted):
        # Use darker shades (from end of palette) for higher values
        # Highest value (i=0) gets darkest shade (last index)
        palette_idx = len(blue_palette) - 1 - min(i, len(blue_palette) - 1)
        color_map[class_name] = blue_palette[palette_idx]
    
    # Prepare pie chart data
    pie_colors = []
    pie_edgecolors = []
    pie_linewidths = []
    
    for class_name in class_totals.index:
        base_color = color_map[class_name]
        pie_colors.append(base_color)
        pie_edgecolors.append('none')
        pie_linewidths.append(0)
    
    # Create pie chart showing story points by class
    explode = [0.08] * len(class_totals)
    pie_radius = 0.9
    wedges, _ = ax.pie(class_totals.values,
                       labels=None,
                       colors=pie_colors,
                       startangle=90,
                       explode=explode,
                       radius=pie_radius)

    # Customize each wedge with appropriate edge color and linewidth
    for wedge, edgecolor, linewidth in zip(wedges, pie_edgecolors, pie_linewidths):
        wedge.set_edgecolor(edgecolor)
        wedge.set_linewidth(linewidth)
    
    # Add legend for internal vs external requests
    # Use representative colors from palettes
    legend_gray = gray_palette[2] if len(gray_palette) > 2 else gray_palette[0]  # Medium gray
    legend_blue = blue_palette[3] if len(blue_palette) > 3 else blue_palette[0]  # Medium blue
    legend_patches = [
        plt.Line2D([0], [0], marker='o', color='w', label='Internal requests',
                   markerfacecolor=legend_gray, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='External requests',
                   markerfacecolor=legend_blue, markersize=10)
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9, frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"int_/ext_ distribution plot (no labels) saved to {output_filename}")
    return True

def create_exploding_int_ext_rx_pie_chart(dataframe, output_filename, description):
    """
    Exploding int_/ext_ pie chart with leader lines where:
    - columns used are exactly R–X (position-based) in the spreadsheet
    - int_* are gray shades, ext_* are blue shades
    - label + leader-line placement avoids overlaps and crossings (brute-force collision checks)
    """
    if not all_class_columns or not story_col:
        print(f"Warning: Cannot create {description} - missing required columns")
        return False

    # Story-point weighted slice sizes:
    # treat each 0-SP request as 0.5 SP-equivalent.
    totals = {c: 0.0 for c in all_class_columns}
    for _, row in dataframe.iterrows():
        sp = pd.to_numeric(row.get(story_col), errors='coerce')
        if pd.isna(sp):
            continue
        if sp == 0:
            sp = 0.5
        for c in all_class_columns:
            val = str(row.get(c, '')).strip().upper()
            if val == 'Y':
                totals[c] += float(sp)

    int_classes = [c for c in totals.keys() if str(c).startswith('int_')]
    ext_classes = [c for c in totals.keys() if str(c).startswith('ext_')]
    int_sorted = sorted(int_classes, key=lambda c: totals[c], reverse=True)
    ext_sorted = sorted(ext_classes, key=lambda c: totals[c], reverse=True)
    ordered_classes = int_sorted + ext_sorted

    # Ensure all R..X categories are represented (even if total weight is 0).
    class_totals = np.array([totals[c] for c in ordered_classes], dtype=float)
    nonzero_vals = class_totals[class_totals > 0]
    if nonzero_vals.size == 0:
        print(f"Warning: No data found for {description}")
        return False
    epsilon = max(float(nonzero_vals.min()) * 1e-3, 1e-6)
    for c in ordered_classes:
        if totals[c] <= 0:
            totals[c] = epsilon
    class_totals = np.array([totals[c] for c in ordered_classes], dtype=float)
    total_weight = class_totals.sum()

    # Palettes (darker = larger share)
    gray_palette = ['#E9E9E9', '#D6D6D6', '#BDBDBD', '#A0A0A0', '#808080', '#616161', '#3F3F3F']
    blue_palette = ['#DCEBFF', '#B9D9FF', '#8EC4FF', '#5DA9FF', '#2F8DFF', '#1C6AD0', '#0E478E']

    # Create color map
    color_map = {}
    for i, c in enumerate(int_sorted):
        palette_idx = len(gray_palette) - 1 - min(i, len(gray_palette) - 1)
        color_map[c] = gray_palette[palette_idx]
    for i, c in enumerate(ext_sorted):
        palette_idx = len(blue_palette) - 1 - min(i, len(blue_palette) - 1)
        color_map[c] = blue_palette[palette_idx]

    pie_colors = [color_map[c] for c in ordered_classes]
    pie_radius = 0.95
    explode = [0.08] * len(ordered_classes)

    fig, ax = plt.subplots(figsize=(9, 9))
    wedges = ax.pie(
        class_totals,
        labels=None,
        colors=pie_colors,
        startangle=90,
        explode=explode,
        radius=pie_radius
    )[0]

    # Ensure we have enough room for leader lines and labels
    end_r = pie_radius + 0.65
    mid_r = pie_radius + 0.25
    ax.set_xlim(-end_r - 0.05, end_r + 0.25)
    ax.set_ylim(-end_r - 0.05, end_r + 0.05)
    ax.set_aspect('equal')

    # Leader-line collision helpers
    def _ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def _seg_intersect(a, b, c, d, eps=1e-9):
        # Returns True if segments AB and CD intersect (including touching, with tolerance)
        a = np.array(a, float); b = np.array(b, float); c = np.array(c, float); d = np.array(d, float)
        # Quick bbox reject
        if (max(a[0], b[0]) + eps < min(c[0], d[0])) or (max(c[0], d[0]) + eps < min(a[0], b[0])):
            return False
        if (max(a[1], b[1]) + eps < min(c[1], d[1])) or (max(c[1], d[1]) + eps < min(a[1], b[1])):
            return False
        return (_ccw(a, c, d) != _ccw(b, c, d)) and (_ccw(a, b, c) != _ccw(a, b, d))

    def _angle_in_wedge(theta, w_theta1, w_theta2):
        # theta, w_* in [0, 2pi)
        if w_theta1 <= w_theta2:
            return w_theta1 <= theta <= w_theta2
        return theta >= w_theta1 or theta <= w_theta2

    wedges_info = []
    for idx, w in enumerate(wedges):
        t1 = np.deg2rad(w.theta1) % (2 * np.pi)
        t2 = np.deg2rad(w.theta2) % (2 * np.pi)
        mid_angle = np.deg2rad((w.theta1 + w.theta2) / 2.0) % (2 * np.pi)
        x = float(np.cos(mid_angle))
        y = float(np.sin(mid_angle))
        wedges_info.append({
            'idx': idx,
            'class': ordered_classes[idx],
            't1': t1,
            't2': t2,
            'mid_angle': mid_angle,
            'x': x,
            'y': y,
        })

    label_fontsize = 9
    # Increase gap to keep leader-line labels collision-free at larger font sizes.
    label_gap = 0.28  # vertical spacing in plot coordinates
    leader_color = '#808080'

    placed_segments = []  # list of ((x1,y1),(x2,y2)) segments already drawn
    placed_labels = []     # list of dicts with side + y for overlap checks

    # Build initial label ordering per side (top-to-bottom)
    right_candidates = sorted([i for i, wi in enumerate(wedges_info) if wi['x'] >= 0], key=lambda i: wedges_info[i]['y'], reverse=True)
    left_candidates = sorted([i for i, wi in enumerate(wedges_info) if wi['x'] < 0], key=lambda i: wedges_info[i]['y'], reverse=True)

    # Place each label with brute-force candidate y search
    def try_place_label(candidate_idx, side_list, existing_y_list):
        wi = wedges_info[candidate_idx]
        cls = wi['class']
        is_right = wi['x'] >= 0

        natural_y = mid_r * wi['y']
        x_sign = 1.0 if is_right else -1.0
        end_x = x_sign * end_r
        mid_x = mid_r * wi['x']
        start_x = pie_radius * wi['x']
        start_y = pie_radius * wi['y']

        # Display + percent text
        # Keep original column name for the slice label
        display_name = class_display_map.get(cls, str(cls))
        pct = (totals[cls] / total_weight) * 100.0 if total_weight else 0.0
        # If slice share is exactly 0.0%, do not place a label.
        # (Avoid skipping small-but-nonzero slices that round to 0.0%.)
        if pct == 0.0:
            return False
        # Percent formatting:
        # - Round up (ceiling) to an integer percent.
        # - For values under 1%, show "< 1%".
        if pct < 1.0:
            percent_text = "< 1%"
        else:
            percent_up_int = int(np.ceil(pct))
            percent_text = f"{percent_up_int}%"
        label_text = f"{display_name}\n({percent_text})"
        text_offset = 0.02
        text_x = end_x + (text_offset if is_right else -text_offset)

        # Candidate y values: natural_y, then progressively farther
        offsets = [0.0]
        for k in range(1, 10):
            offsets.extend([k * label_gap, -k * label_gap])
        candidates_y = [natural_y + off for off in offsets]

        # Clamp to plot region
        candidates_y = [y for y in candidates_y if (-end_r + 0.05) <= y <= (end_r - 0.05)]

        for y_label in candidates_y:
            # Check overlap with labels on same side (fast y-gate)
            if any(abs(y_label - existing) < label_gap * 0.85 for existing in existing_y_list):
                continue

            # Segments:
            #   seg1: start -> (mid_x, y_label)
            #   seg2: (mid_x, y_label) -> (end_x, y_label)
            seg1 = ((start_x, start_y), (mid_x, y_label))
            seg2 = ((mid_x, y_label), (end_x, y_label))

            # Avoid leader diagonal crossing other wedges' interior
            theta_check_start = np.arctan2(seg1[0][1], seg1[0][0])
            # sample points along seg1 (diagonal)
            crosses_other_wedge = False
            for t in np.linspace(0.05, 0.95, 18):
                px = seg1[0][0] + t * (seg1[1][0] - seg1[0][0])
                py = seg1[0][1] + t * (seg1[1][1] - seg1[0][1])
                dist = np.hypot(px, py)
                if dist < pie_radius * 0.995:
                    theta = np.arctan2(py, px) % (2 * np.pi)
                    for other in wedges_info:
                        if other['idx'] == wi['idx']:
                            continue
                        if _angle_in_wedge(theta, other['t1'], other['t2']):
                            crosses_other_wedge = True
                            break
                if crosses_other_wedge:
                    break
            if crosses_other_wedge:
                continue

            # Check leader-line crossings with already placed leaders
            collision = False
            for ex_seg in placed_segments:
                # If new segments intersect any existing segment, reject
                if _seg_intersect(seg1[0], seg1[1], ex_seg[0], ex_seg[1]):
                    collision = True
                    break
                if _seg_intersect(seg2[0], seg2[1], ex_seg[0], ex_seg[1]):
                    collision = True
                    break
            if collision:
                continue

            # Accept this y_label; draw leader lines + text
            ax.plot([seg1[0][0], seg1[1][0]], [seg1[0][1], seg1[1][1]], color=leader_color, linewidth=0.7)
            ax.plot([seg2[0][0], seg2[1][0]], [seg2[0][1], seg2[1][1]], color=leader_color, linewidth=0.7)
            ax.text(text_x, y_label, label_text, ha=('left' if is_right else 'right'), va='center', fontsize=label_fontsize, color='black')

            placed_segments.append(seg1)
            placed_segments.append(seg2)
            placed_labels.append({'side': 'right' if is_right else 'left', 'y': y_label})
            existing_y_list.append(y_label)
            return True

        return False

    # Place right side then left side (or vice versa); both use the same collision checks)
    right_existing_y = []
    for i in right_candidates:
        try_place_label(i, right_candidates, right_existing_y)

    left_existing_y = []
    for i in left_candidates:
        try_place_label(i, left_candidates, left_existing_y)

    # Legend: internal vs external requests
    legend_gray = color_map[int_sorted[0]] if int_sorted else '#808080'
    legend_blue = color_map[ext_sorted[0]] if ext_sorted else '#2F8DFF'
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Internal requests',
                   markerfacecolor=legend_gray, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='External requests',
                   markerfacecolor=legend_blue, markersize=10),
    ]
    ax.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"int_/ext_ exploding pie chart saved to {output_filename}")
    return True

def create_eo_involvement_by_theme_chart(dataframe, output_filename, description):
    """
    New figure:
    - Earth observation involvement comes from columns M–Q (rs_*)
    - Linked to the theme in column L
    - Only plotted if any rs_* column has 'Y' for this dataset
    """
    theme_col = 'theme' if 'theme' in dataframe.columns else None
    if not theme_col:
        print(f"Warning: Cannot create {description} - missing 'theme' column")
        return False

    # Columns M–Q in the sheet are indices 13..17 (1-indexed), python slice [12:17]
    # Exclude columns M and N (rs_UP42-GeoD and rs_BKG) from the analysis.
    rs_cols = list(dataframe.columns[12:17])
    rs_cols = [c for c in rs_cols if str(c).startswith('rs_')]
    rs_cols = [c for c in rs_cols if c not in {'rs_UP42-GeoD', 'rs_BKG'}]
    if not rs_cols:
        print(f"Warning: Cannot create {description} - missing rs_* columns in M–Q")
        return False

    rs_norm = dataframe[rs_cols].fillna('NA').astype(str).apply(lambda s: s.str.strip().str.upper())
    mask_any_rs = (rs_norm == 'Y').any(axis=1)
    if mask_any_rs.sum() == 0:
        print(f"Warning: Skipping {description} - no rs_* values marked 'Y'")
        return False

    df_eo = dataframe.loc[mask_any_rs, [theme_col] + rs_cols].copy()

    # Merge/relabel selected theme buckets before counting.
    def _normalize_theme_label(theme_value):
        t = str(theme_value).strip().lower()
        if t in {'map - dynamic - custom', 'map - dynamic - odp'}:
            return 'EO data visualization: maps & dashboards'
        if t == 'status - change detection':
            return 'EO data applications: change detection & continuous monitoring'
        if t == 'data and methods':
            return 'EO data & processing methods'
        return str(theme_value).strip()

    df_eo[theme_col] = df_eo[theme_col].apply(_normalize_theme_label)
    # Count tickets per theme per rs_* source
    counts = {}
    for c in rs_cols:
        counts[c] = (df_eo[c].fillna('NA').astype(str).str.strip().str.upper() == 'Y').groupby(df_eo[theme_col]).sum()
    counts_df = pd.DataFrame(counts).fillna(0).astype(int)
    counts_df['total'] = counts_df.sum(axis=1)
    counts_df = counts_df.sort_values('total', ascending=False)
    counts_df = counts_df.drop(columns=['total'])

    themes = counts_df.index.tolist()
    matrix = counts_df[rs_cols].values

    fig, ax = plt.subplots(figsize=(14, max(6, 0.45 * len(themes))))
    im = ax.imshow(matrix, aspect='auto', cmap='Blues')

    # Ticks + labels
    rs_label_map = {
        'rs_open': 'open sources',
        'rs_comm': 'commercial sources',
        'rs_hybrid': 'hybrid sources',
    }
    ax.set_xticks(range(len(rs_cols)))
    rs_display = [rs_label_map.get(c, c) for c in rs_cols]
    ax.set_xticklabels(rs_display, fontsize=9, rotation=30, ha='right')
    ax.set_yticks(range(len(themes)))
    y_display = []
    for t in themes:
        if ': ' in t:
            label = t.replace(': ', ':\n', 1)
        elif ':' in t:
            label = t.replace(':', ':\n', 1)
        else:
            label = t
        if 'change detection &' in label:
            label = label.replace(' & ', ' &\n', 1)
        y_display.append(label)
    ax.set_yticklabels(y_display, fontsize=9)
    ax.yaxis.tick_right()
    ax.tick_params(axis='y', labelright=True, labelleft=False)
    for tick in ax.get_yticklabels():
        # Place labels to the right of the plot area, left-aligned for readability.
        tick.set_horizontalalignment('left')
        tick.set_multialignment('left')
    ax.tick_params(axis='y', pad=5)

    # Annotate counts in cells
    for y in range(len(themes)):
        for x in range(len(rs_cols)):
            val = matrix[y, x]
            if val > 0:
                rgba = im.cmap(im.norm(val))
                # Relative luminance to keep text readable on dark fills.
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                txt_color = '#D3D3D3' if luminance < 0.5 else 'black'
                ax.text(x, y, str(val), ha='center', va='center', fontsize=9, color=txt_color)

    ax.set_title('')

    # Reserve extra right space for right-side y-axis labels.
    fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"EO involvement figure saved to {output_filename}")
    return True

def create_user_story_summary_report(dataframe, output_filename, description):
    """Create the Q1 summary text report with current agreed labels."""
    if not story_col:
        print(f"Warning: Cannot create {description} - story points column not found")
        return False

    df_report = dataframe.copy()
    df_report[story_col] = pd.to_numeric(df_report[story_col], errors='coerce').fillna(0).astype(int)

    for col in ['type_issue', 'type_enhancement']:
        if col not in df_report.columns:
            df_report[col] = False
        else:
            df_report[col] = (
                df_report[col]
                .fillna('NA')
                .astype(str)
                .str.strip()
                .str.upper()
                .eq('Y')
            )

    # Keep this mapping aligned with the agreed handover/reporting convention:
    # fixed hours for non-zero SP values; range only for 0-SP requests.
    fixed_hours = {1: 4, 2: 8, 3: 20, 5: 36, 8: 64, 13: 104}

    def _hours_range(series):
        low = 0
        high = 0
        for v in series:
            iv = int(v)
            if iv == 0:
                low += 2
                high += 3
            else:
                h = fixed_hours.get(iv, iv * 4)
                low += h
                high += h
        return low, high

    total_requests = int(len(df_report))
    total_sp = int(df_report[story_col].sum())
    total_low, total_high = _hours_range(df_report[story_col].tolist())

    issue_df = df_report[df_report['type_issue']]
    enh_df = df_report[df_report['type_enhancement']]
    existing_df = df_report[(~df_report['type_issue']) & (~df_report['type_enhancement'])]

    issue_requests = int(len(issue_df))
    enh_requests = int(len(enh_df))
    existing_requests = int(len(existing_df))

    issue_sp = int(issue_df[story_col].sum())
    enh_sp = int(enh_df[story_col].sum())
    existing_sp = int(existing_df[story_col].sum())

    issue_low, issue_high = _hours_range(issue_df[story_col].tolist())
    enh_low, enh_high = _hours_range(enh_df[story_col].tolist())
    existing_low, existing_high = _hours_range(existing_df[story_col].tolist())

    report_lines = [
        'Q1 2026 User Story Summary',
        'Source file: kanban-statistics-v0.xlsx',
        f'Filter: {_scope_filter_text()}',
        '',
        '1) Overall totals',
        f'- Total requests: {total_requests}',
        f'- Total story points (SP): {total_sp} ({total_low}-{total_high} hours)',
        f'- Estimated consultancy effort (all requests): {total_low}-{total_high} hours',
        '',
        '2) Resource allocation',
        '- Resolving issues (type_issue)',
        f'  - Number of requests: {issue_requests}',
        f'  - SP (consultancy-hours): {issue_sp} ({issue_low}-{issue_high} hours)',
        '',
        '- Enhancements or code customizations (type_enhancement)',
        f'  - Number of requests: {enh_requests}',
        f'  - SP (consultancy-hours): {enh_sp} ({enh_low}-{enh_high} hours)',
        '',
        '- Fulfilling requests with existing features',
        f'  - Number of requests: {existing_requests}',
        f'  - SP (consultancy-hours): {existing_sp} ({existing_low}-{existing_high} hours)',
        '',
        '3) SP to consultancy-hours reference used',
        '- 0 SP: 2-3 hours',
        '- 1 SP: 4 hours',
        '- 2 SP: 8 hours (1 day)',
        '- 3 SP: 2-3 days',
        '- 5 SP: 4-5 days',
        '- 8 SP: 6-10 days',
        '- 13 SP: 11-15 days',
        '',
        'Assumption for day-based ranges: 1 day = 8 hours',
    ]

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines) + '\n')

    print(f"User story summary report saved to {output_filename}")
    return True

def create_eo_use_cases_report(dataframe, output_filename, description):
    """
    Create a text report of EO use cases for rs_comm / rs_hybrid rows and whether
    those cases also used rs_open, with sub-categories for rs_UP42-GeoD and rs_BKG.
    """
    required_cols = ['rs_comm', 'rs_hybrid', 'rs_open', 'rs_UP42-GeoD', 'rs_BKG']
    missing = [c for c in required_cols if c not in dataframe.columns]
    if missing:
        print(f"Warning: Cannot create {description} - missing columns: {missing}")
        return False

    # Columns requested by user: F and H..L (position-based from sheet).
    # 0-based indexes: F->5, H..L->7..11.
    citation_idxs = [5, 7, 8, 9, 10, 11]
    citation_cols = [dataframe.columns[i] for i in citation_idxs if i < len(dataframe.columns)]

    def _is_y(series):
        return series.fillna('NA').astype(str).str.strip().str.upper().eq('Y')

    df_rep = dataframe.copy()
    df_rep['__rs_comm'] = _is_y(df_rep['rs_comm'])
    df_rep['__rs_hybrid'] = _is_y(df_rep['rs_hybrid'])
    df_rep['__rs_open'] = _is_y(df_rep['rs_open'])
    df_rep['__rs_geod'] = _is_y(df_rep['rs_UP42-GeoD'])
    df_rep['__rs_bkg'] = _is_y(df_rep['rs_BKG'])

    use_case_mask = df_rep['__rs_open'] | df_rep['__rs_comm'] | df_rep['__rs_hybrid']
    eo_cases = df_rep.loc[use_case_mask].copy()

    if eo_cases.empty:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join([
                'EO Use Cases Report',
                f'Source file: {file_path}',
                f'Filter: {_scope_filter_text()}',
                '',
                'No rs_open/rs_comm/rs_hybrid cases found for the selected scope.',
                ''
            ]))
        print(f"EO use-case report saved to {output_filename}")
        return True

    def _acq_bucket(row):
        if row['__rs_geod'] and row['__rs_bkg']:
            return 'UP42-PlanetLabs + BKG'
        if row['__rs_geod']:
            return 'UP42-PlanetLabs only'
        if row['__rs_bkg']:
            return 'BKG only'
        return 'Other'

    eo_cases['acquisition_bucket'] = eo_cases.apply(_acq_bucket, axis=1)

    bucket_order = [
        'UP42-PlanetLabs + BKG',
        'UP42-PlanetLabs only',
        'BKG only',
        'Other',
    ]

    lines = [
        'EO Use Cases Report',
        f'Source file: {file_path}',
        f'Filter: {_scope_filter_text()}',
        '',
        'Selection logic:',
        '- Included rows where rs_open == Y OR rs_comm == Y OR rs_hybrid == Y',
        '- Sub-categorized by rs_UPP42-GeoD and rs_BKG',
        '- Also indicates whether rs_open == Y',
        '',
        'Columns cited per case: F, H, I, J, K, L',
        f"- Mapped column names: {', '.join(citation_cols)}",
        '',
        f'Total EO use cases (rs_open/rs_comm/rs_hybrid): {len(eo_cases)}',
        ''
    ]

    lines.append('Summary by acquisition bucket')
    for bucket in bucket_order:
        part = eo_cases[eo_cases['acquisition_bucket'] == bucket]
        if part.empty:
            lines.append(f'- {bucket}: 0')
            continue
        open_yes = int(part['__rs_open'].sum())
        open_no = int((~part['__rs_open']).sum())
        lines.append(f'- {bucket}: {len(part)} (open data used: {open_yes}; open data not used: {open_no})')

    lines.append('')
    lines.append('Detailed cases')

    for bucket in bucket_order:
        part = eo_cases[eo_cases['acquisition_bucket'] == bucket]
        if part.empty:
            continue
        lines.append(f'')
        lines.append(f'{bucket}')
        for _, row in part.iterrows():
            cited_values = [str(row.get(c, '')) for c in citation_cols]
            comm_hybrid = []
            if row['__rs_comm']:
                comm_hybrid.append('rs_comm')
            if row['__rs_hybrid']:
                comm_hybrid.append('rs_hybrid')
            comm_hybrid_text = '+'.join(comm_hybrid) if comm_hybrid else 'n/a'
            lines.append(
                f"- [{', '.join(cited_values)}] | class={comm_hybrid_text} | "
                f"UP42-PlanetLabs={'Y' if row['__rs_geod'] else 'N'} | "
                f"BKG={'Y' if row['__rs_bkg'] else 'N'} | "
                f"open={'Y' if row['__rs_open'] else 'N'}"
            )

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"EO use-case report saved to {output_filename}")
    return True

# Create pie charts for both full year and Q3&Q4
plot_data = []  # Initialize for final print statement
if all_class_columns and story_col:
    print(f"Found {len(int_columns)} int_ columns and {len(ext_columns)} ext_ columns")
    print(f"Using story points column: '{story_col}'")
    
    # Create pie charts for Q1 2026
    print("\nCreating pie charts for Q1 2026...")
    create_exploding_int_ext_rx_pie_chart(
        df,
        with_tag(f'{output_subfolder}/user-story-distr-request-type-Q1_2026.png'),
        'Q1 2026',
    )
    create_int_ext_pie_chart_no_labels(
        df,
        with_tag(f'{output_subfolder}/user-story-distr-request-type-no-label-Q1_2026.png'),
        'Q1 2026',
    )

    # Earth-observation heatmap and summary report are part of the same
    # scoped run so outputs stay synchronized for handover.
    create_eo_involvement_by_theme_chart(
        df,
        with_tag(f'{output_subfolder}/user-story-distr-eo-data-use-Q1_2026.png'),
        'Q1 2026',
    )
    create_user_story_summary_report(
        df,
        with_tag(f'{output_subfolder}/user-story-summary-q1_2026.txt'),
        'Q1 2026 summary report',
    )
    create_eo_use_cases_report(
        df,
        with_tag(f'{output_subfolder}/user-story-eo-use-cases-q1_2026.txt'),
        'Q1 2026 EO use-cases report',
    )
    
    # Track plot_data for final print statement
    for idx, row in df.iterrows():
        story_points = pd.to_numeric(row[story_col], errors='coerce')
        if pd.isna(story_points) or story_points == 0:
            continue
        for col in all_class_columns:
            value = str(row[col]).strip().upper() if pd.notna(row[col]) else 'NA'
            if value == 'Y':
                plot_data.append({'class': col})
else:
    if not all_class_columns:
        print("Warning: No columns starting with 'int_' or 'ext_' found")
    if not story_col:
        print("Warning: Story points column not found")

print("\n" + "="*60)
print("Skipping Q1&Q2 vs Q3&Q4 comparison chart for Q1 2026 run.")
print("="*60)

print("\n" + "="*60)
print("Visualization Complete!")
print("="*60)
print("\nGenerated files:")
print(f"  - {with_tag(f'{output_subfolder}/user-story-distr-deliverables-Q1_2026.png')}")
if dept_col and story_col:
    print(f"  - {with_tag(f'{output_subfolder}/user-story-distr-requester-Q1_2026.png')}")
if all_class_columns and story_col and len(plot_data) > 0:
    print(f"  - {with_tag(f'{output_subfolder}/user-story-distr-request-type-Q1_2026.png')}")
    print(f"  - {with_tag(f'{output_subfolder}/user-story-distr-request-type-no-label-Q1_2026.png')}")
    print(f"  - {with_tag(f'{output_subfolder}/user-story-distr-eo-data-use-Q1_2026.png')}")
    print(f"  - {with_tag(f'{output_subfolder}/user-story-summary-q1_2026.txt')}")
    print(f"  - {with_tag(f'{output_subfolder}/user-story-eo-use-cases-q1_2026.txt')}")

